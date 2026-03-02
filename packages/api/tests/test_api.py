"""
Tests for the SnowClaw API — upload, status, results, and error cases.

Uses:
  - ``starlette.testclient.TestClient`` for synchronous HTTP + WebSocket tests
  - Mocked Celery task (no Redis/worker dependency)
  - In-memory ``JobStore`` (no external state)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from api.app import app
from api.config import JobStatus, PipelineStage, job_store, settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_job_store():
    """Ensure the job store is empty before and after each test."""
    job_store.clear()
    yield
    job_store.clear()


@pytest.fixture()
def tmp_results(tmp_path: Path):
    """Point settings.RESULTS_DIR to a temporary directory for the test."""
    original = settings.RESULTS_DIR
    settings.RESULTS_DIR = tmp_path / "results"
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    yield settings.RESULTS_DIR
    settings.RESULTS_DIR = original


@pytest.fixture()
def tmp_uploads(tmp_path: Path):
    """Point settings.UPLOAD_DIR to a temporary directory for the test."""
    original = settings.UPLOAD_DIR
    settings.UPLOAD_DIR = tmp_path / "uploads"
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield settings.UPLOAD_DIR
    settings.UPLOAD_DIR = original


@pytest.fixture()
def client(tmp_uploads, tmp_results):
    """Provide a Starlette TestClient bound to the FastAPI app."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_completed_job(
    job_id: str,
    results_dir: Path,
) -> None:
    """Insert a completed job with result files on disk."""
    job_store.create(job_id, "/fake/input.mp4")
    job_store.update(
        job_id,
        status=JobStatus.COMPLETED,
        stage=PipelineStage.SAVING_RESULTS,
        progress=100,
        completed_at=time.time(),
    )

    # Write stub result files
    job_dir = results_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    (job_dir / "annotated.mp4").write_bytes(b"\x00\x00\x00\x1cftypisom")
    (job_dir / "poses.json").write_text(json.dumps([
        {
            "frame_idx": 0,
            "timestamp_s": 0.0,
            "pose": {
                "head": [0, 1.7, 0],
                "neck": [0, 1.5, 0],
                "left_shoulder": [-0.2, 1.4, 0],
                "right_shoulder": [0.2, 1.4, 0],
                "left_elbow": [-0.3, 1.1, 0],
                "right_elbow": [0.3, 1.1, 0],
                "left_wrist": [-0.3, 0.9, 0],
                "right_wrist": [0.3, 0.9, 0],
                "left_hip": [-0.1, 0.9, 0],
                "right_hip": [0.1, 0.9, 0],
                "left_knee": [-0.1, 0.5, 0],
                "right_knee": [0.1, 0.5, 0],
                "left_ankle": [-0.1, 0.05, 0],
                "right_ankle": [0.1, 0.05, 0],
            },
            "metrics": {
                "left_knee_deg": 170.0,
                "right_knee_deg": 170.0,
                "inclination_deg": 5.0,
                "com_height_pct": 0.55,
            },
        }
    ]))


# ---------------------------------------------------------------------------
# Upload tests
# ---------------------------------------------------------------------------


class TestUpload:
    """Tests for POST /api/upload."""

    def test_upload_returns_job_id(self, client: TestClient):
        """Successful upload returns 202 with job_id and status 'queued'."""
        with patch("api.tasks.process_video") as mock_task:
            mock_task.delay = MagicMock()

            resp = client.post(
                "/api/upload",
                files={"file": ("test.mp4", b"fake-video-bytes", "video/mp4")},
            )

        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "queued"

        # Verify Celery task was dispatched
        mock_task.delay.assert_called_once()
        call_args = mock_task.delay.call_args
        assert call_args[0][0] == body["job_id"]  # job_id arg

    def test_upload_unsupported_extension(self, client: TestClient):
        """Uploading a non-video file extension returns 400."""
        resp = client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"not a video", "text/plain")},
        )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_upload_no_filename(self, client: TestClient):
        """Upload with an empty filename is rejected (422 from FastAPI validation)."""
        resp = client.post(
            "/api/upload",
            files={"file": ("", b"data", "video/mp4")},
        )
        assert resp.status_code == 422

    def test_upload_size_limit(self, client: TestClient):
        """Upload exceeding the max size returns 413."""
        original_max = settings.MAX_UPLOAD_SIZE_MB
        settings.MAX_UPLOAD_SIZE_MB = 0  # 0 MB = effectively no upload allowed

        try:
            with patch("api.tasks.process_video") as mock_task:
                mock_task.delay = MagicMock()
                resp = client.post(
                    "/api/upload",
                    files={"file": ("big.mp4", b"x" * 1024, "video/mp4")},
                )
            assert resp.status_code == 413
            assert "exceeds maximum size" in resp.json()["detail"]
        finally:
            settings.MAX_UPLOAD_SIZE_MB = original_max

    def test_upload_creates_job_in_store(self, client: TestClient):
        """Upload creates a job record in the job store."""
        with patch("api.tasks.process_video") as mock_task:
            mock_task.delay = MagicMock()
            resp = client.post(
                "/api/upload",
                files={"file": ("run.mov", b"fake", "video/quicktime")},
            )

        job_id = resp.json()["job_id"]
        job = job_store.get(job_id)
        assert job is not None
        assert job.status == JobStatus.QUEUED
        assert job.input_path is not None


# ---------------------------------------------------------------------------
# Results — video endpoint tests
# ---------------------------------------------------------------------------


class TestResultVideo:
    """Tests for GET /api/results/{job_id}/video."""

    def test_get_video_success(self, client: TestClient, tmp_results: Path):
        """Completed job returns 200 with the annotated video."""
        job_id = "test-video-ok"
        _create_completed_job(job_id, tmp_results)

        resp = client.get(f"/api/results/{job_id}/video")
        assert resp.status_code == 200
        assert "video/mp4" in resp.headers["content-type"]

    def test_get_video_not_found(self, client: TestClient):
        """Non-existent job returns 404."""
        resp = client.get("/api/results/nonexistent/video")
        assert resp.status_code == 404

    def test_get_video_not_completed(self, client: TestClient):
        """Queued (non-completed) job returns 409."""
        job_id = "test-video-pending"
        job_store.create(job_id, "/fake/input.mp4")

        resp = client.get(f"/api/results/{job_id}/video")
        assert resp.status_code == 409
        assert "not completed" in resp.json()["detail"]

    def test_get_video_file_missing(self, client: TestClient, tmp_results: Path):
        """Completed job but missing file on disk returns 404."""
        job_id = "test-video-missing-file"
        job_store.create(job_id, "/fake/input.mp4")
        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=time.time(),
        )
        # No files written to disk

        resp = client.get(f"/api/results/{job_id}/video")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Results — poses endpoint tests
# ---------------------------------------------------------------------------


class TestResultPoses:
    """Tests for GET /api/results/{job_id}/poses."""

    def test_get_poses_success(self, client: TestClient, tmp_results: Path):
        """Completed job returns 200 with per-frame pose JSON."""
        job_id = "test-poses-ok"
        _create_completed_job(job_id, tmp_results)

        resp = client.get(f"/api/results/{job_id}/poses")
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["frame_idx"] == 0
        assert "pose" in data[0]
        assert "metrics" in data[0]

    def test_get_poses_not_found(self, client: TestClient):
        """Non-existent job returns 404."""
        resp = client.get("/api/results/nonexistent/poses")
        assert resp.status_code == 404

    def test_get_poses_not_completed(self, client: TestClient):
        """Job in processing state returns 409."""
        job_id = "test-poses-processing"
        job_store.create(job_id, "/fake/input.mp4")
        job_store.update(job_id, status=JobStatus.PROCESSING)

        resp = client.get(f"/api/results/{job_id}/poses")
        assert resp.status_code == 409

    def test_get_poses_file_missing(self, client: TestClient, tmp_results: Path):
        """Completed job but missing poses.json returns 404."""
        job_id = "test-poses-missing"
        job_store.create(job_id, "/fake/input.mp4")
        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=time.time(),
        )

        resp = client.get(f"/api/results/{job_id}/poses")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# WebSocket status tests
# ---------------------------------------------------------------------------


class TestWebSocketStatus:
    """Tests for WS /api/ws/status/{job_id} using Starlette TestClient."""

    def test_ws_nonexistent_job(self, client: TestClient):
        """WebSocket for a non-existent job receives an error and closes."""
        with client.websocket_connect("/api/ws/status/nope") as ws:
            msg = ws.receive_json()
            assert msg["error"] == "Job not found"

    def test_ws_completed_job_sends_final(self, client: TestClient, tmp_results: Path):
        """WebSocket for a completed job sends one message and closes."""
        job_id = "test-ws-done"
        _create_completed_job(job_id, tmp_results)

        with client.websocket_connect(f"/api/ws/status/{job_id}") as ws:
            msg = ws.receive_json()
            assert msg["status"] == "completed"
            assert msg["progress"] == 100

    def test_ws_queued_job_reports_status(self, client: TestClient):
        """WebSocket for a queued job reports queued status."""
        job_id = "test-ws-queued"
        job_store.create(job_id, "/fake/input.mp4")

        # Pre-set the job to completed so the WS loop terminates after sending
        # both the queued message and the completed message.
        # We first connect, read the queued state, then update and read again.
        import threading

        def _mark_complete():
            time.sleep(0.8)
            job_store.update(
                job_id,
                status=JobStatus.COMPLETED,
                stage=PipelineStage.SAVING_RESULTS,
                progress=100,
                completed_at=time.time(),
            )

        t = threading.Thread(target=_mark_complete)
        t.start()

        messages = []
        with client.websocket_connect(f"/api/ws/status/{job_id}") as ws:
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg.get("status") in ("completed", "failed"):
                    break

        t.join(timeout=5)

        assert len(messages) >= 2
        assert messages[0]["status"] == "queued"
        assert messages[-1]["status"] == "completed"

    def test_ws_failed_job_reports_error(self, client: TestClient):
        """WebSocket for a failed job includes the error message."""
        job_id = "test-ws-failed"
        job_store.create(job_id, "/fake/input.mp4")
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            error="Model not found",
        )

        with client.websocket_connect(f"/api/ws/status/{job_id}") as ws:
            msg = ws.receive_json()
            assert msg["status"] == "failed"
            assert msg["error"] == "Model not found"


# ---------------------------------------------------------------------------
# Job store unit tests
# ---------------------------------------------------------------------------


class TestJobStore:
    """Unit tests for the in-memory JobStore."""

    def test_create_and_get(self):
        job = job_store.create("j1", "/path/to/video.mp4")
        assert job.job_id == "j1"
        assert job.status == JobStatus.QUEUED

        retrieved = job_store.get("j1")
        assert retrieved is not None
        assert retrieved.job_id == "j1"

    def test_get_nonexistent(self):
        assert job_store.get("nonexistent") is None

    def test_update(self):
        job_store.create("j2", "/path/to/video.mp4")
        updated = job_store.update(
            "j2",
            status=JobStatus.PROCESSING,
            stage=PipelineStage.POSE_3D,
            progress=42.5,
        )
        assert updated is not None
        assert updated.status == JobStatus.PROCESSING
        assert updated.stage == PipelineStage.POSE_3D
        assert updated.progress == 42.5

    def test_update_nonexistent(self):
        assert job_store.update("nope", status=JobStatus.FAILED) is None

    def test_delete(self):
        job_store.create("j3", "/path")
        assert job_store.delete("j3") is True
        assert job_store.get("j3") is None
        assert job_store.delete("j3") is False

    def test_all_jobs(self):
        job_store.create("a", "/a")
        job_store.create("b", "/b")
        jobs = job_store.all_jobs()
        ids = {j.job_id for j in jobs}
        assert ids == {"a", "b"}

    def test_clear(self):
        job_store.create("c", "/c")
        job_store.clear()
        assert job_store.all_jobs() == []


# ---------------------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------------------


class TestCleanup:
    """Tests for result cleanup logic."""

    def test_cleanup_removes_expired_jobs(self, tmp_path: Path):
        """Expired completed jobs are removed by cleanup."""
        original_results = settings.RESULTS_DIR
        settings.RESULTS_DIR = tmp_path / "results"
        settings.RESULTS_DIR.mkdir()

        try:
            from api.app import _cleanup_expired_results

            job_id = "expired-job"
            job_store.create(job_id, "/fake/input.mp4")

            # Set completed_at to well past retention
            past_time = time.time() - settings.RESULT_RETENTION_SECONDS - 3600
            job_store.update(
                job_id,
                status=JobStatus.COMPLETED,
                completed_at=past_time,
            )

            # Create result directory
            result_dir = settings.RESULTS_DIR / job_id
            result_dir.mkdir()
            (result_dir / "annotated.mp4").write_bytes(b"data")

            _cleanup_expired_results()

            assert job_store.get(job_id) is None
            assert not result_dir.exists()
        finally:
            settings.RESULTS_DIR = original_results

    def test_cleanup_keeps_recent_jobs(self, tmp_path: Path):
        """Recent completed jobs are NOT removed by cleanup."""
        original_results = settings.RESULTS_DIR
        settings.RESULTS_DIR = tmp_path / "results"
        settings.RESULTS_DIR.mkdir()

        try:
            from api.app import _cleanup_expired_results

            job_id = "recent-job"
            job_store.create(job_id, "/fake/input.mp4")
            job_store.update(
                job_id,
                status=JobStatus.COMPLETED,
                completed_at=time.time(),  # Just now
            )

            _cleanup_expired_results()

            assert job_store.get(job_id) is not None
        finally:
            settings.RESULTS_DIR = original_results


# ---------------------------------------------------------------------------
# Error / edge-case tests
# ---------------------------------------------------------------------------


class TestErrorCases:
    """Miscellaneous error and edge-case tests."""

    def test_upload_no_file(self, client: TestClient):
        """POST /api/upload with no file returns 422."""
        resp = client.post("/api/upload")
        assert resp.status_code == 422

    def test_results_failed_job(self, client: TestClient):
        """Failed job returns 409 for both video and poses endpoints."""
        job_id = "test-failed"
        job_store.create(job_id, "/fake/input.mp4")
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            error="Pipeline exploded",
        )

        resp_video = client.get(f"/api/results/{job_id}/video")
        assert resp_video.status_code == 409

        resp_poses = client.get(f"/api/results/{job_id}/poses")
        assert resp_poses.status_code == 409

    def test_upload_multiple_allowed_extensions(self, client: TestClient):
        """All allowed video extensions are accepted."""
        for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            with patch("api.tasks.process_video") as mock_task:
                mock_task.delay = MagicMock()
                resp = client.post(
                    "/api/upload",
                    files={"file": (f"video{ext}", b"data", "video/mp4")},
                )
            assert resp.status_code == 202, f"Expected 202 for {ext}, got {resp.status_code}"
