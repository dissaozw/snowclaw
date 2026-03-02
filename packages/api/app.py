"""
FastAPI application for the SnowClaw API server.

Endpoints:
    POST /api/upload                  — Upload a video and start processing.
    GET  /api/ws/status/{job_id}      — WebSocket for live progress updates.
    GET  /api/results/{job_id}/video  — Serve the annotated video file.
    GET  /api/results/{job_id}/poses  — Serve per-frame Pose3D JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from api.config import (
    JobStatus,
    job_store,
    settings,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — setup and teardown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create required directories on startup; clean up expired results on shutdown."""
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Start background cleanup task
    cleanup_task = asyncio.create_task(_periodic_cleanup())

    yield

    # Shutdown: cancel the cleanup loop
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="SnowClaw API",
    description="AI skiing & snowboarding video analysis pipeline",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Background cleanup
# ---------------------------------------------------------------------------

async def _periodic_cleanup() -> None:
    """Periodically remove expired results based on retention setting."""
    while True:
        try:
            await asyncio.sleep(600)  # Check every 10 minutes
            _cleanup_expired_results()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error in periodic cleanup")


def _cleanup_expired_results() -> None:
    """Delete result directories and job records older than the retention period."""
    now = time.time()
    cutoff = now - settings.RESULT_RETENTION_SECONDS

    for job in job_store.all_jobs():
        if job.completed_at is not None and job.completed_at < cutoff:
            # Remove result files
            result_dir = settings.RESULTS_DIR / job.job_id
            if result_dir.exists():
                shutil.rmtree(result_dir, ignore_errors=True)
                logger.info("Cleaned up expired results for job %s", job.job_id)

            # Remove uploaded file
            if job.input_path:
                input_file = Path(job.input_path)
                if input_file.exists():
                    input_file.unlink(missing_ok=True)

            # Remove job record
            job_store.delete(job.job_id)


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_video(file: UploadFile) -> JSONResponse:
    """
    Accept a video file upload and queue it for processing.

    Returns a JSON object with ``job_id`` and ``status`` ("queued").
    """
    # Validate file name and extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(settings.ALLOWED_VIDEO_EXTENSIONS)}",
        )

    # Validate content type
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Generate job ID and save upload
    job_id = str(uuid.uuid4())
    upload_dir = settings.UPLOAD_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / f"input{ext}"

    # Stream-write to disk (avoids loading entire file in memory)
    size = 0
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    with open(input_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            size += len(chunk)
            if size > max_bytes:
                # Clean up partial file
                input_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB} MB",
                )
            f.write(chunk)

    # Create job record
    job = job_store.create(job_id, str(input_path))

    # Dispatch Celery task
    from api.tasks import process_video

    process_video.delay(job_id, str(input_path))

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": job.status.value},
    )


# ---------------------------------------------------------------------------
# GET /api/ws/status/{job_id} — WebSocket progress updates
# ---------------------------------------------------------------------------

@app.websocket("/api/ws/status/{job_id}")
async def ws_job_status(websocket: WebSocket, job_id: str) -> None:
    """
    WebSocket endpoint that streams stage name and progress percentage.

    Sends JSON messages of the form:
        {"status": "processing", "stage": "pose_2d", "progress": 45.0}

    Closes when the job reaches "completed" or "failed".
    """
    await websocket.accept()

    job = job_store.get(job_id)
    if job is None:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close(code=4004)
        return

    try:
        while True:
            job = job_store.get(job_id)
            if job is None:
                await websocket.send_json({"error": "Job not found"})
                await websocket.close(code=4004)
                return

            payload = {
                "status": job.status.value,
                "stage": job.stage.value if job.stage else None,
                "progress": job.progress,
            }

            if job.error:
                payload["error"] = job.error

            await websocket.send_json(payload)

            # Terminal states — send final message and close
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                await websocket.close()
                return

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error for job %s", job_id)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# GET /api/results/{job_id}/video
# ---------------------------------------------------------------------------

@app.get("/api/results/{job_id}/video")
async def get_result_video(job_id: str) -> FileResponse:
    """Serve the annotated video for a completed job."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed (current status: {job.status.value})",
        )

    video_path = settings.RESULTS_DIR / job_id / "annotated.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not found")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"{job_id}_annotated.mp4",
    )


# ---------------------------------------------------------------------------
# GET /api/results/{job_id}/poses
# ---------------------------------------------------------------------------

@app.get("/api/results/{job_id}/poses")
async def get_result_poses(job_id: str) -> JSONResponse:
    """Serve per-frame Pose3D JSON for the 3D viewer."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed (current status: {job.status.value})",
        )

    poses_path = settings.RESULTS_DIR / job_id / "poses.json"
    if not poses_path.exists():
        raise HTTPException(status_code=404, detail="Poses data not found")

    data = json.loads(poses_path.read_text())
    return JSONResponse(content=data)
