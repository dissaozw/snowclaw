"""Integration tests — full pipeline with mock ML backends."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.schemas import Pose3D


# ── Helpers ──────────────────────────────────────────────────


def _make_pose(i: int = 0) -> Pose3D:
    """Create a synthetic Pose3D with slight variation per frame."""
    offset = i * 0.01
    return Pose3D(
        head=[0, 1.8 + offset, 0],
        neck=[0, 1.6 + offset, 0],
        left_shoulder=[-0.2, 1.5 + offset, 0],
        right_shoulder=[0.2, 1.5 + offset, 0],
        left_elbow=[-0.3, 1.2 + offset, 0],
        right_elbow=[0.3, 1.2 + offset, 0],
        left_wrist=[-0.3, 1.0 + offset, 0],
        right_wrist=[0.3, 1.0 + offset, 0],
        left_hip=[-0.1, 1.0 + offset, 0],
        right_hip=[0.1, 1.0 + offset, 0],
        left_knee=[-0.1, 0.5 + offset, 0],
        right_knee=[0.1, 0.5 + offset, 0],
        left_ankle=[-0.1, 0.0, 0],
        right_ankle=[0.1, 0.0, 0],
        confidence={j: 0.9 for j in [
            "head", "neck", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ]},
    )


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> Path:
    """Generate a 2-second test video."""
    out = tmp_path_factory.mktemp("integration") / "test.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=green:s=320x240:d=2:r=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-v", "error", str(out),
        ],
        check=True, capture_output=True,
    )
    return out


# ── Integration: mock backends → annotated video ─────────────

class TestAnnotatedVideoIntegration:
    def test_mock_pipeline_produces_video(self, sample_video, tmp_path):
        """Full pipeline with synthetic poses → annotated video is produced."""
        from video_pipeline import extract_frames, extract_metadata
        from video_annotation.renderer import annotate_video

        # Extract real frames
        meta = extract_metadata(sample_video)
        frames = extract_frames(sample_video)
        assert len(frames) > 0

        # Create mock poses (one per frame)
        poses = [_make_pose(i) for i in range(len(frames))]

        # Annotate video
        output_video = tmp_path / "annotated.mp4"
        annotate_video(sample_video, poses, output_video)

        assert output_video.exists()
        assert output_video.stat().st_size > 0

        # Verify output is a valid video
        out_meta = extract_metadata(output_video)
        assert out_meta.width > 0
        assert out_meta.height > 0


# ── Integration: mock backends → valid pose JSON ─────────────

class TestPoseJSONIntegration:
    def test_pose_json_is_valid(self):
        """Verify that Pose3D → JSON → Pose3D round-trips correctly."""
        original = _make_pose(5)
        json_str = original.model_dump_json()
        restored = Pose3D.model_validate_json(json_str)

        np.testing.assert_array_almost_equal(
            original.to_np("head"),
            restored.to_np("head"),
        )
        assert original.confidence == restored.confidence

    def test_poses_json_format(self):
        """Verify the JSON format expected by the 3D viewer."""
        from video_annotation.skeleton import format_metrics

        poses = [_make_pose(i) for i in range(5)]
        frames_data = []
        for i, pose in enumerate(poses):
            frame_data = {
                "frame_idx": i,
                "timestamp_s": round(i / 10.0, 4),
                "pose": pose.model_dump(exclude_none=True),
                "metrics": format_metrics(pose),
            }
            frames_data.append(frame_data)

        json_str = json.dumps(frames_data)
        parsed = json.loads(json_str)

        assert len(parsed) == 5
        assert parsed[0]["frame_idx"] == 0
        assert "head" in parsed[0]["pose"]
        assert "left_knee_deg" in parsed[0]["metrics"]
        assert 0 <= parsed[0]["metrics"]["com_height_pct"] <= 1


# ── Full test suite ──────────────────────────────────────────

class TestFullSuite:
    def test_all_packages_importable(self):
        """Verify all packages can be imported without errors."""
        import core
        import biomechanics
        import video_pipeline
        import pose_estimation
        import video_annotation
        import snowclaw

        assert core.Pose3D is not None
        assert biomechanics.edge_angle is not None
        assert video_pipeline.extract_frames is not None
        assert pose_estimation.Keypoints2D is not None
        assert video_annotation.annotate_frames is not None
        assert snowclaw is not None
