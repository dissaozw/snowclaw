"""Tests for the SnowClaw CLI tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cli.process import main


class TestCLIProcess:
    def test_no_args_shows_help(self, capsys):
        result = main([])
        assert result == 1

    def test_invalid_video_path(self, capsys):
        result = main(["process", "/tmp/nonexistent_video.mp4"])
        assert result == 1

    def test_creates_output_dir(self, tmp_path):
        output_dir = tmp_path / "new_output"
        assert not output_dir.exists()
        # Will fail at ffmpeg_check or later, but output dir should be created
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        with patch("video_pipeline.ffmpeg_check", side_effect=Exception("no ffmpeg")):
            result = main(["process", str(video), "--output-dir", str(output_dir)])
        assert result == 1
        assert output_dir.exists()

    def test_help_flag(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["process", "--help"])
        assert exc.value.code == 0

    def test_process_subcommand_exists(self):
        """Verify 'process' is a valid subcommand."""
        # Should fail on missing file, not on unknown subcommand
        result = main(["process", "/tmp/nonexistent.mp4"])
        assert result == 1

    def test_mock_flag_accepted(self, capsys):
        """--mock flag should be accepted by the argument parser."""
        with pytest.raises(SystemExit) as exc:
            main(["process", "--help"])
        captured = capsys.readouterr()
        assert "--mock" in captured.out


class TestCLIIntegration:
    """
    Task 12.4: Full CLI integration test using --mock flag.

    Runs the complete pipeline (FFmpeg → frames → 2D keypoints → 3D poses
    → annotated video + poses.json) using mock backends so no GPU or model
    download is required.
    """

    SAMPLE_VIDEO = Path(__file__).parents[3] / "data" / "samples" / "ski_demo.mp4"

    def test_full_pipeline_with_mock(self, tmp_path):
        """
        CLI produces annotated.mp4 and poses.json from the sample video.
        This is task 12.4 from the Phase 1 spec.
        """
        if not self.SAMPLE_VIDEO.exists():
            pytest.skip(f"Sample video not found: {self.SAMPLE_VIDEO}")

        output_dir = tmp_path / "results"
        result = main([
            "process",
            str(self.SAMPLE_VIDEO),
            "--output-dir", str(output_dir),
            "--mock",
        ])

        assert result == 0, "CLI should exit with code 0 on success"

        annotated = output_dir / "annotated.mp4"
        poses_json = output_dir / "poses.json"

        assert annotated.exists(), "annotated.mp4 should be created"
        assert annotated.stat().st_size > 1000, "annotated.mp4 should not be empty"
        assert poses_json.exists(), "poses.json should be created"

        import json
        with open(poses_json) as f:
            poses = json.load(f)

        assert isinstance(poses, list)
        assert len(poses) > 0, "poses.json should contain at least one frame"

        # Validate structure of first frame
        first = poses[0]
        assert "frame_idx" in first
        assert "timestamp_s" in first
        assert "pose" in first
        assert "metrics" in first

        # Validate pose has required joints
        pose = first["pose"]
        for joint in ("head", "neck", "left_shoulder", "right_shoulder",
                       "left_hip", "right_hip", "left_knee", "right_knee",
                       "left_ankle", "right_ankle"):
            assert joint in pose, f"Joint '{joint}' missing from poses.json"
            assert len(pose[joint]) == 3, f"Joint '{joint}' should have 3D coords"

    def test_mock_env_var(self, tmp_path, monkeypatch):
        """SNOWCLAW_MOCK_MODELS=1 should enable mock mode without --mock flag."""
        if not self.SAMPLE_VIDEO.exists():
            pytest.skip(f"Sample video not found: {self.SAMPLE_VIDEO}")

        monkeypatch.setenv("SNOWCLAW_MOCK_MODELS", "1")
        output_dir = tmp_path / "results_env"
        result = main([
            "process",
            str(self.SAMPLE_VIDEO),
            "--output-dir", str(output_dir),
        ])
        assert result == 0
        assert (output_dir / "poses.json").exists()
