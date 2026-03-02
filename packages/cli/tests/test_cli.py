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
