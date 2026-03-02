"""Tests for video_pipeline — metadata extraction, frame extraction, ffmpeg check."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from video_pipeline import (
    DependencyError,
    VideoProcessingError,
    extract_frames,
    extract_metadata,
    ffmpeg_check,
)
from video_pipeline.frames import _compute_scale_filter
from video_pipeline.metadata import VideoMetadata

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> Path:
    """Generate a 2-second 320x240 test video using FFmpeg."""
    out = tmp_path_factory.mktemp("video") / "test.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "color=c=red:s=320x240:d=2:r=10",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-v", "error",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


# ── FFmpeg check ─────────────────────────────────────────────


class TestFfmpegCheck:
    def test_ffmpeg_available(self):
        ffmpeg_check()  # Should not raise

    def test_ffmpeg_missing_raises(self, monkeypatch):
        monkeypatch.setenv("PATH", "")
        with pytest.raises(DependencyError, match="FFmpeg not found"):
            ffmpeg_check()


# ── Metadata extraction ──────────────────────────────────────


class TestExtractMetadata:
    def test_basic_metadata(self, sample_video):
        meta = extract_metadata(sample_video)
        assert isinstance(meta, VideoMetadata)
        assert meta.width == 320
        assert meta.height == 240
        assert meta.fps == pytest.approx(10.0, abs=0.1)
        assert meta.duration_s == pytest.approx(2.0, abs=0.5)
        assert meta.frame_count >= 15  # ~20 frames at 10fps for 2s
        assert meta.codec == "h264"

    def test_missing_file_raises(self):
        with pytest.raises(VideoProcessingError, match="not found"):
            extract_metadata("/tmp/nonexistent_video.mp4")


# ── Resolution normalization ─────────────────────────────────


class TestScaleFilter:
    def test_no_scaling_needed(self):
        assert _compute_scale_filter(1280, 720, 1920) is None

    def test_exact_match_no_scaling(self):
        assert _compute_scale_filter(1920, 1080, 1920) is None

    def test_landscape_downscale(self):
        f = _compute_scale_filter(3840, 2160, 1920)
        assert f == "scale=1920:-2"

    def test_portrait_downscale(self):
        f = _compute_scale_filter(1080, 1920, 1280)
        assert f == "scale=-2:1280"

    def test_square_downscale(self):
        f = _compute_scale_filter(4000, 4000, 1920)
        assert f == "scale=1920:-2"


# ── Frame extraction ─────────────────────────────────────────


class TestExtractFrames:
    def test_extracts_frames(self, sample_video):
        frames = extract_frames(sample_video)
        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.dtype == np.uint8
            assert frame.shape == (240, 320, 3)  # H, W, C

    def test_custom_fps(self, sample_video):
        frames = extract_frames(sample_video, target_fps=5)
        # At 5fps for 2s, expect ~10 frames (±1)
        assert 8 <= len(frames) <= 12

    def test_max_dimension_downscale(self, sample_video):
        frames = extract_frames(sample_video, max_dimension=160)
        assert len(frames) > 0
        for frame in frames:
            assert max(frame.shape[0], frame.shape[1]) <= 162  # ±2 for even rounding

    def test_no_upscale(self, sample_video):
        """max_dimension larger than video should not upscale."""
        frames = extract_frames(sample_video, max_dimension=4096)
        assert len(frames) > 0
        assert frames[0].shape == (240, 320, 3)

    def test_missing_file_raises(self):
        with pytest.raises(VideoProcessingError, match="not found"):
            extract_frames("/tmp/nonexistent.mp4")

    def test_frames_are_rgb(self, sample_video):
        """Test video is solid red — R channel should dominate."""
        frames = extract_frames(sample_video)
        frame = frames[0]
        r_mean = frame[:, :, 0].mean()
        g_mean = frame[:, :, 1].mean()
        b_mean = frame[:, :, 2].mean()
        # Red channel should be much higher than green/blue
        assert r_mean > 200
        assert g_mean < 50
        assert b_mean < 50
