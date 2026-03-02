"""Frame extraction and resolution normalization via FFmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .exceptions import DependencyError, VideoProcessingError
from .metadata import extract_metadata


def ffmpeg_check() -> None:
    """
    Verify that FFmpeg is available on PATH.

    Raises:
        DependencyError: If FFmpeg is not found.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise DependencyError("FFmpeg returned a non-zero exit code")
    except FileNotFoundError:
        raise DependencyError(
            "FFmpeg not found on PATH. Install it: https://ffmpeg.org/download.html"
        )


def _compute_scale_filter(
    width: int, height: int, max_dimension: int
) -> str | None:
    """
    Return an FFmpeg scale filter string to downscale to max_dimension on the
    long edge, preserving aspect ratio. Returns None if no scaling needed.
    """
    if max(width, height) <= max_dimension:
        return None
    if width >= height:
        return f"scale={max_dimension}:-2"
    else:
        return f"scale=-2:{max_dimension}"


def extract_frames(
    video_path: str | Path,
    target_fps: float | None = None,
    max_dimension: int = 1920,
) -> list[np.ndarray]:
    """
    Extract RGB frames from a video file using FFmpeg.

    Args:
        video_path: Path to the video file.
        target_fps: Target frames per second. If None, uses the native FPS.
        max_dimension: Maximum pixel dimension on the long edge (default 1920).
                       Frames are downscaled preserving aspect ratio. No upscaling.

    Returns:
        List of numpy arrays with shape (H, W, 3), dtype uint8, in RGB order.

    Raises:
        DependencyError: If FFmpeg is not on PATH.
        VideoProcessingError: If extraction fails.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoProcessingError(f"Video file not found: {video_path}")

    meta = extract_metadata(video_path)

    # Build filter chain
    filters: list[str] = []

    # FPS filter
    effective_fps = target_fps if target_fps is not None else meta.fps
    if target_fps is not None and target_fps != meta.fps:
        filters.append(f"fps={target_fps}")

    # Scale filter (downscale only)
    scale = _compute_scale_filter(meta.width, meta.height, max_dimension)
    if scale is not None:
        filters.append(scale)

    vf_arg = ",".join(filters) if filters else None

    # Build FFmpeg command: output raw RGB24 frames to stdout
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
    ]
    if vf_arg:
        cmd.extend(["-vf", vf_arg])
    cmd.extend(["-v", "error", "pipe:1"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=300,
        )
    except FileNotFoundError:
        raise DependencyError(
            "FFmpeg not found on PATH. Install it: https://ffmpeg.org/download.html"
        )

    if result.returncode != 0:
        raise VideoProcessingError(
            f"FFmpeg frame extraction failed: {result.stderr.decode().strip()}"
        )

    raw = result.stdout
    if len(raw) == 0:
        raise VideoProcessingError("FFmpeg produced no output frames")

    # Determine output dimensions after scaling
    if scale is not None:
        # We need to figure out actual output size; run a quick probe
        probe_cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
        ]
        if vf_arg:
            probe_cmd.extend(["-vf", vf_arg])
        probe_cmd.extend(["-frames:v", "1", "-v", "error", "pipe:1"])

        probe_result = subprocess.run(probe_cmd, capture_output=True, timeout=30)
        probe_bytes = len(probe_result.stdout)
        if probe_bytes == 0:
            raise VideoProcessingError("Could not determine output frame dimensions")

        # Solve for dimensions: H * W * 3 = probe_bytes
        # Use aspect ratio to determine
        aspect = meta.width / meta.height
        if meta.width >= meta.height:
            out_w = max_dimension
            out_h = int(round(max_dimension / aspect))
            # FFmpeg -2 makes height even
            if out_h % 2 != 0:
                out_h += 1
        else:
            out_h = max_dimension
            out_w = int(round(max_dimension * aspect))
            if out_w % 2 != 0:
                out_w += 1

        # Verify against actual bytes
        expected_frame_bytes = out_w * out_h * 3
        if probe_bytes != expected_frame_bytes:
            # Fall back to deriving from probe
            out_w = probe_bytes // (out_h * 3) if out_h > 0 else 0
            if out_w * out_h * 3 != probe_bytes:
                out_h = probe_bytes // (out_w * 3) if out_w > 0 else 0
    else:
        out_w = meta.width
        out_h = meta.height

    frame_bytes = out_w * out_h * 3
    if frame_bytes == 0:
        raise VideoProcessingError("Computed zero-size frames")

    num_frames = len(raw) // frame_bytes
    if num_frames == 0:
        raise VideoProcessingError("No complete frames in FFmpeg output")

    frames = []
    for i in range(num_frames):
        frame_data = raw[i * frame_bytes : (i + 1) * frame_bytes]
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(out_h, out_w, 3)
        frames.append(frame)

    return frames
