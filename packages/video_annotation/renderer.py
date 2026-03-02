"""Video annotation renderer — draws overlays and encodes output video."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from core.schemas import Pose3D
from video_pipeline.exceptions import DependencyError, VideoProcessingError

from .skeleton import draw_com_plumb_line, draw_metrics_text, draw_skeleton


def annotate_frames(
    frames: list[np.ndarray],
    poses: list[Pose3D],
) -> list[np.ndarray]:
    """
    Draw skeleton overlay, COM plumb line, and metrics on each frame.

    Args:
        frames: List of BGR images (H, W, 3), uint8.
        poses: List of Pose3D, one per frame. Must match len(frames).

    Returns:
        List of annotated frames (copies, originals not modified).
    """
    if len(frames) != len(poses):
        raise ValueError(
            f"frames ({len(frames)}) and poses ({len(poses)}) must have the same length"
        )

    annotated = []
    for frame, pose in zip(frames, poses):
        out = frame.copy()
        draw_skeleton(out, pose)
        draw_com_plumb_line(out, pose)
        draw_metrics_text(out, pose)
        annotated.append(out)

    return annotated


def annotate_video(
    input_path: str | Path,
    poses: list[Pose3D],
    output_path: str | Path,
    fps: float | None = None,
) -> Path:
    """
    Annotate a full video file: draw skeleton overlay on each frame, encode output.

    Preserves original audio track.

    Args:
        input_path: Path to the original video.
        poses: List of Pose3D, one per frame.
        output_path: Path for the output annotated video.
        fps: FPS for the output video. If None, uses source FPS.

    Returns:
        Path to the output video file.

    Raises:
        DependencyError: If FFmpeg is not available.
        VideoProcessingError: If encoding fails.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from video_pipeline.metadata import extract_metadata
    meta = extract_metadata(input_path)

    if fps is None:
        fps = meta.fps

    # Extract frames from source video
    from video_pipeline.frames import extract_frames
    frames_rgb = extract_frames(input_path)

    # Match poses to frames (use min of both lengths)
    n = min(len(frames_rgb), len(poses))

    # Convert RGB to BGR for OpenCV annotation
    annotated_bgr = []
    for i in range(n):
        bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        draw_skeleton(bgr, poses[i])
        draw_com_plumb_line(bgr, poses[i])
        draw_metrics_text(bgr, poses[i])
        annotated_bgr.append(bgr)

    if not annotated_bgr:
        raise VideoProcessingError("No frames to annotate")

    h, w = annotated_bgr[0].shape[:2]

    # Write annotated frames to temp raw file, then encode with FFmpeg
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp_path = tmp.name
        for frame in annotated_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tmp.write(rgb.tobytes())

    # Build FFmpeg command to encode from raw frames + copy audio from original
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", tmp_path,
        "-i", str(input_path),
        "-map", "0:v",      # Video from raw frames
        "-map", "1:a?",     # Audio from original (if exists)
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        "-v", "error",
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
    except FileNotFoundError:
        raise DependencyError("FFmpeg not found on PATH")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise VideoProcessingError(
            f"FFmpeg encoding failed: {result.stderr.decode().strip()}"
        )

    return output_path
