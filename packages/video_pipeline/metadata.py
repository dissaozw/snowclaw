"""Video metadata extraction via ffprobe."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .exceptions import DependencyError, VideoProcessingError


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata extracted from a video file."""

    duration_s: float
    fps: float
    width: int
    height: int
    codec: str
    frame_count: int


def extract_metadata(video_path: str | Path) -> VideoMetadata:
    """
    Extract metadata from a video file using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoMetadata with duration, fps, resolution, codec, and frame count.

    Raises:
        DependencyError: If ffprobe is not on PATH.
        VideoProcessingError: If the file is invalid or ffprobe fails.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoProcessingError(f"Video file not found: {video_path}")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise DependencyError(
            "ffprobe not found on PATH. Install FFmpeg: https://ffmpeg.org/download.html"
        )

    if result.returncode != 0:
        raise VideoProcessingError(
            f"ffprobe failed for {video_path}: {result.stderr.strip()}"
        )

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise VideoProcessingError("ffprobe returned invalid JSON")

    # Find the first video stream
    video_stream = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise VideoProcessingError(f"No video stream found in {video_path}")

    # Parse FPS from r_frame_rate (e.g. "30/1" or "30000/1001")
    r_frame_rate = video_stream.get("r_frame_rate", "0/1")
    num, den = r_frame_rate.split("/")
    fps = float(num) / float(den) if float(den) > 0 else 0.0

    # Duration: prefer stream duration, fall back to format duration
    duration_s = float(
        video_stream.get("duration")
        or probe.get("format", {}).get("duration", 0)
    )

    # Frame count: prefer nb_frames, else estimate from duration
    nb_frames = video_stream.get("nb_frames")
    if nb_frames and nb_frames != "N/A":
        frame_count = int(nb_frames)
    else:
        frame_count = int(duration_s * fps) if fps > 0 else 0

    return VideoMetadata(
        duration_s=duration_s,
        fps=fps,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        codec=video_stream.get("codec_name", "unknown"),
        frame_count=frame_count,
    )
