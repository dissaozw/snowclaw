"""
SnowClaw Video Pipeline — FFmpeg-based video preprocessing.

Extracts frames, normalizes resolution, and reads metadata.
No ML dependencies — pure subprocess calls to FFmpeg.
"""

from .exceptions import DependencyError, VideoProcessingError
from .metadata import VideoMetadata, extract_metadata
from .frames import extract_frames, ffmpeg_check

__all__ = [
    "DependencyError",
    "VideoProcessingError",
    "VideoMetadata",
    "extract_metadata",
    "extract_frames",
    "ffmpeg_check",
]
