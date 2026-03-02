"""Custom exceptions for the video pipeline."""


class VideoProcessingError(Exception):
    """Raised when video processing fails (invalid file, encoding error, etc.)."""


class DependencyError(Exception):
    """Raised when a required external dependency (e.g. FFmpeg) is not available."""
