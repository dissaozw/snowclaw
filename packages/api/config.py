"""
Configuration and in-process job storage for the SnowClaw API.

Settings are loaded from environment variables with sensible defaults.
Job state is stored in a module-level dict so it can be swapped for
Redis in production without changing the rest of the code.
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings:
    """Application settings loaded from environment variables."""

    UPLOAD_DIR: Path = Path(os.getenv("SNOWCLAW_UPLOAD_DIR", "/tmp/snowclaw/uploads"))
    RESULTS_DIR: Path = Path(os.getenv("SNOWCLAW_RESULTS_DIR", "/tmp/snowclaw/results"))
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("SNOWCLAW_MAX_UPLOAD_MB", "500"))
    RESULT_RETENTION_SECONDS: int = int(
        os.getenv("SNOWCLAW_RESULT_RETENTION_SECONDS", str(24 * 60 * 60))
    )
    ALLOWED_VIDEO_EXTENSIONS: set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


settings = Settings()


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    """Lifecycle states for a video processing job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStage(str, Enum):
    """Named stages of the ML pipeline, reported via WebSocket."""

    EXTRACTING_FRAMES = "extracting_frames"
    POSE_2D = "pose_2d"
    POSE_3D = "pose_3d"
    ANNOTATING_VIDEO = "annotating_video"
    COMPUTING_METRICS = "computing_metrics"
    SAVING_RESULTS = "saving_results"


@dataclass
class Job:
    """Mutable job record."""

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    stage: Optional[PipelineStage] = None
    progress: float = 0.0  # 0-100
    error: Optional[str] = None
    input_path: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# ---------------------------------------------------------------------------
# In-memory job store (thread-safe)
# ---------------------------------------------------------------------------

class JobStore:
    """
    Thread-safe in-memory job store.

    Production deployments should replace this with a Redis-backed
    implementation sharing the same interface.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, input_path: str) -> Job:
        """Create a new job and return it."""
        job = Job(job_id=job_id, input_path=input_path)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Return a job by ID, or None if not found."""
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        stage: Optional[PipelineStage] = None,
        progress: Optional[float] = None,
        error: Optional[str] = None,
        completed_at: Optional[float] = None,
    ) -> Optional[Job]:
        """Update fields on an existing job. Returns the updated job or None."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if progress is not None:
                job.progress = progress
            if error is not None:
                job.error = error
            if completed_at is not None:
                job.completed_at = completed_at
            return job

    def all_jobs(self) -> list[Job]:
        """Return a snapshot of all jobs."""
        with self._lock:
            return list(self._jobs.values())

    def delete(self, job_id: str) -> bool:
        """Remove a job. Returns True if it existed."""
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def clear(self) -> None:
        """Remove all jobs (useful for tests)."""
        with self._lock:
            self._jobs.clear()


# Module-level singleton — imported by app.py and tasks.py
job_store = JobStore()
