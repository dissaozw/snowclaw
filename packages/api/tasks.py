"""
Celery tasks for the SnowClaw async video processing pipeline.

The main task `process_video` runs the full ML pipeline:
  1. Extract frames  (video_pipeline)
  2. 2D pose estimation (pose_estimation — ViTPose+)
  3. 3D pose lifting  (pose_estimation — MotionBERT)
  4. Video annotation  (video_annotation)
  5. Per-frame metrics  (video_annotation.skeleton.format_metrics)
  6. Save results      (annotated.mp4 + poses.json)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from celery import Celery

from api.config import (
    Job,
    JobStatus,
    PipelineStage,
    job_store,
    settings,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

celery_app = Celery(
    "snowclaw",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update_progress(
    job_id: str,
    stage: PipelineStage,
    progress: float,
) -> None:
    """Convenience wrapper to update a job's stage and progress."""
    job_store.update(
        job_id,
        status=JobStatus.PROCESSING,
        stage=stage,
        progress=round(progress, 1),
    )


def _results_dir(job_id: str) -> Path:
    """Return (and create) the results directory for a job."""
    d = settings.RESULTS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Main pipeline task
# ---------------------------------------------------------------------------

@celery_app.task(name="process_video", bind=True, max_retries=0)
def process_video(self, job_id: str, input_path: str) -> dict:
    """
    Run the full video -> skeleton -> annotation pipeline.

    Args:
        job_id: UUID identifying the job.
        input_path: Absolute path to the uploaded video file.

    Returns:
        Dict with ``annotated_video`` and ``poses_json`` result paths.
    """
    try:
        job_store.update(job_id, status=JobStatus.PROCESSING)
        results_dir = _results_dir(job_id)

        # ---- Stage 1: Extract frames ----
        _update_progress(job_id, PipelineStage.EXTRACTING_FRAMES, 0)
        from video_pipeline import extract_frames

        frames = extract_frames(input_path)
        _update_progress(job_id, PipelineStage.EXTRACTING_FRAMES, 100)
        logger.info("Job %s: extracted %d frames", job_id, len(frames))

        # ---- Stage 2: 2D pose estimation (ViTPose+) ----
        _update_progress(job_id, PipelineStage.POSE_2D, 0)
        from pose_estimation.vitpose_backend import ViTPoseBackend

        estimator = ViTPoseBackend()
        keypoints_2d = estimator.predict(frames)
        _update_progress(job_id, PipelineStage.POSE_2D, 100)
        logger.info("Job %s: 2D pose estimation complete", job_id)

        # ---- Stage 3: 3D pose lifting (MotionBERT) ----
        _update_progress(job_id, PipelineStage.POSE_3D, 0)
        from pose_estimation.motionbert_backend import MotionBERTBackend

        lifter = MotionBERTBackend()
        poses_3d = lifter.lift(keypoints_2d)
        _update_progress(job_id, PipelineStage.POSE_3D, 100)
        logger.info("Job %s: 3D lifting complete (%d poses)", job_id, len(poses_3d))

        # ---- Stage 4: Annotate video ----
        _update_progress(job_id, PipelineStage.ANNOTATING_VIDEO, 0)
        from video_annotation import annotate_video

        annotated_path = results_dir / "annotated.mp4"
        annotate_video(input_path, poses_3d, annotated_path)
        _update_progress(job_id, PipelineStage.ANNOTATING_VIDEO, 100)
        logger.info("Job %s: annotated video saved", job_id)

        # ---- Stage 5: Compute per-frame metrics ----
        _update_progress(job_id, PipelineStage.COMPUTING_METRICS, 0)
        from video_annotation.skeleton import format_metrics
        from video_pipeline import extract_metadata

        metadata = extract_metadata(input_path)
        per_frame: list[dict] = []
        for idx, pose in enumerate(poses_3d):
            timestamp_s = idx / metadata.fps if metadata.fps > 0 else 0.0
            metrics = format_metrics(pose)
            per_frame.append({
                "frame_idx": idx,
                "timestamp_s": round(timestamp_s, 4),
                "pose": pose.model_dump(),
                "metrics": metrics,
            })
            # Report granular progress
            if len(poses_3d) > 0:
                pct = (idx + 1) / len(poses_3d) * 100
                _update_progress(job_id, PipelineStage.COMPUTING_METRICS, pct)

        # ---- Stage 6: Save results ----
        _update_progress(job_id, PipelineStage.SAVING_RESULTS, 0)
        poses_json_path = results_dir / "poses.json"
        poses_json_path.write_text(json.dumps(per_frame, indent=2))
        _update_progress(job_id, PipelineStage.SAVING_RESULTS, 100)
        logger.info("Job %s: poses.json saved (%d frames)", job_id, len(per_frame))

        # ---- Mark complete ----
        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            stage=PipelineStage.SAVING_RESULTS,
            progress=100,
            completed_at=time.time(),
        )

        return {
            "annotated_video": str(annotated_path),
            "poses_json": str(poses_json_path),
        }

    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            error=str(exc),
        )
        raise
