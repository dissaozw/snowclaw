## Why

SnowCoach AI currently has a biomechanics engine but no way to go from a video to visible results. The product has zero user-facing capability. Phase 1 delivers two concrete outputs a user can see immediately: (1) an annotated video with skeleton overlay and metrics (like SkiPro AI but with better accuracy), and (2) a simple 3D skeleton viewer where users can pause any frame and orbit around their skeleton.

Shared data structures (`Pose3D`, `Frame`, etc.) currently live inside `biomechanics`, creating unwanted coupling. These belong in a `core` package so that every pipeline stage is independently importable.

## What Changes

- **New `core` package** — Extract shared data structures (`Pose3D`, `Frame`, `Discipline`, `TurnPhaseLabel`, `SessionMetrics`, `TurnPhase`) from `biomechanics` into `packages/core/`. All packages import from here.
- **Refactor `biomechanics`** — Import schemas from `core`, re-export for backward compatibility.
- **New `video-pipeline` package** — FFmpeg-based video preprocessing: frame extraction, resolution normalization, metadata extraction. No ML dependencies.
- **New `pose-estimation` package** — Pluggable backends: ViTPose+ (2D) and MotionBERT (3D lifting). Outputs `core.Pose3D`. Does not depend on `biomechanics`.
- **New video annotation renderer** — Draws skeleton overlay (joint dots, connecting lines, COM plumb line, metric numbers) onto video frames and encodes an output video. Uses OpenCV/Pillow for drawing, FFmpeg for encoding.
- **New simple 3D viewer** — React Three Fiber component: render skeleton as joints + bones, orbit controls, frame scrubber/timeline. Skeleton only — no body mesh, no snow surface, no ski geometry (Phase 2).
- **FastAPI backend** — Video upload endpoint, async ML pipeline via Celery, WebSocket progress updates, serves annotated video and 3D skeleton data.

## Capabilities

### New Capabilities
- `core-schemas`: Shared data structures extracted into independent core package
- `video-preprocessing`: FFmpeg frame extraction, normalization, metadata
- `pose-estimation-2d`: ViTPose+ 2D keypoint detection with pluggable backend
- `pose-estimation-3d`: MotionBERT 3D lifting with pluggable backend
- `video-annotation`: Render skeleton overlay + metrics onto video frames, encode output video
- `viewer-3d-skeleton`: Simple Three.js/R3F skeleton viewer with orbit controls and frame scrubbing
- `api-backend`: FastAPI video upload, async pipeline, WebSocket status, serve results
- `gpu-deployment`: Dockerfile with CUDA, docker-compose, GPU cloud deployment support

### Modified Capabilities
- `biomechanics`: Refactor to import schemas from `core` (no behavioral changes)

## Impact

- **New packages**: `packages/core/`, `packages/video-pipeline/`, `packages/pose-estimation/`, `packages/video-annotation/`, `packages/viewer-3d/`, `packages/api/`
- **Modified packages**: `packages/biomechanics/` — schemas move to `core`
- **New frontend code**: `packages/viewer-3d/` (React Three Fiber, TypeScript)
- **Dependencies**: `core` → numpy + pydantic. `video-pipeline` → zero (FFmpeg subprocess). `pose-estimation` → core + onnxruntime. `video-annotation` → core + opencv-python. `api` → FastAPI + Celery + Redis.
- **Infrastructure**: Docker with NVIDIA Container Toolkit for GPU inference, Redis for job queue, ML model weights pre-warmed in Docker image. Deployable to any GPU cloud (AWS, GCP, Lambda Labs, RunPod).
