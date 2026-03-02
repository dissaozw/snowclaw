## 1. Core Package — Shared Schemas

- [ ] 1.1 Scaffold `packages/core/` with `__init__.py`, `schemas.py`, `README.md`, and `tests/` directory
- [ ] 1.2 Move `Pose3D`, `Frame`, `Discipline`, `TurnPhaseLabel`, `TurnPhase`, `SessionMetrics` from `biomechanics/schemas.py` into `core/schemas.py`
- [ ] 1.3 Update `core/__init__.py` to export all shared types
- [ ] 1.4 Write tests for `core` schemas (Pose3D construction, Frame validation, enum values)

## 2. Refactor Biomechanics — Import from Core

- [ ] 2.1 Replace `biomechanics/schemas.py` with imports from `core.schemas`
- [ ] 2.2 Update `biomechanics/__init__.py` to re-export all types from `core` for backward compatibility
- [ ] 2.3 Update internal biomechanics imports (metrics.py, snow_iq.py, turn_segmentation.py)
- [ ] 2.4 Run `pytest packages/biomechanics/` — all existing tests must pass unchanged

## 3. Video Pipeline Package

- [ ] 3.1 Scaffold `packages/video-pipeline/` with `__init__.py`, `README.md`, `tests/`
- [ ] 3.2 Implement `ffmpeg_check()` — verify FFmpeg on PATH, raise `DependencyError` if missing
- [ ] 3.3 Implement `VideoMetadata` dataclass (duration_s, fps, width, height, codec, frame_count)
- [ ] 3.4 Implement `extract_metadata(video_path) -> VideoMetadata` via ffprobe subprocess
- [ ] 3.5 Implement `extract_frames(video_path, target_fps, max_dimension) -> list[np.ndarray]` — RGB frames via FFmpeg
- [ ] 3.6 Implement resolution normalization (downscale to max_dimension, preserve aspect ratio, no upscale)
- [ ] 3.7 Define `VideoProcessingError` and `DependencyError` exceptions
- [ ] 3.8 Write tests for metadata extraction, frame extraction, resolution normalization, ffmpeg check

## 4. Pose Estimation — Interfaces & Data

- [ ] 4.1 Scaffold `packages/pose-estimation/` with `__init__.py`, `README.md`, `tests/`
- [ ] 4.2 Define `Keypoints2D` dataclass (points: ndarray Jx2, confidence: ndarray J, image_size: tuple)
- [ ] 4.3 Define `PoseEstimator2D` ABC with `predict(frames) -> list[Keypoints2D]`
- [ ] 4.4 Define `PoseLifter3D` ABC with `lift(keypoints_2d) -> list[core.Pose3D]`
- [ ] 4.5 Implement COCO-to-Pose3D joint mapping (17 COCO → 14 core.Pose3D joints)
- [ ] 4.6 Implement model weight download utility with progress bar and cache (~/.cache/snowclaw/models/)

## 5. Pose Estimation — ViTPose+ Backend (2D)

- [ ] 5.1 Implement `ViTPoseBackend(PoseEstimator2D)` with ONNX Runtime inference
- [ ] 5.2 Preprocessing: resize frames, normalize, build input tensor
- [ ] 5.3 Postprocessing: decode heatmaps → pixel coordinates + confidence → `Keypoints2D`
- [ ] 5.4 Batch processing (split frames into configurable batch sizes)
- [ ] 5.5 Device auto-detection (CUDA if available, else CPU)
- [ ] 5.6 Write tests with mock ONNX session (input/output shapes, batching)

## 6. Pose Estimation — MotionBERT Backend (3D)

- [ ] 6.1 Implement `MotionBERTBackend(PoseLifter3D)` with ONNX Runtime inference
- [ ] 6.2 Preprocessing: normalize 2D keypoints to [-1, 1], assemble temporal windows
- [ ] 6.3 Postprocessing: convert to meters, Y-up coordinate system, map to core.Pose3D joints
- [ ] 6.4 Confidence propagation from 2D keypoints → core.Pose3D.confidence
- [ ] 6.5 Temporal smoothing (Savitzky-Golay filter on 3D trajectories)
- [ ] 6.6 Write tests with mock ONNX session (coordinate system, joint mapping, smoothing)

## 7. Video Annotation Renderer

- [ ] 7.1 Scaffold `packages/video-annotation/` with `__init__.py`, `README.md`, `tests/`
- [ ] 7.2 Implement skeleton drawer: joint dots (green/yellow/red by confidence), bone lines
- [ ] 7.3 Implement COM plumb line: vertical line from center of mass downward
- [ ] 7.4 Implement metrics text overlay: knee flex angle, inclination, COM height % (top-left)
- [ ] 7.5 Implement `annotate_video(input_path, poses, output_path)` — draw all frames, encode MP4 with FFmpeg, preserve audio
- [ ] 7.6 Write tests for skeleton drawing on synthetic frames, metrics text formatting

## 8. Simple 3D Viewer (React Three Fiber)

- [ ] 8.1 Scaffold `packages/viewer-3d/` as TypeScript/React project (Vite + React Three Fiber)
- [ ] 8.2 Implement skeleton component: spheres at 14 joints, cylinders for bones
- [ ] 8.3 Implement orbit controls (rotate, zoom, pan around skeleton center of mass)
- [ ] 8.4 Implement flat grid ground plane at y=0 as spatial reference
- [ ] 8.5 Implement frame scrubber / timeline (slider + play/pause)
- [ ] 8.6 Implement metrics panel: current frame's knee angle, inclination, COM height
- [ ] 8.7 Fetch pose data from API (`/api/results/{job_id}/poses`) and render

## 9. FastAPI Backend

- [ ] 9.1 Scaffold `packages/api/` with FastAPI app, Celery worker config, Redis connection
- [ ] 9.2 Implement `POST /api/upload` — accept video, create job, return job_id
- [ ] 9.3 Implement Celery task: video-pipeline → pose-2d → pose-3d → video-annotation
- [ ] 9.4 Implement `GET /api/ws/status/{job_id}` — WebSocket progress (stage + percentage)
- [ ] 9.5 Implement `GET /api/results/{job_id}/video` — serve annotated video
- [ ] 9.6 Implement `GET /api/results/{job_id}/poses` — serve per-frame Pose3D JSON for viewer
- [ ] 9.7 Implement result cleanup (delete after configurable retention period)
- [ ] 9.8 Write API tests (upload, status, results, error cases)

## 10. Docker & GPU Deployment

- [ ] 10.1 Create `Dockerfile` based on NVIDIA CUDA base image with Python, FFmpeg, ONNX Runtime GPU
- [ ] 10.2 Add model weight download step to Docker build (pre-warm ViTPose+ and MotionBERT in image)
- [ ] 10.3 Create `Dockerfile.frontend` for the 3D viewer (Node.js + Vite build)
- [ ] 10.4 Create `docker-compose.yml` — API server, Celery worker (GPU), Redis, frontend. Single `docker compose up`
- [ ] 10.5 Configure NVIDIA Container Toolkit GPU passthrough in compose
- [ ] 10.6 Add CPU fallback: detect GPU availability, warn if falling back to CPU
- [ ] 10.7 Write GPU cloud deployment guide (AWS, GCP, RunPod, Lambda Labs)

## 11. Integration & End-to-End

- [ ] 11.1 Integration test: mock backends → annotated video is produced
- [ ] 11.2 Integration test: mock backends → 3D viewer receives valid pose JSON
- [ ] 11.3 Run full test suite (`pytest -v`) — all packages pass
- [ ] 11.4 Docker build + `docker compose up` → full stack starts with GPU
- [ ] 11.5 Manual end-to-end: upload real ski video → annotated video + working 3D viewer
