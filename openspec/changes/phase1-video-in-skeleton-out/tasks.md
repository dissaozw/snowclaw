## 1. Core Package — Shared Schemas

- [x] 1.1 Scaffold `packages/core/` with `__init__.py`, `schemas.py`, `README.md`, and `tests/` directory
- [x] 1.2 Move `Pose3D`, `Frame`, `Discipline`, `TurnPhaseLabel`, `TurnPhase`, `SessionMetrics` from `biomechanics/schemas.py` into `core/schemas.py`
- [x] 1.3 Update `core/__init__.py` to export all shared types
- [x] 1.4 Write tests for `core` schemas (Pose3D construction, Frame validation, enum values)

## 2. Refactor Biomechanics — Import from Core

- [x] 2.1 Replace `biomechanics/schemas.py` with imports from `core.schemas`
- [x] 2.2 Update `biomechanics/__init__.py` to re-export all types from `core` for backward compatibility
- [x] 2.3 Update internal biomechanics imports (metrics.py, snow_iq.py, turn_segmentation.py)
- [x] 2.4 Run `pytest packages/biomechanics/` — all existing tests must pass unchanged

## 3. Video Pipeline Package

- [x] 3.1 Scaffold `packages/video-pipeline/` with `__init__.py`, `README.md`, `tests/`
- [x] 3.2 Implement `ffmpeg_check()` — verify FFmpeg on PATH, raise `DependencyError` if missing
- [x] 3.3 Implement `VideoMetadata` dataclass (duration_s, fps, width, height, codec, frame_count)
- [x] 3.4 Implement `extract_metadata(video_path) -> VideoMetadata` via ffprobe subprocess
- [x] 3.5 Implement `extract_frames(video_path, target_fps, max_dimension) -> list[np.ndarray]` — RGB frames via FFmpeg
- [x] 3.6 Implement resolution normalization (downscale to max_dimension, preserve aspect ratio, no upscale)
- [x] 3.7 Define `VideoProcessingError` and `DependencyError` exceptions
- [x] 3.8 Write tests for metadata extraction, frame extraction, resolution normalization, ffmpeg check

## 4. Pose Estimation — Interfaces & Data

- [x] 4.1 Scaffold `packages/pose-estimation/` with `__init__.py`, `README.md`, `tests/`
- [x] 4.2 Define `Keypoints2D` dataclass (points: ndarray Jx2, confidence: ndarray J, image_size: tuple)
- [x] 4.3 Define `PoseEstimator2D` ABC with `predict(frames) -> list[Keypoints2D]`
- [x] 4.4 Define `PoseLifter3D` ABC with `lift(keypoints_2d) -> list[core.Pose3D]`
- [x] 4.5 Implement COCO-to-Pose3D joint mapping (17 COCO → 14 core.Pose3D joints)
- [x] 4.6 Implement model weight download utility with progress bar and cache (~/.cache/snowclaw/models/)

## 5. Pose Estimation — ViTPose+ Backend (2D)

- [x] 5.1 Implement `ViTPoseBackend(PoseEstimator2D)` with ONNX Runtime inference
- [x] 5.2 Preprocessing: resize frames, normalize, build input tensor
- [x] 5.3 Postprocessing: decode heatmaps → pixel coordinates + confidence → `Keypoints2D`
- [x] 5.4 Batch processing (split frames into configurable batch sizes)
- [x] 5.5 Device auto-detection (CUDA if available, else CPU)
- [x] 5.6 Write tests with mock ONNX session (input/output shapes, batching)

## 6. Pose Estimation — MotionBERT Backend (3D)

- [x] 6.1 Implement `MotionBERTBackend(PoseLifter3D)` with ONNX Runtime inference
- [x] 6.2 Preprocessing: normalize 2D keypoints to [-1, 1], assemble temporal windows
- [x] 6.3 Postprocessing: convert to meters, Y-up coordinate system, map to core.Pose3D joints
- [x] 6.4 Confidence propagation from 2D keypoints → core.Pose3D.confidence
- [x] 6.5 Temporal smoothing (Savitzky-Golay filter on 3D trajectories)
- [x] 6.6 Write tests with mock ONNX session (coordinate system, joint mapping, smoothing)

## 7. Video Annotation Renderer

- [x] 7.1 Scaffold `packages/video-annotation/` with `__init__.py`, `README.md`, `tests/`
- [x] 7.2 Implement skeleton drawer: joint dots (green/yellow/red by confidence), bone lines
- [x] 7.3 Implement COM plumb line: vertical line from center of mass downward
- [x] 7.4 Implement metrics text overlay: knee flex angle, inclination, COM height % (top-left)
- [x] 7.5 Implement `annotate_video(input_path, poses, output_path)` — draw all frames, encode MP4 with FFmpeg, preserve audio
- [x] 7.6 Write tests for skeleton drawing on synthetic frames, metrics text formatting

## 8. Simple 3D Viewer (React Three Fiber)

- [x] 8.1 Scaffold `packages/viewer-3d/` as TypeScript/React project (Vite + React Three Fiber)
- [x] 8.2 Implement skeleton component: spheres at 14 joints, cylinders for bones
- [x] 8.3 Implement orbit controls (rotate, zoom, pan around skeleton center of mass)
- [x] 8.4 Implement flat grid ground plane at y=0 as spatial reference
- [x] 8.5 Implement frame scrubber / timeline (slider + play/pause)
- [x] 8.6 Implement metrics panel: current frame's knee angle, inclination, COM height
- [x] 8.7 Implement standalone mode: load pose data from local JSON file (no API needed)
- [x] 8.8 Implement API mode: fetch pose data from `/api/results/{job_id}/poses`

## 9. FastAPI Backend

- [x] 9.1 Scaffold `packages/api/` with FastAPI app, Celery worker config, Redis connection
- [x] 9.2 Implement `POST /api/upload` — accept video, create job, return job_id
- [x] 9.3 Implement Celery task: video-pipeline → pose-2d → pose-3d → video-annotation
- [x] 9.4 Implement `GET /api/ws/status/{job_id}` — WebSocket progress (stage + percentage)
- [x] 9.5 Implement `GET /api/results/{job_id}/video` — serve annotated video
- [x] 9.6 Implement `GET /api/results/{job_id}/poses` — serve per-frame Pose3D JSON for viewer
- [x] 9.7 Implement result cleanup (delete after configurable retention period)
- [x] 9.8 Write API tests (upload, status, results, error cases)

## 10. CLI Tool & Sample Data

- [x] 10.1 Implement `PYTHONPATH=packages python -m snowclaw.cli process <video> --output-dir <dir>` — runs full pipeline without web stack
- [x] 10.2 CLI outputs: `annotated.mp4` + `poses.json` (per-frame Pose3D array with timestamps)
- [x] 10.3 Add a sample ski video (5-10s, public domain) to `data/samples/` for testing
- [x] 10.4 Write CLI tests (valid video, invalid path, output directory creation)
- [x] 10.5 Namespace fix: moved CLI package from `packages/cli/` to `packages/snowclaw/`; canonical invocation is `PYTHONPATH=packages python -m snowclaw.cli process <video> --output-dir <dir>`

## 11. Docker & GPU Deployment

- [x] 11.1 Create `Dockerfile` based on NVIDIA CUDA base image with Python, FFmpeg, ONNX Runtime GPU
- [x] 11.2 Add model weight download step to Docker build (pre-warm ViTPose+ and MotionBERT in image)
- [x] 11.3 Create `Dockerfile.frontend` for the 3D viewer (Node.js + Vite build)
- [x] 11.4 Create `docker-compose.yml` — API server, Celery worker (GPU), Redis, frontend. Single `docker compose up`
- [x] 11.5 Configure NVIDIA Container Toolkit GPU passthrough in compose
- [x] 11.6 Add CPU fallback: detect GPU availability, warn if falling back to CPU
- [x] 11.7 Write GPU cloud deployment guide (AWS, GCP, RunPod, Lambda Labs)

## 12. Integration & End-to-End Testing

- [x] 12.1 Integration test: mock backends → annotated video is produced
- [x] 12.2 Integration test: mock backends → 3D viewer receives valid pose JSON
- [x] 12.3 Run full test suite (`pytest -v`) — all packages pass
- [x] 12.4 CLI test: `PYTHONPATH=packages python -m snowclaw.cli process data/samples/ski_demo.mp4 --output-dir ./results/` → verify annotated.mp4 and poses.json
- [ ] 12.5 Viewer test: `npm run dev -- --data ./results/poses.json` → verify skeleton renders, orbit works, scrubber works
- [ ] 12.6 Docker test: `docker compose up` → full stack starts, upload via browser works
- [ ] 12.7 Full end-to-end: upload sample video via web UI → watch annotated video → orbit 3D skeleton
