## 1. Core Package — Setup & Shared Schemas

- [ ] 1.1 Scaffold `packages/core/` with `__init__.py`, `schemas.py`, `README.md`, and `tests/` directory
- [ ] 1.2 Move `Pose3D`, `Frame`, `Discipline`, `TurnPhaseLabel`, `TurnPhase`, `SessionMetrics` from `biomechanics/schemas.py` into `core/schemas.py`
- [ ] 1.3 Update `core/__init__.py` to export all shared types
- [ ] 1.4 Write tests for `core` schemas (Pose3D construction, Frame validation, enum values)

## 2. Refactor Biomechanics — Import from Core

- [ ] 2.1 Replace `biomechanics/schemas.py` with imports from `core.schemas` (re-export for backward compatibility)
- [ ] 2.2 Update all internal `biomechanics` imports (metrics.py, snow_iq.py, turn_segmentation.py) to use `core` types
- [ ] 2.3 Update `biomechanics/__init__.py` to re-export `Pose3D`, `Frame`, `TurnPhase`, `SessionMetrics` from `core`
- [ ] 2.4 Run `pytest packages/biomechanics/` — verify all existing tests pass unchanged

## 3. Video Pipeline Package — Setup

- [ ] 3.1 Scaffold `packages/video-pipeline/` with `__init__.py`, `README.md`, and `tests/` directory
- [ ] 3.2 Add `video-pipeline` optional dependency group to root `pyproject.toml`
- [ ] 3.3 Implement `ffmpeg_check()` utility that verifies FFmpeg is on PATH, raises `DependencyError` if not

## 4. Video Pipeline Package — Core

- [ ] 4.1 Implement `VideoMetadata` dataclass with fields: duration_s, fps, width, height, codec, frame_count
- [ ] 4.2 Implement `extract_metadata(video_path) -> VideoMetadata` using `ffprobe` subprocess
- [ ] 4.3 Implement `extract_frames(video_path, target_fps, max_dimension) -> list[np.ndarray]` using FFmpeg subprocess — returns RGB frames as NumPy arrays
- [ ] 4.4 Implement resolution normalization (downscale to max_dimension preserving aspect ratio, no upscaling)
- [ ] 4.5 Implement `select_keyframes(frames, threshold) -> list[np.ndarray]` using normalized frame differencing
- [ ] 4.6 Define `VideoProcessingError` and `DependencyError` exception classes

## 5. Video Pipeline Package — Tests

- [ ] 5.1 Write tests for `extract_metadata` with a small test video fixture
- [ ] 5.2 Write tests for `extract_frames` (default FPS, custom FPS, invalid file)
- [ ] 5.3 Write tests for resolution normalization (downscale, no-upscale cases)
- [ ] 5.4 Write tests for `select_keyframes` (threshold filtering)
- [ ] 5.5 Write tests for `ffmpeg_check` (available and missing scenarios)

## 6. Pose Estimation Package — Setup

- [ ] 6.1 Scaffold `packages/pose-estimation/` with `__init__.py`, `README.md`, and `tests/` directory
- [ ] 6.2 Add `pose-estimation` optional dependency groups to root `pyproject.toml` (`[onnx]` and `[torch]` extras)
- [ ] 6.3 Implement model weight download utility: `download_weights(model_name, url, cache_dir)` with progress bar and cache check

## 7. Pose Estimation Package — Data Structures & Interfaces

- [ ] 7.1 Define `Keypoints2D` dataclass (points: ndarray Jx2, confidence: ndarray J, image_size: tuple) — internal to pose-estimation
- [ ] 7.2 Define `PoseEstimator2D` ABC with `predict(frames) -> list[Keypoints2D]`
- [ ] 7.3 Define `PoseLifter3D` ABC with `lift(keypoints_2d) -> list[core.Pose3D]` — imports Pose3D from core, not biomechanics
- [ ] 7.4 Implement COCO-to-Pose3D joint mapping utility (17 COCO joints → 14 core.Pose3D joints, derive neck from shoulder midpoint)

## 8. Pose Estimation Package — ViTPose+ Backend

- [ ] 8.1 Implement `ViTPoseBackend(PoseEstimator2D)` with ONNX Runtime inference
- [ ] 8.2 Add preprocessing: resize input frames, normalize, build input tensor
- [ ] 8.3 Add postprocessing: decode heatmaps to pixel coordinates, extract confidence scores, return `Keypoints2D`
- [ ] 8.4 Implement batch processing (split frames into batches of configurable size)
- [ ] 8.5 Implement device auto-detection (CUDA if available, else CPU)

## 9. Pose Estimation Package — MotionBERT Backend

- [ ] 9.1 Implement `MotionBERTBackend(PoseLifter3D)` with ONNX Runtime inference
- [ ] 9.2 Add preprocessing: normalize 2D keypoints to [-1, 1] range, assemble temporal windows
- [ ] 9.3 Add postprocessing: convert output to meters in Y-up coordinate system, map to core.Pose3D joints
- [ ] 9.4 Implement confidence propagation from 2D keypoints to core.Pose3D.confidence

## 10. Pose Estimation Package — Tests

- [ ] 10.1 Write unit tests for `Keypoints2D` dataclass
- [ ] 10.2 Write unit tests for COCO-to-Pose3D joint mapping
- [ ] 10.3 Write tests for `ViTPoseBackend` with mock ONNX session (verify input/output shapes, batching)
- [ ] 10.4 Write tests for `MotionBERTBackend` with mock ONNX session (verify coordinate system, joint mapping)
- [ ] 10.5 Write tests for model weight download utility (cache hit, cache miss)

## 11. Pipeline Orchestration

- [ ] 11.1 Define `PipelineConfig` dataclass (target_fps, max_dimension, batch_size, discipline, device) — uses `core.Discipline`
- [ ] 11.2 Implement `Pipeline` class with configurable 2D/3D backends and default fallbacks — does NOT depend on biomechanics
- [ ] 11.3 Implement `Pipeline.run(video_path, on_progress) -> PipelineResult` chaining video → 2D → 3D stages only
- [ ] 11.4 Define `PipelineResult` dataclass (poses: list[core.Pose3D], timestamps: list[float], metadata: VideoMetadata)
- [ ] 11.5 Wire progress callback through each pipeline stage

## 12. Integration Tests

- [ ] 12.1 Write integration test: Pipeline with mock backends, verify full data flow from video path to PipelineResult with core.Pose3D objects
- [ ] 12.2 Write test for custom backend injection
- [ ] 12.3 Write test for progress callback invocation
- [ ] 12.4 Write test: PipelineResult.poses are compatible with biomechanics metric functions (both use core.Pose3D)
- [ ] 12.5 Run full test suite (`pytest -v`) and verify all tests pass including existing biomechanics tests
