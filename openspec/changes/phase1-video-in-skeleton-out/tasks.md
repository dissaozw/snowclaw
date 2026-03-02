## 1. Video Pipeline Package — Setup

- [ ] 1.1 Scaffold `packages/video-pipeline/` with `__init__.py`, `README.md`, and `tests/` directory
- [ ] 1.2 Add `video-pipeline` optional dependency group to root `pyproject.toml`
- [ ] 1.3 Implement `ffmpeg_check()` utility that verifies FFmpeg is on PATH, raises `DependencyError` if not

## 2. Video Pipeline Package — Core

- [ ] 2.1 Implement `VideoMetadata` dataclass with fields: duration_s, fps, width, height, codec, frame_count
- [ ] 2.2 Implement `extract_metadata(video_path) -> VideoMetadata` using `ffprobe` subprocess
- [ ] 2.3 Implement `extract_frames(video_path, target_fps, max_dimension) -> list[np.ndarray]` using FFmpeg subprocess — returns RGB frames as NumPy arrays
- [ ] 2.4 Implement resolution normalization (downscale to max_dimension preserving aspect ratio, no upscaling)
- [ ] 2.5 Implement `select_keyframes(frames, threshold) -> list[np.ndarray]` using normalized frame differencing
- [ ] 2.6 Define `VideoProcessingError` exception class

## 3. Video Pipeline Package — Tests

- [ ] 3.1 Write tests for `extract_metadata` with a small test video fixture
- [ ] 3.2 Write tests for `extract_frames` (default FPS, custom FPS, invalid file)
- [ ] 3.3 Write tests for resolution normalization (downscale, no-upscale cases)
- [ ] 3.4 Write tests for `select_keyframes` (threshold filtering)
- [ ] 3.5 Write tests for `ffmpeg_check` (available and missing scenarios)

## 4. Pose Estimation Package — Setup

- [ ] 4.1 Scaffold `packages/pose-estimation/` with `__init__.py`, `README.md`, and `tests/` directory
- [ ] 4.2 Add `pose-estimation` optional dependency groups to root `pyproject.toml` (`[onnx]` and `[torch]` extras)
- [ ] 4.3 Implement model weight download utility: `download_weights(model_name, url, cache_dir)` with progress bar and cache check

## 5. Pose Estimation Package — Data Structures & Interfaces

- [ ] 5.1 Define `Keypoints2D` dataclass (points: ndarray Jx2, confidence: ndarray J, image_size: tuple)
- [ ] 5.2 Define `PoseEstimator2D` ABC with `predict(frames) -> list[Keypoints2D]`
- [ ] 5.3 Define `PoseLifter3D` ABC with `lift(keypoints_2d) -> list[Pose3D]`
- [ ] 5.4 Implement COCO-to-Pose3D joint mapping utility (17 COCO joints → 14 Pose3D joints, derive neck from shoulder midpoint)

## 6. Pose Estimation Package — ViTPose+ Backend

- [ ] 6.1 Implement `ViTPoseBackend(PoseEstimator2D)` with ONNX Runtime inference
- [ ] 6.2 Add preprocessing: resize input frames, normalize, build input tensor
- [ ] 6.3 Add postprocessing: decode heatmaps to pixel coordinates, extract confidence scores, return `Keypoints2D`
- [ ] 6.4 Implement batch processing (split frames into batches of configurable size)
- [ ] 6.5 Implement device auto-detection (CUDA if available, else CPU)

## 7. Pose Estimation Package — MotionBERT Backend

- [ ] 7.1 Implement `MotionBERTBackend(PoseLifter3D)` with ONNX Runtime inference
- [ ] 7.2 Add preprocessing: normalize 2D keypoints to [-1, 1] range, assemble temporal windows
- [ ] 7.3 Add postprocessing: convert output to meters in Y-up coordinate system, map to Pose3D joints
- [ ] 7.4 Implement confidence propagation from 2D keypoints to Pose3D.confidence

## 8. Pose Estimation Package — Tests

- [ ] 8.1 Write unit tests for `Keypoints2D` dataclass
- [ ] 8.2 Write unit tests for COCO-to-Pose3D joint mapping
- [ ] 8.3 Write tests for `ViTPoseBackend` with mock ONNX session (verify input/output shapes, batching)
- [ ] 8.4 Write tests for `MotionBERTBackend` with mock ONNX session (verify coordinate system, joint mapping)
- [ ] 8.5 Write tests for model weight download utility (cache hit, cache miss)

## 9. Pipeline Orchestration

- [ ] 9.1 Define `PipelineConfig` dataclass (target_fps, max_dimension, batch_size, discipline, device)
- [ ] 9.2 Implement `Pipeline` class with configurable 2D/3D backends and default fallbacks
- [ ] 9.3 Implement `Pipeline.run(video_path, on_progress) -> PipelineResult` chaining all stages
- [ ] 9.4 Define `PipelineResult` dataclass (frames: list[Frame], metrics: SessionMetrics, turns: list[TurnPhase])
- [ ] 9.5 Wire progress callback through each pipeline stage

## 10. Pipeline Integration Tests

- [ ] 10.1 Write integration test: Pipeline with mock backends, verify full data flow from video path to PipelineResult
- [ ] 10.2 Write test for custom backend injection
- [ ] 10.3 Write test for progress callback invocation
- [ ] 10.4 Run full test suite (`pytest -v`) and verify all tests pass
