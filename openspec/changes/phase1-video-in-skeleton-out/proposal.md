## Why

The biomechanics engine is complete but has no data to analyze — there is no way to go from a raw ski/snowboard video to the `Pose3D` frames it consumes. Phase 1 ("Video In, Skeleton Out") closes this gap by building the first two ML pipeline stages: video preprocessing and pose estimation. Without these, the entire product remains a library with no user-facing capability.

## What Changes

- **New `video-pipeline` package** — FFmpeg-based video preprocessing: frame extraction at configurable FPS, resolution normalization, scene/shot detection, keyframe selection, and metadata extraction. Pure Python + FFmpeg subprocess calls, no ML dependencies.
- **New `pose-estimation` package** — Pluggable backend architecture wrapping 2D pose estimation (ViTPose+) and 3D pose lifting (MotionBERT). Defines abstract `PoseEstimator2D` and `PoseLfiter3D` interfaces so models are swappable. Converts raw model output into the existing `biomechanics.Pose3D` schema, completing the data flow from video frames to biomechanical analysis.
- **Integration glue** — An end-to-end `Pipeline` class that chains video-pipeline → pose-estimation → biomechanics, accepting a video file path and returning `SessionMetrics` + per-frame `Pose3D` data.

## Capabilities

### New Capabilities
- `video-preprocessing`: Extract, normalize, and select frames from raw ski/snowboard video using FFmpeg
- `pose-estimation-2d`: Detect 2D human keypoints from video frames with a pluggable model backend (ViTPose+ default)
- `pose-estimation-3d`: Lift 2D keypoint sequences to 3D poses with temporal consistency (MotionBERT default)
- `pipeline-orchestration`: Chain video → 2D pose → 3D pose → biomechanics into a single callable pipeline

### Modified Capabilities
_(none — biomechanics schemas are consumed as-is, no changes needed)_

## Impact

- **New packages**: `packages/video-pipeline/`, `packages/pose-estimation/`
- **Dependencies**: `video-pipeline` adds ffmpeg-python; `pose-estimation` adds torch, onnxruntime, and model-specific deps as optional extras
- **Existing code**: No modifications to `biomechanics` — new packages produce `Pose3D` objects that feed directly into existing metric functions
- **Infrastructure**: Requires FFmpeg binary on the host system; ML model weights downloaded on first use via a `download_models.py` script
- **pyproject.toml**: Each new package gets its own optional dependency group
