## Why

The biomechanics engine is complete but has no data to analyze — there is no way to go from a raw ski/snowboard video to the `Pose3D` frames it consumes. Phase 1 ("Video In, Skeleton Out") closes this gap by building the first two ML pipeline stages: video preprocessing and pose estimation.

Additionally, shared data structures like `Pose3D`, `Frame`, `Discipline`, and `TurnPhaseLabel` currently live inside the `biomechanics` package. This creates an undesirable coupling — any package that produces or consumes poses would need to depend on `biomechanics`. These shared types belong in a `core` package (already planned in AGENTS.md) so that each pipeline stage is independently importable with no cross-package dependencies.

## What Changes

- **New `core` package** — Extract shared data structures (`Pose3D`, `Frame`, `Discipline`, `TurnPhaseLabel`, `SessionMetrics`, `TurnPhase`) from `biomechanics.schemas` into `packages/core/`. This becomes the common type foundation that all packages import from. Dependencies: numpy, pydantic only.
- **Refactor `biomechanics` package** — Replace locally-defined schemas with imports from `core`. No logic changes, only import paths change.
- **New `video-pipeline` package** — FFmpeg-based video preprocessing: frame extraction at configurable FPS, resolution normalization, keyframe selection, and metadata extraction. Pure Python + FFmpeg subprocess calls, no ML dependencies. Does not depend on `core` or `biomechanics`.
- **New `pose-estimation` package** — Pluggable backend architecture wrapping 2D pose estimation (ViTPose+) and 3D pose lifting (MotionBERT). Defines abstract `PoseEstimator2D` and `PoseLifter3D` interfaces so models are swappable. Outputs `core.Pose3D` objects. Depends on `core` only — does **not** depend on `biomechanics`.
- **Integration glue** — An end-to-end `Pipeline` class that chains video-pipeline → pose-estimation, accepting a video file path and returning per-frame `core.Pose3D` data. Biomechanical analysis is a separate downstream step, not baked into the pipeline.

## Capabilities

### New Capabilities
- `core-schemas`: Shared data structures (Pose3D, Frame, Discipline, etc.) extracted from biomechanics into an independent core package
- `video-preprocessing`: Extract, normalize, and select frames from raw ski/snowboard video using FFmpeg
- `pose-estimation-2d`: Detect 2D human keypoints from video frames with a pluggable model backend (ViTPose+ default)
- `pose-estimation-3d`: Lift 2D keypoint sequences to 3D poses with temporal consistency (MotionBERT default)
- `pipeline-orchestration`: Chain video → 2D pose → 3D pose into a single callable pipeline

### Modified Capabilities
- `biomechanics`: Refactor to import shared schemas from `core` instead of defining them locally. No behavioral changes.

## Impact

- **New packages**: `packages/core/`, `packages/video-pipeline/`, `packages/pose-estimation/`
- **Modified packages**: `packages/biomechanics/` — `schemas.py` replaced with imports from `core`
- **Dependencies**: `core` depends on numpy + pydantic. `video-pipeline` has zero Python deps (FFmpeg subprocess only). `pose-estimation` depends on `core` + onnxruntime (with optional torch extra). `biomechanics` adds `core` as a dependency.
- **Infrastructure**: Requires FFmpeg binary on the host system; ML model weights downloaded on first use
- **pyproject.toml**: Each new package gets its own optional dependency group
