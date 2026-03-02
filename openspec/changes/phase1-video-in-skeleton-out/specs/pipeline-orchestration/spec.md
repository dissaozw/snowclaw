## ADDED Requirements

### Requirement: End-to-end pipeline class
The system SHALL provide a `Pipeline` class that chains video preprocessing → 2D pose estimation → 3D lifting into a single callable interface. The pipeline SHALL NOT depend on or invoke the `biomechanics` package.

#### Scenario: Process a video file end-to-end
- **WHEN** `Pipeline.run(video_path)` is called with a valid ski video
- **THEN** the system SHALL return a `PipelineResult` containing: a list of `core.Pose3D` objects (one per extracted frame), video metadata, and frame timestamps

#### Scenario: Invalid video path
- **WHEN** `Pipeline.run()` is called with a non-existent path
- **THEN** a `VideoProcessingError` SHALL be raised

### Requirement: Configurable pipeline components
The system SHALL allow users to configure which backends are used by passing custom `PoseEstimator2D` and `PoseLifter3D` instances to the `Pipeline` constructor.

#### Scenario: Use default backends
- **WHEN** `Pipeline()` is constructed with no arguments
- **THEN** it SHALL use `ViTPoseBackend` for 2D and `MotionBERTBackend` for 3D

#### Scenario: Use custom backends
- **WHEN** `Pipeline(estimator_2d=MyCustom2D(), lifter_3d=MyCustom3D())` is constructed
- **THEN** the pipeline SHALL use the provided custom backends

### Requirement: Pipeline progress reporting
The system SHALL report progress through each pipeline stage via a callback or iterator.

#### Scenario: Track progress with callback
- **WHEN** `Pipeline.run(video_path, on_progress=callback)` is called
- **THEN** the callback SHALL be invoked with stage name and percentage for each pipeline stage (extracting frames, estimating 2D pose, lifting to 3D)

### Requirement: Pipeline configuration
The system SHALL accept a `PipelineConfig` that controls: target FPS, max resolution, 2D model batch size, and discipline (ski/snowboard).

#### Scenario: Configure for a specific FPS
- **WHEN** `PipelineConfig(target_fps=15, max_dimension=1280)` is provided
- **THEN** the pipeline SHALL extract frames at 15 FPS and resize to max 1280px

### Requirement: Pipeline output is independent of biomechanics
The `PipelineResult` SHALL use only types from `core` (e.g., `core.Pose3D`, `core.Discipline`). Biomechanical analysis is a separate downstream step composed by the caller.

#### Scenario: Compose with biomechanics externally
- **WHEN** a caller receives `PipelineResult` from `Pipeline.run()`
- **THEN** they SHALL be able to pass the `Pose3D` objects to `biomechanics` metric functions without any intermediate conversion — both use `core.Pose3D`
