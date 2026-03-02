## ADDED Requirements

### Requirement: Abstract 3D pose lifter interface
The system SHALL define a `PoseLifter3D` abstract base class with a `lift` method that accepts a list of `Keypoints2D` and returns a list of `core.Pose3D` objects.

#### Scenario: Implementing a custom 3D lifter
- **WHEN** a developer subclasses `PoseLifter3D` and implements `lift`
- **THEN** the subclass SHALL be usable anywhere the base class is accepted

### Requirement: MotionBERT backend
The system SHALL provide a `MotionBERTBackend` implementation of `PoseLifter3D` that lifts 2D keypoint sequences to 3D using MotionBERT via ONNX Runtime.

#### Scenario: Lift a sequence of 2D keypoints to 3D
- **WHEN** a list of `Keypoints2D` objects from a video sequence is passed to `MotionBERTBackend.lift()`
- **THEN** a list of `core.Pose3D` objects SHALL be returned, one per frame, with joint positions in meters using the Y-up coordinate system

#### Scenario: Temporal consistency
- **WHEN** a sequence of 2D keypoints is lifted
- **THEN** the resulting 3D poses SHALL exhibit temporal smoothness (no single-frame jitter) due to MotionBERT's temporal transformer architecture

### Requirement: Coordinate system conversion
The 3D lifter SHALL output joint positions in the project's Y-up, right-hand coordinate system (X=right, Y=up, Z=toward camera) with positions in meters.

#### Scenario: Coordinate system compliance
- **WHEN** `Pose3D` objects are returned from any `PoseLifter3D` backend
- **THEN** all joint positions SHALL use the Y-up coordinate system defined in `core.schemas`

### Requirement: Joint mapping to Pose3D
The system SHALL map model-specific joint indices to the 14 core body joints defined in `core.Pose3D` (head, neck, left/right shoulder, elbow, wrist, hip, knee, ankle).

#### Scenario: COCO-to-Pose3D mapping
- **WHEN** MotionBERT outputs 17 COCO-format 3D joints
- **THEN** the backend SHALL map them to the 14 `Pose3D` core joints, deriving `neck` from the midpoint of left/right shoulder if not directly available

### Requirement: Confidence propagation
The system SHALL propagate per-joint confidence scores from 2D detection through to the `Pose3D.confidence` field.

#### Scenario: Low-confidence joint handling
- **WHEN** a 2D keypoint has confidence below 0.3
- **THEN** the corresponding 3D joint SHALL still be present in `Pose3D` but its confidence value SHALL reflect the low 2D confidence

### Requirement: Model weight management
The system SHALL download MotionBERT weights on first use to `~/.cache/snowclaw/models/` and reuse cached weights on subsequent runs.

#### Scenario: First-time model load
- **WHEN** `MotionBERTBackend` is instantiated and weights are not cached
- **THEN** weights SHALL be downloaded with a progress indicator and saved to the cache directory
