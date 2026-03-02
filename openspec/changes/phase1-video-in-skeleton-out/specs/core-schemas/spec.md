## ADDED Requirements

### Requirement: Core package with shared data structures
The system SHALL provide a `packages/core/` package containing all shared data structures used across multiple packages. This package SHALL depend only on numpy and pydantic — no ML frameworks, no biomechanics logic.

#### Scenario: Import Pose3D from core
- **WHEN** any package needs the `Pose3D` type
- **THEN** it SHALL import from `core` (e.g., `from core.schemas import Pose3D`), not from `biomechanics`

### Requirement: Pose3D model in core
The `core` package SHALL define the `Pose3D` Pydantic v2 model with 14 core body joints (head, neck, left/right shoulder, elbow, wrist, hip, knee, ankle), optional equipment keypoints (ski tips/tails, board nose/tail), per-joint confidence scores, and helper properties (com, hip_midpoint, shoulder_midpoint, ankle_midpoint, body_height).

#### Scenario: Create a Pose3D from 3D lifter output
- **WHEN** a pose estimation backend produces 3D joint positions
- **THEN** it SHALL construct a `core.Pose3D` with all 14 core joints in Y-up coordinates (meters)

#### Scenario: Backward compatibility with biomechanics
- **WHEN** existing code imports `from biomechanics import Pose3D`
- **THEN** it SHALL continue to work because `biomechanics` re-exports `Pose3D` from `core`

### Requirement: Frame model in core
The `core` package SHALL define the `Frame` Pydantic v2 model with fields: frame_idx, timestamp_s, pose (Pose3D), snow_surface_normal, and discipline.

#### Scenario: Construct a Frame
- **WHEN** a pipeline produces a Pose3D for a given video frame
- **THEN** a `core.Frame` SHALL be constructable with the pose, frame index, and timestamp

### Requirement: Enums in core
The `core` package SHALL define `Discipline` (ski, snowboard) and `TurnPhaseLabel` (initiation, fall_line, completion, transition) enums.

#### Scenario: Use Discipline enum across packages
- **WHEN** `video-pipeline` or `pose-estimation` needs to reference the discipline
- **THEN** it SHALL use `core.Discipline` without importing `biomechanics`

### Requirement: TurnPhase and SessionMetrics in core
The `core` package SHALL define `TurnPhase` and `SessionMetrics` Pydantic v2 models, moved from `biomechanics.schemas`.

#### Scenario: Biomechanics produces SessionMetrics
- **WHEN** the biomechanics engine computes aggregate metrics
- **THEN** it SHALL return a `core.SessionMetrics` instance

### Requirement: Biomechanics refactored to import from core
The `biomechanics` package SHALL remove its local schema definitions and instead import all shared types from `core`. It SHALL re-export them from `biomechanics.__init__` for backward compatibility.

#### Scenario: Existing biomechanics tests pass unchanged
- **WHEN** `pytest packages/biomechanics/` is run after the refactor
- **THEN** all existing tests SHALL pass without modification
