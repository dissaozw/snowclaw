# SnowClaw Core

Shared data structures for the SnowClaw pipeline. All packages import schema types from `core` to keep the dependency graph clean.

## Types

- **`Pose3D`** — 3D keypoint set for a single person at a single frame (14 body joints + optional equipment keypoints)
- **`Frame`** — Single video frame with pose, timestamp, and scene data
- **`Discipline`** — Enum: `ski` or `snowboard`
- **`TurnPhaseLabel`** — Enum: `initiation`, `fall_line`, `completion`, `transition`
- **`TurnPhase`** — Detected turn phase segment
- **`SessionMetrics`** — Aggregate biomechanical metrics for a session

## Coordinate System

Y-up, right-hand: X=right, Y=up, Z=toward camera. All angles in degrees, all distances in meters.
