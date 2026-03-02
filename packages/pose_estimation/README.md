# SnowClaw Pose Estimation

Pluggable 2D keypoint detection and 3D pose lifting for the SnowClaw pipeline.

## Interfaces

- **`PoseEstimator2D`** (ABC) — Detect 2D keypoints in video frames
- **`PoseLifter3D`** (ABC) — Lift 2D keypoints to 3D poses
- **`Keypoints2D`** — Dataclass for 2D detection results

## Backends

- **ViTPose+** — 2D keypoint detection via ONNX Runtime
- **MotionBERT** — Temporal 3D lifting via ONNX Runtime

## Joint Mapping

Maps 17 COCO keypoints to 14 core Pose3D joints. Neck is derived from shoulder midpoint.
