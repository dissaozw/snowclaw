"""COCO-to-Pose3D joint mapping (17 COCO joints -> 14 core Pose3D joints)."""

from __future__ import annotations

import numpy as np

from core.schemas import Pose3D

# COCO keypoint indices (17 joints):
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

# Mapping from Pose3D joint name -> COCO indices to average
# "head" comes from nose (0), "neck" is derived from shoulders midpoint
COCO_TO_POSE3D: dict[str, list[int]] = {
    "head": [0],             # nose
    "neck": [5, 6],          # midpoint of left_shoulder + right_shoulder
    "left_shoulder": [5],
    "right_shoulder": [6],
    "left_elbow": [7],
    "right_elbow": [8],
    "left_wrist": [9],
    "right_wrist": [10],
    "left_hip": [11],
    "right_hip": [12],
    "left_knee": [13],
    "right_knee": [14],
    "left_ankle": [15],
    "right_ankle": [16],
}


def coco_keypoints_to_pose3d(
    joints_3d: np.ndarray,
    confidence: np.ndarray | None = None,
) -> Pose3D:
    """
    Convert 17 COCO 3D joint positions to a core.Pose3D.

    Args:
        joints_3d: Array of shape (17, 3) with 3D positions in meters (Y-up).
        confidence: Optional array of shape (17,) with per-joint confidence.

    Returns:
        Pose3D with 14 core body joints populated.
    """
    if joints_3d.shape != (17, 3):
        raise ValueError(f"Expected shape (17, 3), got {joints_3d.shape}")

    pose_data: dict[str, list[float]] = {}
    conf_data: dict[str, float] = {}

    for joint_name, coco_indices in COCO_TO_POSE3D.items():
        # Average positions from mapped COCO joints
        pos = np.mean(joints_3d[coco_indices], axis=0)
        pose_data[joint_name] = pos.tolist()

        if confidence is not None:
            conf_data[joint_name] = float(np.mean(confidence[coco_indices]))

    return Pose3D(
        **pose_data,
        confidence=conf_data if conf_data else None,
    )
