"""
SnowClaw Pose Estimation — pluggable 2D keypoint detection and 3D lifting.

Provides abstract interfaces (PoseEstimator2D, PoseLifter3D) and concrete
backends (ViTPose+, MotionBERT) using ONNX Runtime.
"""

from .interfaces import Keypoints2D, PoseEstimator2D, PoseLifter3D
from .joint_mapping import COCO_TO_POSE3D, coco_keypoints_to_pose3d
from .subject_tracker import SubjectTracker
from .model_cache import download_model

__all__ = [
    "Keypoints2D",
    "PoseEstimator2D",
    "PoseLifter3D",
    "COCO_TO_POSE3D",
    "coco_keypoints_to_pose3d",
    "SubjectTracker",
    "download_model",
]
