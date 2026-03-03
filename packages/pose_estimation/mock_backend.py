"""
Mock pose estimation backends for local testing and CI.

Use when real ONNX model weights are not available (no GPU, no download).
Produces deterministic, anatomically-plausible synthetic keypoints — good
enough to exercise the full pipeline (video annotation, 3D viewer, CLI).

Enable via CLI flag:
    python -m cli process video.mp4 --mock

Or environment variable:
    SNOWCLAW_MOCK_MODELS=1 python -m cli process video.mp4
"""

from __future__ import annotations

import os

import numpy as np

from core.schemas import Pose3D
from .interfaces import Keypoints2D, PoseEstimator2D, PoseLifter3D
from .joint_mapping import coco_keypoints_to_pose3d


def mock_mode_enabled() -> bool:
    """Return True if mock models are requested via env var."""
    return os.environ.get("SNOWCLAW_MOCK_MODELS", "").strip() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Anatomically-plausible skeleton template (Y-up, standing, metres)
# 17 COCO keypoints in order:
#   0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
#   5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
#   9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
#   13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
# ---------------------------------------------------------------------------
_SKELETON_3D_TEMPLATE = np.array([
    [0.00,  1.75,  0.00],   # 0  nose
    [-0.06, 1.78,  0.00],   # 1  left_eye
    [0.06,  1.78,  0.00],   # 2  right_eye
    [-0.10, 1.76,  0.00],   # 3  left_ear
    [0.10,  1.76,  0.00],   # 4  right_ear
    [-0.20, 1.50,  0.00],   # 5  left_shoulder
    [0.20,  1.50,  0.00],   # 6  right_shoulder
    [-0.35, 1.20,  0.05],   # 7  left_elbow
    [0.35,  1.20,  0.05],   # 8  right_elbow
    [-0.40, 0.90,  0.10],   # 9  left_wrist
    [0.40,  0.90,  0.10],   # 10 right_wrist
    [-0.12, 0.95,  0.00],   # 11 left_hip
    [0.12,  0.95,  0.00],   # 12 right_hip
    [-0.14, 0.52,  0.00],   # 13 left_knee
    [0.14,  0.52,  0.00],   # 14 right_knee
    [-0.14, 0.08,  0.00],   # 15 left_ankle
    [0.14,  0.08,  0.00],   # 16 right_ankle
], dtype=np.float32)

# Ski athletic stance adjustments
_SKI_STANCE_DELTA = np.array([
    [0.00,  0.00,  0.00],   # nose
    [0.00,  0.00,  0.00],   # left_eye
    [0.00,  0.00,  0.00],   # right_eye
    [0.00,  0.00,  0.00],   # left_ear
    [0.00,  0.00,  0.00],   # right_ear
    [0.00, -0.05,  0.02],   # left_shoulder  (slightly lowered, forward)
    [0.00, -0.05,  0.02],   # right_shoulder
    [0.00, -0.05,  0.05],   # left_elbow     (forward)
    [0.00, -0.05,  0.05],   # right_elbow
    [0.00,  0.00,  0.08],   # left_wrist
    [0.00,  0.00,  0.08],   # right_wrist
    [0.00, -0.05,  0.03],   # left_hip       (hips back)
    [0.00, -0.05,  0.03],   # right_hip
    [0.00, -0.10,  0.04],   # left_knee      (knee flex)
    [0.00, -0.10,  0.04],   # right_knee
    [0.00,  0.00,  0.00],   # left_ankle
    [0.00,  0.00,  0.00],   # right_ankle
], dtype=np.float32)

_SKELETON_3D_SKI = _SKELETON_3D_TEMPLATE + _SKI_STANCE_DELTA


def _project_3d_to_2d(
    points_3d: np.ndarray,
    image_size: tuple[int, int],
    jitter_px: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Project 3D skeleton template onto 2D image plane.

    Assumes camera is at Z+ looking toward Z- (frontal view).
    X → horizontal, Y → vertical (top=high Y).

    Args:
        points_3d:  (17, 3) skeleton in world coords.
        image_size: (H, W) of target image.
        jitter_px:  Gaussian noise in pixels for realism.
        rng:        Optional random generator (for reproducibility).

    Returns:
        (17, 2) pixel coordinates.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    h, w = image_size
    cx, cy = w / 2, h / 2

    # Simple orthographic projection (X, Y) → pixel
    scale = min(w, h) * 0.35  # fits skeleton in ~70% of the frame
    px = cx + points_3d[:, 0] * scale
    py = cy - points_3d[:, 1] * scale  # flip Y (image Y down)

    if jitter_px > 0:
        px += rng.normal(0, jitter_px, size=17)
        py += rng.normal(0, jitter_px, size=17)

    points_2d = np.stack([px, py], axis=1)
    return points_2d.astype(np.float32)


class MockViTPoseBackend(PoseEstimator2D):
    """
    Mock 2D keypoint detector — no model download, no GPU required.

    Generates deterministic, anatomically-plausible 2D keypoints by projecting
    a 3D ski-stance skeleton template onto each frame's image plane with slight
    per-frame jitter to simulate realistic motion.
    """

    def __init__(self, image_size: tuple[int, int] | None = None, jitter_px: float = 3.0) -> None:
        """
        Args:
            image_size: Override image size for projection. If None, uses actual frame shape.
            jitter_px:  Per-frame pixel jitter for realistic motion.
        """
        self._image_size = image_size
        self._jitter_px = jitter_px

    def predict(self, frames: list[np.ndarray]) -> list[Keypoints2D]:
        """
        Generate mock 2D keypoints for each frame.

        Args:
            frames: List of RGB images (H, W, 3) uint8.

        Returns:
            List of Keypoints2D, one per frame, with confidence=0.9 for all joints.
        """
        results = []
        for i, frame in enumerate(frames):
            h, w = (self._image_size or frame.shape[:2])
            rng = np.random.default_rng(seed=i * 17)  # deterministic per frame

            points_2d = _project_3d_to_2d(
                _SKELETON_3D_SKI, (h, w), jitter_px=self._jitter_px, rng=rng
            )
            confidence = np.full(17, 0.92, dtype=np.float32)
            # Add slight variation in confidence
            confidence += rng.uniform(-0.05, 0.05, size=17).astype(np.float32)
            confidence = np.clip(confidence, 0.0, 1.0)

            results.append(Keypoints2D(
                points=points_2d,
                confidence=confidence,
                image_size=(h, w),
            ))
        return results


class MockMotionBERTBackend(PoseLifter3D):
    """
    Mock 3D pose lifter — no model download, no GPU required.

    Converts mock 2D keypoints back to 3D using the original ski-stance template,
    with small per-frame oscillation to simulate dynamic movement (weight transfer,
    knee flex, etc.).
    """

    def lift(self, keypoints_2d: list[Keypoints2D]) -> list[Pose3D]:
        """
        Generate mock 3D poses from 2D keypoints.

        Args:
            keypoints_2d: List of Keypoints2D (typically from MockViTPoseBackend).

        Returns:
            List of Pose3D, one per frame.
        """
        poses = []
        n = len(keypoints_2d)

        for i, kp in enumerate(keypoints_2d):
            # Oscillate the skeleton slightly over time (weight transfer simulation)
            t = i / max(n - 1, 1) * 2 * np.pi
            sway_x = np.sin(t) * 0.04       # lateral sway ±4 cm
            flex_y = np.cos(t * 2) * 0.02   # COM height oscillation ±2 cm
            depth_z = np.sin(t * 0.5) * 0.03  # slight fore-aft shift

            offset = np.array([sway_x, flex_y, depth_z], dtype=np.float32)
            joints_3d = _SKELETON_3D_SKI + offset[np.newaxis, :]

            # Build confidence dict from 2D confidence (propagated to 3D joints)
            conf_arr = kp.confidence  # (17,)

            pose3d = coco_keypoints_to_pose3d(joints_3d, conf_arr)
            poses.append(pose3d)

        return poses
