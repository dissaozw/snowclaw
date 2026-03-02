"""MotionBERT 3D pose lifting backend using ONNX Runtime."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from core.schemas import Pose3D

from .interfaces import Keypoints2D, PoseLifter3D
from .joint_mapping import coco_keypoints_to_pose3d

logger = logging.getLogger(__name__)

# MotionBERT config
MOTIONBERT_WINDOW_SIZE = 243  # Temporal window (frames)
MOTIONBERT_NUM_JOINTS = 17


def _normalize_keypoints_2d(
    keypoints: Keypoints2D,
) -> np.ndarray:
    """
    Normalize 2D keypoints to [-1, 1] range based on image size.

    Args:
        keypoints: 2D detection result.

    Returns:
        Normalized keypoints of shape (17, 2).
    """
    h, w = keypoints.image_size
    normalized = keypoints.points.copy().astype(np.float32)
    normalized[:, 0] = normalized[:, 0] / w * 2.0 - 1.0  # x to [-1, 1]
    normalized[:, 1] = normalized[:, 1] / h * 2.0 - 1.0  # y to [-1, 1]
    return normalized


def _assemble_temporal_window(
    keypoints_list: list[np.ndarray],
    window_size: int,
    center_idx: int,
) -> np.ndarray:
    """
    Assemble a temporal window centered on center_idx, padding with edge frames.

    Args:
        keypoints_list: List of normalized keypoints, each shape (17, 2).
        window_size: Target window size.
        center_idx: Center frame index.

    Returns:
        Array of shape (window_size, 17, 2).
    """
    n = len(keypoints_list)
    half = window_size // 2
    start = center_idx - half
    end = start + window_size

    indices = list(range(start, end))
    # Clamp to valid range (pad with edge frames)
    indices = [max(0, min(i, n - 1)) for i in indices]

    return np.stack([keypoints_list[i] for i in indices], axis=0)


def _apply_temporal_smoothing(
    joints_3d: np.ndarray,
    window_length: int = 15,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to smooth 3D joint trajectories.

    Args:
        joints_3d: Array of shape (N, 17, 3) — per-frame 3D positions.
        window_length: SG filter window (must be odd, > polyorder).
        polyorder: Polynomial order for SG filter.

    Returns:
        Smoothed array of same shape.
    """
    n = joints_3d.shape[0]
    if n < window_length:
        window_length = n if n % 2 == 1 else max(n - 1, polyorder + 1)
    if window_length <= polyorder:
        return joints_3d  # Too few frames to smooth

    smoothed = joints_3d.copy()
    num_joints = joints_3d.shape[1]

    for j in range(num_joints):
        for c in range(3):  # x, y, z
            smoothed[:, j, c] = savgol_filter(
                joints_3d[:, j, c],
                window_length=window_length,
                polyorder=polyorder,
            )

    return smoothed


class MotionBERTBackend(PoseLifter3D):
    """
    MotionBERT 3D pose lifting using ONNX Runtime.

    Takes 2D keypoints and produces 3D poses in the Y-up coordinate system.
    Includes temporal smoothing for stable skeleton output.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "auto",
        smoothing_window: int = 15,
        smoothing_poly: int = 3,
    ):
        """
        Args:
            model_path: Path to MotionBERT ONNX model. Downloads if None.
            device: "auto", "cuda", or "cpu".
            smoothing_window: Savitzky-Golay window length for temporal smoothing.
            smoothing_poly: SG polynomial order.
        """
        self._model_path = model_path
        self._session = None
        self._smoothing_window = smoothing_window
        self._smoothing_poly = smoothing_poly

        from .vitpose_backend import _get_onnx_providers
        self._providers = _get_onnx_providers(device)

    def _get_session(self):
        """Lazy-load ONNX session."""
        if self._session is not None:
            return self._session

        import onnxruntime as ort

        if self._model_path is None:
            from .model_cache import download_model
            self._model_path = download_model(
                url="https://huggingface.co/snowclaw/motionbert/resolve/main/motionbert_lite.onnx",
                filename="motionbert_lite.onnx",
            )

        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=self._providers,
        )
        return self._session

    def lift(self, keypoints_2d: list[Keypoints2D]) -> list[Pose3D]:
        """
        Lift 2D keypoints to 3D poses using MotionBERT.

        Args:
            keypoints_2d: List of 2D detections from PoseEstimator2D.

        Returns:
            List of core.Pose3D, one per frame. Temporally smoothed.
        """
        if not keypoints_2d:
            return []

        session = self._get_session()
        input_name = session.get_inputs()[0].name

        # Normalize all 2D keypoints
        normalized = [_normalize_keypoints_2d(kp) for kp in keypoints_2d]
        n_frames = len(normalized)

        # Run inference frame-by-frame with temporal windows
        all_joints_3d = np.zeros((n_frames, MOTIONBERT_NUM_JOINTS, 3), dtype=np.float32)

        for i in range(n_frames):
            window = _assemble_temporal_window(
                normalized, MOTIONBERT_WINDOW_SIZE, i
            )
            # Shape: (1, window_size, 17, 2)
            input_data = window[np.newaxis].astype(np.float32)
            outputs = session.run(None, {input_name: input_data})

            # Output shape: (1, window_size, 17, 3)
            # Take the center frame prediction
            center = MOTIONBERT_WINDOW_SIZE // 2
            all_joints_3d[i] = outputs[0][0, center]

        # Apply temporal smoothing
        all_joints_3d = _apply_temporal_smoothing(
            all_joints_3d,
            window_length=self._smoothing_window,
            polyorder=self._smoothing_poly,
        )

        # Convert to Pose3D objects with confidence propagation
        poses: list[Pose3D] = []
        for i in range(n_frames):
            confidence_2d = keypoints_2d[i].confidence
            pose = coco_keypoints_to_pose3d(all_joints_3d[i], confidence_2d)
            poses.append(pose)

        return poses
