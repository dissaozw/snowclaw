"""MotionBERT 3D pose lifting backend using PyTorch."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from core.schemas import Pose3D

from .interfaces import Keypoints2D, PoseLifter3D
from .joint_mapping import coco_keypoints_to_pose3d

logger = logging.getLogger(__name__)

# MotionBERT config (lite model)
MOTIONBERT_WINDOW_SIZE = 243  # Temporal window (frames)
MOTIONBERT_NUM_JOINTS = 17

# DSTformer lite config (from MB_ft_h36m_global_lite.yaml)
LITE_CONFIG = dict(
    dim_in=3,
    dim_out=3,
    dim_feat=256,
    dim_rep=512,
    depth=5,
    num_heads=8,
    mlp_ratio=4,
    num_joints=17,
    maxlen=243,
    att_fuse=True,
)

HF_REPO_ID = "walterzhu/MotionBERT"
HF_CHECKPOINT = "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"


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
    if n <= polyorder:
        return joints_3d  # Too few frames to smooth

    # Savitzky-Golay requires odd window_length <= n and > polyorder.
    window_length = min(window_length, n)
    if window_length % 2 == 0:
        window_length -= 1

    min_valid_window = polyorder + 1
    if min_valid_window % 2 == 0:
        min_valid_window += 1

    if window_length < min_valid_window:
        if n < min_valid_window:
            return joints_3d
        window_length = min_valid_window

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


def _get_device() -> str:
    """Select best available PyTorch device."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class MotionBERTBackend(PoseLifter3D):
    """
    MotionBERT 3D pose lifting using PyTorch.

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
            model_path: Path to MotionBERT checkpoint (.bin). Downloads if None.
            device: "auto" (detect MPS/CUDA), "cuda", "mps", or "cpu".
            smoothing_window: Savitzky-Golay window length for temporal smoothing.
            smoothing_poly: SG polynomial order.
        """
        self._model_path = model_path
        self._model = None
        self._device_pref = device
        self._device = None
        self._smoothing_window = smoothing_window
        self._smoothing_poly = smoothing_poly

    def _load_model(self):
        """Lazy-load the DSTformer model with checkpoint weights."""
        if self._model is not None:
            return self._model

        import torch

        from .motionbert import DSTformer

        # Resolve device
        if self._device_pref == "auto":
            self._device = _get_device()
        else:
            self._device = self._device_pref
        logger.info("MotionBERT using device: %s", self._device)

        # Download checkpoint if needed
        if self._model_path is None:
            from huggingface_hub import hf_hub_download
            self._model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_CHECKPOINT,
            )

        # Build model
        model = DSTformer(**LITE_CONFIG)

        # Load checkpoint (weights_only=False needed for legacy PyTorch saves)
        checkpoint = torch.load(
            str(self._model_path),
            map_location="cpu",
            weights_only=False,
        )
        # The checkpoint may have 'model', 'model_pos', or be a raw state_dict
        if "model_pos" in checkpoint:
            state_dict = checkpoint["model_pos"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Strip 'module.' prefix from DataParallel-wrapped checkpoints
        state_dict = {
            k.removeprefix("module."): v for k, v in state_dict.items()
        }

        model.load_state_dict(state_dict, strict=True)
        model.to(self._device)
        model.eval()
        self._model = model
        return self._model

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

        import torch

        model = self._load_model()

        # Normalize all 2D keypoints
        normalized = [_normalize_keypoints_2d(kp) for kp in keypoints_2d]
        n_frames = len(normalized)

        # Run inference frame-by-frame with temporal windows
        all_joints_3d = np.zeros((n_frames, MOTIONBERT_NUM_JOINTS, 3), dtype=np.float32)

        with torch.no_grad():
            for i in range(n_frames):
                window = _assemble_temporal_window(
                    normalized, MOTIONBERT_WINDOW_SIZE, i
                )
                # MotionBERT expects (B, T, J, C) where C includes confidence
                # Append a dummy confidence channel of 1.0
                conf = np.ones((*window.shape[:-1], 1), dtype=np.float32)
                window_3ch = np.concatenate([window, conf], axis=-1)  # (243, 17, 3)

                # Shape: (1, 243, 17, 3)
                input_tensor = torch.from_numpy(window_3ch[np.newaxis]).float().to(self._device)
                output = model(input_tensor)  # (1, 243, 17, 3)

                # Take the center frame prediction
                center = MOTIONBERT_WINDOW_SIZE // 2
                all_joints_3d[i] = output[0, center].cpu().numpy()

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
            # Anchor 3D projection to actual 2D hip pixel position
            kp2d = keypoints_2d[i].points  # (K, 2) as (x, y)
            left_hip_px = kp2d[11]   # COCO index 11
            right_hip_px = kp2d[12]  # COCO index 12
            hip_px = (left_hip_px + right_hip_px) / 2.0
            pose.anchor_px = [float(hip_px[0]), float(hip_px[1])]
            poses.append(pose)

        return poses
