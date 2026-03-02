"""ViTPose+ 2D keypoint detection backend using ONNX Runtime."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .interfaces import Keypoints2D, PoseEstimator2D

logger = logging.getLogger(__name__)

# ViTPose+ ONNX model config
VITPOSE_INPUT_SIZE = (256, 192)  # H, W expected by the model
VITPOSE_NUM_KEYPOINTS = 17  # COCO format


def _get_onnx_providers(device: str = "auto") -> list[str]:
    """Return ONNX Runtime execution providers based on device preference."""
    if device == "cpu":
        return ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        return ["CPUExecutionProvider"]

    if device == "auto":
        if "CUDAExecutionProvider" in available:
            logger.info("Using CUDA for ViTPose+ inference")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("CUDA not available, using CPU for ViTPose+ inference")
        return ["CPUExecutionProvider"]

    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            logger.warning("CUDA requested but not available, falling back to CPU")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def _preprocess_frame(frame: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    """
    Resize and normalize a single frame for ViTPose+ input.

    Args:
        frame: RGB image (H, W, 3) uint8.
        input_size: Target (H, W) for the model.

    Returns:
        Preprocessed array of shape (3, H, W), float32, normalized.
    """
    import cv2

    target_h, target_w = input_size
    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1] then apply ImageNet normalization
    img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # HWC -> CHW
    return img.transpose(2, 0, 1)


def _decode_heatmaps(
    heatmaps: np.ndarray, image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode heatmaps to pixel coordinates and confidence scores.

    Args:
        heatmaps: Shape (num_keypoints, heatmap_H, heatmap_W).
        image_size: Original image (H, W) to scale coordinates back.

    Returns:
        points: (num_keypoints, 2) pixel coordinates.
        confidence: (num_keypoints,) confidence scores.
    """
    num_kp, hm_h, hm_w = heatmaps.shape
    orig_h, orig_w = image_size

    points = np.zeros((num_kp, 2), dtype=np.float32)
    confidence = np.zeros(num_kp, dtype=np.float32)

    for i in range(num_kp):
        hm = heatmaps[i]
        flat_idx = np.argmax(hm)
        y, x = divmod(flat_idx, hm_w)

        # Scale to original image coordinates
        points[i, 0] = x / hm_w * orig_w
        points[i, 1] = y / hm_h * orig_h
        confidence[i] = float(hm[y, x])

    return points, confidence


class ViTPoseBackend(PoseEstimator2D):
    """
    ViTPose+ 2D keypoint detection using ONNX Runtime.

    Detects 17 COCO-format keypoints per frame.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "auto",
        batch_size: int = 16,
    ):
        """
        Args:
            model_path: Path to the ViTPose+ ONNX model file.
                        If None, will attempt to download to cache.
            device: "auto" (detect GPU), "cuda", or "cpu".
            batch_size: Number of frames to process in each batch.
        """
        self.batch_size = batch_size
        self._providers = _get_onnx_providers(device)
        self._session = None
        self._model_path = model_path

    def _get_session(self):
        """Lazy-load ONNX session."""
        if self._session is not None:
            return self._session

        import onnxruntime as ort

        if self._model_path is None:
            from .model_cache import download_model
            self._model_path = download_model(
                url="https://huggingface.co/snowclaw/vitpose-plus/resolve/main/vitpose_plus_base.onnx",
                filename="vitpose_plus_base.onnx",
            )

        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=self._providers,
        )
        return self._session

    def predict(self, frames: list[np.ndarray]) -> list[Keypoints2D]:
        """
        Detect 2D keypoints in video frames.

        Args:
            frames: List of RGB images (H, W, 3), uint8.

        Returns:
            List of Keypoints2D, one per frame.
        """
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        results: list[Keypoints2D] = []

        for batch_start in range(0, len(frames), self.batch_size):
            batch_frames = frames[batch_start:batch_start + self.batch_size]

            # Preprocess batch
            batch_input = np.stack(
                [_preprocess_frame(f, VITPOSE_INPUT_SIZE) for f in batch_frames],
                axis=0,
            ).astype(np.float32)

            # Run inference
            outputs = session.run(None, {input_name: batch_input})
            batch_heatmaps = outputs[0]  # shape (batch, 17, H, W)

            # Decode each frame
            for i, frame in enumerate(batch_frames):
                hm = batch_heatmaps[i]
                image_size = (frame.shape[0], frame.shape[1])
                points, confidence = _decode_heatmaps(hm, image_size)
                results.append(Keypoints2D(
                    points=points,
                    confidence=confidence,
                    image_size=image_size,
                ))

        return results
