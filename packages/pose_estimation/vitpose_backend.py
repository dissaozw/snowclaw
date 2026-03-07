"""ViTPose 2D keypoint detection using easy_ViTPose library.

Uses VitInference which handles the full pipeline:
YOLOv8 person detection → crop → ViTPose inference → heatmap decoding.
"""

from __future__ import annotations

import logging

import numpy as np
from huggingface_hub import hf_hub_download

from .interfaces import Keypoints2D, PoseEstimator2D

logger = logging.getLogger(__name__)

VITPOSE_NUM_KEYPOINTS = 17  # COCO format


class ViTPoseBackend(PoseEstimator2D):
    """ViTPose 2D keypoint detection with YOLOv8 person detection.

    Detects 17 COCO-format keypoints per frame using easy_ViTPose's
    VitInference, which handles preprocessing and postprocessing correctly.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        batch_size: int = 16,
    ):
        self.batch_size = batch_size
        self._device = None if device == "auto" else device
        self._model_path = model_path
        self._vitpose = None

    def _get_vitpose(self):
        """Lazy-load VitInference."""
        if self._vitpose is not None:
            return self._vitpose

        from easy_ViTPose import VitInference

        vit_path = self._model_path or hf_hub_download(
            repo_id="JunkyByte/easy_ViTPose",
            filename="onnx/coco/vitpose-b-coco.onnx",
        )
        yolo_path = hf_hub_download(
            repo_id="JunkyByte/easy_ViTPose",
            filename="yolov8/yolov8n.pt",
        )
        logger.info("Loading VitInference (model=%s, yolo=%s)", vit_path, yolo_path)

        self._vitpose = VitInference(
            model=vit_path,
            yolo=yolo_path,
            is_video=True,
            single_pose=True,
            device=self._device,
        )
        return self._vitpose

    def predict(self, frames: list[np.ndarray]) -> list[Keypoints2D]:
        """Detect 2D keypoints in video frames.

        Args:
            frames: List of RGB images (H, W, 3), uint8.

        Returns:
            List of Keypoints2D, one per frame.
        """
        vitpose = self._get_vitpose()
        vitpose.reset()
        results: list[Keypoints2D] = []

        for frame in frames:
            image_size = (frame.shape[0], frame.shape[1])

            kp_dict = vitpose.inference(frame)

            if kp_dict:
                # easy_ViTPose returns (K, 3) in (y, x, conf) order — swap to (x, y)
                kp = next(iter(kp_dict.values()))
                points = kp[:, :2][:, ::-1].copy()  # (K, 2) as (x, y)
                confidence = kp[:, 2].copy()  # (K,)
            else:
                points = np.zeros((VITPOSE_NUM_KEYPOINTS, 2), dtype=np.float32)
                confidence = np.zeros(VITPOSE_NUM_KEYPOINTS, dtype=np.float32)

            results.append(Keypoints2D(
                points=points,
                confidence=confidence,
                image_size=image_size,
            ))

        return results
