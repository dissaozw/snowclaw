"""ViTPose 2D keypoint detection using easy_ViTPose library.

Uses VitInference which handles the full pipeline:
YOLOv8 person detection → crop → ViTPose inference → heatmap decoding.

Subject locking:
    When multiple people are detected the SubjectTracker selects the best
    candidate using a two-stage strategy:
      1. First frame  → largest keypoint bounding-box (closest-to-camera heuristic).
      2. Later frames → centroid nearest to the previous frame's selection,
                        falling back to largest-bbox if drift exceeds the threshold.
    A minimum-confidence gate filters out shadows and blurry background figures.
"""

from __future__ import annotations

import logging

import numpy as np
from huggingface_hub import hf_hub_download

from .interfaces import Keypoints2D, PoseEstimator2D
from .subject_tracker import SubjectTracker

logger = logging.getLogger(__name__)

VITPOSE_NUM_KEYPOINTS = 17  # COCO format


class ViTPoseBackend(PoseEstimator2D):
    """ViTPose 2D keypoint detection with YOLOv8 person detection.

    Detects 17 COCO-format keypoints per frame using easy_ViTPose's
    VitInference, which handles preprocessing and postprocessing correctly.
    A SubjectTracker is used to lock onto a single person across frames.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        max_drift_px: float = 300.0,
        min_confident_joints: int = 5,
        min_confidence: float = 0.3,
    ):
        self._device = None if device == "auto" else device
        self._model_path = model_path
        self._vitpose = None
        self._tracker = SubjectTracker(
            min_confident_joints=min_confident_joints,
            min_confidence=min_confidence,
            max_drift_px=max_drift_px,
        )

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

        # single_pose=False so we receive all detected people and can apply
        # our own subject-lock logic via SubjectTracker.
        self._vitpose = VitInference(
            model=vit_path,
            yolo=yolo_path,
            is_video=True,
            single_pose=False,
            device=self._device,
        )
        return self._vitpose

    def predict(self, frames: list[np.ndarray]) -> list[Keypoints2D]:
        """Detect 2D keypoints in video frames.

        Args:
            frames: List of RGB images (H, W, 3), uint8.

        Returns:
            List of Keypoints2D, one per frame.  When no valid person is
            detected the keypoints and confidences are all-zero.
        """
        vitpose = self._get_vitpose()
        vitpose.reset()
        self._tracker.reset()
        results: list[Keypoints2D] = []

        for frame in frames:
            image_size = (frame.shape[0], frame.shape[1])

            # kp_dict: {person_id: ndarray(17, 3)} in (y, x, conf) order
            kp_dict = vitpose.inference(frame)

            selected = self._tracker.select(kp_dict) if kp_dict else None

            if selected is not None:
                # Swap (y, x) → (x, y) for downstream consumers
                points = selected[:, :2][:, ::-1].copy()
                confidence = selected[:, 2].copy()
            else:
                points = np.zeros((VITPOSE_NUM_KEYPOINTS, 2), dtype=np.float32)
                confidence = np.zeros(VITPOSE_NUM_KEYPOINTS, dtype=np.float32)

            results.append(Keypoints2D(
                points=points,
                confidence=confidence,
                image_size=image_size,
            ))

        return results
