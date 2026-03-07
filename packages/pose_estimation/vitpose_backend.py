"""ViTPose 2D keypoint detection with YOLOv8 person detection pipeline.

Pipeline per frame:
1. YOLOv8n detects person bounding boxes in the full frame
2. Crop the highest-confidence person box (with padding), letterbox to 256x192
3. Run ViTPose ONNX on the crop to get heatmaps
4. Decode heatmaps and unproject keypoints back to full-frame coordinates
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .interfaces import Keypoints2D, PoseEstimator2D

logger = logging.getLogger(__name__)

VITPOSE_INPUT_SIZE = (256, 192)  # H, W expected by ViTPose
VITPOSE_NUM_KEYPOINTS = 17  # COCO format
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
            logger.info("Using CUDA for ViTPose inference")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("CUDA not available, using CPU for ViTPose inference")
        return ["CPUExecutionProvider"]

    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            logger.warning("CUDA requested but not available, falling back to CPU")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def _detect_persons(yolo_model, frame: np.ndarray, conf_threshold: float = 0.3) -> list[dict]:
    """Run YOLOv8 person detection on a frame.

    Returns list of dicts with 'box' [x1,y1,x2,y2] and 'conf' float,
    sorted by confidence descending.
    """
    results = yolo_model(frame, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf >= conf_threshold:  # class 0 = person
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                detections.append({"box": xyxy, "conf": conf})
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections


def _crop_and_preprocess(
    frame: np.ndarray,
    bbox: list[float],
    input_size: tuple[int, int] = VITPOSE_INPUT_SIZE,
    pad_ratio: float = 0.2,
) -> tuple[np.ndarray, dict]:
    """Crop person region, letterbox to input_size, normalize.

    Args:
        frame: Full RGB frame (H, W, 3) uint8.
        bbox: [x1, y1, x2, y2] person bounding box.
        input_size: (H, W) target size for ViTPose.
        pad_ratio: Fraction of box size to pad on each side.

    Returns:
        input_tensor: (3, H, W) float32 normalized tensor.
        transform: Dict with keys needed to unproject keypoints back.
    """
    import cv2

    img_h, img_w = frame.shape[:2]
    target_h, target_w = input_size

    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    # Add padding
    x1_pad = max(0, x1 - bw * pad_ratio)
    y1_pad = max(0, y1 - bh * pad_ratio)
    x2_pad = min(img_w, x2 + bw * pad_ratio)
    y2_pad = min(img_h, y2 + bh * pad_ratio)

    # Crop
    crop = frame[int(y1_pad):int(y2_pad), int(x1_pad):int(x2_pad)]
    crop_h, crop_w = crop.shape[:2]

    if crop_h == 0 or crop_w == 0:
        # Degenerate box — return zeros
        tensor = np.zeros((3, target_h, target_w), dtype=np.float32)
        transform = {
            "x1_pad": 0, "y1_pad": 0,
            "scale": 1.0, "pad_left": 0, "pad_top": 0,
        }
        return tensor, transform

    # Letterbox: fit crop into target_w x target_h maintaining aspect ratio
    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size with gray (128)
    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    # Normalize
    img = canvas.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = img.transpose(2, 0, 1)  # HWC -> CHW

    transform = {
        "x1_pad": float(x1_pad),
        "y1_pad": float(y1_pad),
        "scale": float(scale),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
    }
    return tensor, transform


def _decode_and_unproject(
    heatmaps: np.ndarray,
    transform: dict,
    input_size: tuple[int, int] = VITPOSE_INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode heatmaps to full-frame pixel coordinates.

    Args:
        heatmaps: (num_keypoints, hm_H, hm_W) float heatmaps.
        transform: Dict from _crop_and_preprocess with unproject params.
        input_size: (H, W) of the ViTPose input.

    Returns:
        points: (num_keypoints, 2) pixel coordinates in original frame.
        confidence: (num_keypoints,) confidence scores.
    """
    num_kp, hm_h, hm_w = heatmaps.shape
    target_h, target_w = input_size

    points = np.zeros((num_kp, 2), dtype=np.float32)
    confidence = np.zeros(num_kp, dtype=np.float32)

    pad_left = transform["pad_left"]
    pad_top = transform["pad_top"]
    scale = transform["scale"]
    x1_pad = transform["x1_pad"]
    y1_pad = transform["y1_pad"]

    for i in range(num_kp):
        hm = heatmaps[i]
        flat_idx = int(np.argmax(hm))
        y_hm, x_hm = divmod(flat_idx, hm_w)

        # Heatmap coords -> crop input coords (256x192 space)
        x_crop = (x_hm + 0.5) / hm_w * target_w
        y_crop = (y_hm + 0.5) / hm_h * target_h

        # Remove letterbox padding, undo resize, add crop offset
        x_frame = (x_crop - pad_left) / scale + x1_pad
        y_frame = (y_crop - pad_top) / scale + y1_pad

        points[i, 0] = x_frame
        points[i, 1] = y_frame
        confidence[i] = float(hm[y_hm, x_hm])

    return points, confidence


class ViTPoseBackend(PoseEstimator2D):
    """ViTPose 2D keypoint detection with YOLOv8 person detection.

    Detects 17 COCO-format keypoints per frame using a proper
    detect -> crop -> pose pipeline.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "auto",
        batch_size: int = 16,
    ):
        self.batch_size = batch_size
        self._providers = _get_onnx_providers(device)
        self._session = None
        self._yolo = None
        self._model_path = model_path

    def _get_yolo(self):
        """Lazy-load YOLOv8n person detector."""
        if self._yolo is not None:
            return self._yolo

        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        yolo_path = hf_hub_download(
            repo_id="JunkyByte/easy_ViTPose",
            filename="yolov8/yolov8n.pt",
        )
        logger.info("Loaded YOLOv8n person detector from %s", yolo_path)
        self._yolo = YOLO(yolo_path)
        return self._yolo

    def _get_session(self):
        """Lazy-load ViTPose ONNX session."""
        if self._session is not None:
            return self._session

        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        if self._model_path is None:
            self._model_path = hf_hub_download(
                repo_id="JunkyByte/easy_ViTPose",
                filename="onnx/coco/vitpose-b-coco.onnx",
            )
            logger.info("Downloaded ViTPose-B COCO model to %s", self._model_path)

        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=self._providers,
        )
        return self._session

    def predict(self, frames: list[np.ndarray]) -> list[Keypoints2D]:
        """Detect 2D keypoints in video frames.

        Args:
            frames: List of RGB images (H, W, 3), uint8.

        Returns:
            List of Keypoints2D, one per frame.
        """
        yolo = self._get_yolo()
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        results: list[Keypoints2D] = []

        for frame in frames:
            image_size = (frame.shape[0], frame.shape[1])

            # Step 1: Detect persons
            detections = _detect_persons(yolo, frame)

            if detections:
                bbox = detections[0]["box"]  # highest confidence
            else:
                # Fallback: use full frame as bounding box
                logger.debug("No person detected, using full frame")
                bbox = [0.0, 0.0, float(frame.shape[1]), float(frame.shape[0])]

            # Step 2: Crop and preprocess
            tensor, transform = _crop_and_preprocess(frame, bbox)
            batch_input = tensor[np.newaxis].astype(np.float32)  # (1, 3, H, W)

            # Step 3: Run ViTPose
            outputs = session.run(None, {input_name: batch_input})
            heatmaps = outputs[0][0]  # (17, hm_H, hm_W)

            # Step 4: Decode and unproject to full-frame coords
            points, confidence = _decode_and_unproject(heatmaps, transform)

            results.append(Keypoints2D(
                points=points,
                confidence=confidence,
                image_size=image_size,
            ))

        return results
