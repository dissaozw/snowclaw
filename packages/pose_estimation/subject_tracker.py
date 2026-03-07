"""
Subject lock — selects and tracks a single person across frames.

Strategy:
  Frame 0   : pick the person with the largest keypoint bounding-box (closest to camera).
  Frame N>0 : pick the person whose centroid is closest to the previous frame's centroid,
              provided it is within `max_drift_px` pixels.  If nobody is within the drift
              threshold (subject left frame / re-entered), fall back to largest-bbox selection.

False-positive filter:
  Detections with fewer than `min_confident_joints` keypoints above `min_confidence`
  are discarded before selection.  This suppresses shadows and blurry background figures
  that YOLO sometimes picks up.
"""

from __future__ import annotations

import numpy as np


_DEFAULT_MIN_CONFIDENT_JOINTS = 5   # require at least 5 joints above threshold
_DEFAULT_MIN_CONFIDENCE = 0.3       # per-joint confidence threshold
_DEFAULT_MAX_DRIFT_PX = 300.0       # max centroid movement between consecutive frames


class SubjectTracker:
    """
    Stateful single-subject selector for multi-person pose detections.

    Usage::

        tracker = SubjectTracker()
        for frame_kps in all_frame_detections:          # dict[id -> (17,3) kp]
            selected = tracker.select(frame_kps)        # (17,3) or None
    """

    def __init__(
        self,
        min_confident_joints: int = _DEFAULT_MIN_CONFIDENT_JOINTS,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        max_drift_px: float = _DEFAULT_MAX_DRIFT_PX,
    ) -> None:
        self.min_confident_joints = min_confident_joints
        self.min_confidence = min_confidence
        self.max_drift_px = max_drift_px
        self._prev_centroid: np.ndarray | None = None  # (2,) in (x, y)

    def reset(self) -> None:
        """Reset tracker state (call between videos)."""
        self._prev_centroid = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        detections: dict[int, np.ndarray],
    ) -> np.ndarray | None:
        """
        Select the best matching person from a dict of detections.

        Args:
            detections: Mapping of person-id → keypoints array of shape (17, 3),
                        where columns are (y, x, confidence) — easy_ViTPose order.

        Returns:
            Selected keypoints array (17, 3) or None if no valid detection.
        """
        candidates = self._filter(detections)
        if not candidates:
            # No valid detection — keep previous centroid so next frame can recover
            return None

        if self._prev_centroid is None:
            chosen = self._largest(candidates)
        else:
            chosen = self._nearest(candidates)

        # Update centroid from high-confidence joints (x, y order)
        self._prev_centroid = _centroid(chosen, self.min_confidence)
        return chosen

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter(
        self, detections: dict[int, np.ndarray]
    ) -> list[np.ndarray]:
        """Return detections that pass the minimum confidence gate."""
        valid = []
        for kp in detections.values():
            conf = kp[:, 2]
            if np.sum(conf >= self.min_confidence) >= self.min_confident_joints:
                valid.append(kp)
        return valid

    def _largest(self, candidates: list[np.ndarray]) -> np.ndarray:
        """Pick the candidate with the largest keypoint bounding-box area."""
        return max(candidates, key=_bbox_area)

    def _nearest(self, candidates: list[np.ndarray]) -> np.ndarray:
        """
        Pick the candidate whose centroid is closest to the previous centroid.
        Falls back to largest-bbox if all candidates exceed max_drift_px.
        """
        assert self._prev_centroid is not None

        best: np.ndarray | None = None
        best_dist = float("inf")

        for kp in candidates:
            c = _centroid(kp, self.min_confidence)
            dist = float(np.linalg.norm(c - self._prev_centroid))
            if dist < best_dist:
                best_dist = dist
                best = kp

        if best_dist <= self.max_drift_px:
            return best  # type: ignore[return-value]

        # Drift too large — subject likely left frame, pick largest new detection
        return self._largest(candidates)


# ------------------------------------------------------------------
# Free helpers
# ------------------------------------------------------------------

def _centroid(kp: np.ndarray, min_conf: float = 0.3) -> np.ndarray:
    """
    Compute (x, y) centroid of high-confidence keypoints.

    kp is (17, 3) in easy_ViTPose order: (y, x, conf).
    Returns centroid in (x, y) pixel order.
    """
    conf = kp[:, 2]
    mask = conf >= min_conf
    if not np.any(mask):
        # Fall back to all joints
        mask = np.ones(len(kp), dtype=bool)
    # easy_ViTPose (y, x) → swap to (x, y)
    xy = kp[mask][:, :2][:, ::-1]
    return xy.mean(axis=0)


def _bbox_area(kp: np.ndarray) -> float:
    """
    Bounding-box area of keypoints (all joints, ignoring confidence).

    kp is (17, 3) in (y, x, conf) order.
    """
    ys = kp[:, 0]
    xs = kp[:, 1]
    return float((ys.max() - ys.min()) * (xs.max() - xs.min()))
