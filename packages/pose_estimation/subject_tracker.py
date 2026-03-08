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

Resolution note:
  `max_drift_px` is in **pixel** units and is therefore resolution-dependent.
  The default (300 px) is calibrated for ~1080p footage at 30 fps.  For 4K footage
  multiply by ~2; for 720p divide by ~1.5.  A future improvement would normalise by
  image diagonal automatically.
"""

from __future__ import annotations

import numpy as np


_DEFAULT_MIN_CONFIDENT_JOINTS = 5   # require at least 5 joints above threshold
_DEFAULT_MIN_CONFIDENCE = 0.3       # per-joint confidence threshold
_DEFAULT_MAX_DRIFT_PX = 300.0       # max centroid movement between consecutive frames (1080p)


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
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        if min_confident_joints < 1:
            raise ValueError(f"min_confident_joints must be >= 1, got {min_confident_joints}")
        if max_drift_px < 0:
            raise ValueError(f"max_drift_px must be >= 0, got {max_drift_px}")
        # Note: max_drift_px=0.0 is valid but effectively disables temporal tracking —
        # floating-point centroid jitter from mean() will almost always exceed 0, so
        # _nearest always falls back to largest-bbox. Use a small positive value instead.
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
                        An empty dict is valid input and returns None.

        Returns:
            Selected keypoints array (17, 3) or None if no valid detection.
        """
        candidates = self._filter(detections)
        if not candidates:
            # No valid detection — preserve previous centroid so next frame can recover
            return None

        if self._prev_centroid is None:
            chosen = self._largest(candidates)
        else:
            chosen = self._nearest(candidates)

        # Update state from high-confidence joints only
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
            if int(np.sum(conf >= self.min_confidence)) >= self.min_confident_joints:
                valid.append(kp)
        return valid

    def _largest(self, candidates: list[np.ndarray]) -> np.ndarray:
        """Pick the candidate with the largest bounding-box over *confident* joints.

        Using only confident joints prevents a single garbage/misplaced joint from
        inflating the bounding box and causing a false-positive to "win".
        """
        return max(candidates, key=lambda kp: _bbox_area(kp, self.min_confidence))

    def _nearest(self, candidates: list[np.ndarray]) -> np.ndarray:
        """
        Pick the candidate whose centroid is closest to the previous centroid.
        Falls back to largest-bbox if all candidates exceed max_drift_px.
        """
        if self._prev_centroid is None:
            raise RuntimeError("_nearest called before _prev_centroid is set")

        best_kp = candidates[0]
        best_dist = float(np.linalg.norm(
            _centroid(candidates[0], self.min_confidence) - self._prev_centroid
        ))

        for kp in candidates[1:]:
            c = _centroid(kp, self.min_confidence)
            dist = float(np.linalg.norm(c - self._prev_centroid))
            if dist < best_dist:
                best_dist = dist
                best_kp = kp

        if best_dist <= self.max_drift_px:
            return best_kp

        # Drift too large — subject likely left frame, pick largest new detection
        return self._largest(candidates)


# ------------------------------------------------------------------
# Free helpers
# ------------------------------------------------------------------

def _centroid(kp: np.ndarray, min_conf: float = 0.3) -> np.ndarray:
    """
    Compute (x, y) centroid of high-confidence, finite keypoints.

    kp is (17, 3) in easy_ViTPose order: (y, x, conf).
    Returns centroid as (x, y) in pixel coordinates.

    NaN/Inf coordinates are excluded to prevent poisoning the tracker state.
    If no valid joints remain after filtering, falls back to all joints (NaNs
    still excluded).
    """
    conf = kp[:, 2]
    coords = kp[:, :2]
    finite_mask = np.isfinite(coords).all(axis=1)
    conf_mask = (conf >= min_conf) & finite_mask
    if not np.any(conf_mask):
        # No confident+finite joints — fall back to any finite joint so we
        # can still compute a centroid for distance comparison. This matches
        # _bbox_area's behaviour of using whatever finite coords are available
        # rather than returning a sentinel value that would poison comparisons.
        conf_mask = finite_mask
    if not np.any(conf_mask):
        # All joints are NaN/Inf — return zeros as a safe sentinel.
        # _bbox_area also returns 0.0 in this case (consistent behaviour).
        return np.zeros(2, dtype=float)
    # easy_ViTPose stores (y, x) — swap to (x, y) for consistent pixel coords.
    # This is the SubjectTracker-internal swap; the public output swap is in
    # vitpose_backend._yx_to_xy().
    xy = coords[conf_mask][:, ::-1]
    return xy.mean(axis=0)


def _bbox_area(kp: np.ndarray, min_conf: float = 0.0) -> float:
    """
    Bounding-box area over confident, finite keypoints.

    Filtering to confident and finite joints prevents:
    - A single misdetected joint at the image border inflating the bbox
    - NaN coordinates producing NaN area and poisoning selection logic

    kp is (17, 3) in (y, x, conf) order.
    Returns 0.0 if no valid joints meet the threshold.
    """
    conf = kp[:, 2]
    coords = kp[:, :2]
    finite_mask = np.isfinite(coords).all(axis=1)
    mask = (conf >= min_conf) & finite_mask
    if not np.any(mask):
        return 0.0
    ys = kp[mask, 0]
    xs = kp[mask, 1]
    return float((ys.max() - ys.min()) * (xs.max() - xs.min()))
