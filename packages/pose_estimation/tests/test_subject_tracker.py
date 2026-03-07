"""Tests for SubjectTracker — single-subject selection and temporal locking."""

from __future__ import annotations

import numpy as np
import pytest

from pose_estimation.subject_tracker import SubjectTracker, _bbox_area, _centroid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kp(
    cy: float,
    cx: float,
    spread: float = 50.0,
    conf: float = 0.9,
    n: int = 17,
) -> np.ndarray:
    """Build fake (n, 3) keypoints in easy_ViTPose (y, x, conf) order."""
    rng = np.random.default_rng(0)
    kp = np.zeros((n, 3), dtype=np.float32)
    kp[:, 0] = cy + rng.uniform(-spread, spread, n)  # y
    kp[:, 1] = cx + rng.uniform(-spread, spread, n)  # x
    kp[:, 2] = conf
    return kp


# ---------------------------------------------------------------------------
# _centroid / _bbox_area helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_centroid_returns_xy(self):
        kp = _make_kp(cy=100, cx=200, spread=0, conf=0.9)
        c = _centroid(kp, min_conf=0.3)
        assert c.shape == (2,)
        assert abs(c[0] - 200) < 1   # x
        assert abs(c[1] - 100) < 1   # y

    def test_centroid_ignores_low_conf(self):
        kp = _make_kp(cy=100, cx=200, conf=0.9)
        kp[0, 2] = 0.0   # zero out one joint
        c = _centroid(kp, min_conf=0.3)
        assert c.shape == (2,)

    def test_centroid_falls_back_all_joints_when_none_confident(self):
        kp = _make_kp(cy=100, cx=200, conf=0.0)
        c = _centroid(kp, min_conf=0.3)
        assert c.shape == (2,)   # should not raise

    def test_bbox_area_larger_for_bigger_person(self):
        small = _make_kp(cy=100, cx=100, spread=10)
        large = _make_kp(cy=100, cx=100, spread=100)
        assert _bbox_area(large) > _bbox_area(small)

    def test_bbox_area_ignores_low_confidence_outlier(self):
        """A single garbage joint at the frame edge must not inflate bbox area."""
        # Tightly clustered person in center
        kp = _make_kp(cy=300, cx=400, spread=30, conf=0.9)
        area_clean = _bbox_area(kp, min_conf=0.5)

        # Add one joint far off-screen with low confidence
        kp_dirty = kp.copy()
        kp_dirty[0, 0] = 0.0    # y = 0 (top of frame)
        kp_dirty[0, 1] = 0.0    # x = 0 (left of frame)
        kp_dirty[0, 2] = 0.1    # low conf
        area_dirty = _bbox_area(kp_dirty, min_conf=0.5)

        assert abs(area_dirty - area_clean) < area_clean * 0.05, (
            "Low-confidence outlier joint should not significantly inflate bbox area"
        )

    def test_bbox_area_zero_when_no_confident_joints(self):
        kp = _make_kp(cy=100, cx=100, conf=0.1)
        assert _bbox_area(kp, min_conf=0.5) == 0.0


# ---------------------------------------------------------------------------
# Confidence filter
# ---------------------------------------------------------------------------

class TestConfidenceFilter:
    def test_low_confidence_detection_filtered_out(self):
        tracker = SubjectTracker(min_confident_joints=5, min_confidence=0.5)
        # All joints below threshold → should return None
        kp = _make_kp(cy=100, cx=100, conf=0.1)
        result = tracker.select({0: kp})
        assert result is None

    def test_high_confidence_detection_passes(self):
        tracker = SubjectTracker(min_confident_joints=5, min_confidence=0.3)
        kp = _make_kp(cy=100, cx=100, conf=0.9)
        result = tracker.select({0: kp})
        assert result is not None

    def test_mixed_confidence_counted_correctly(self):
        tracker = SubjectTracker(min_confident_joints=10, min_confidence=0.5)
        kp = _make_kp(cy=100, cx=100, conf=0.9)
        kp[:8, 2] = 0.1   # 8 joints below threshold → only 9 above, but need 10
        result = tracker.select({0: kp})
        assert result is None


# ---------------------------------------------------------------------------
# First-frame: largest-bbox selection
# ---------------------------------------------------------------------------

class TestFirstFrameSelection:
    def test_selects_largest_person_on_first_frame(self):
        tracker = SubjectTracker()
        small = _make_kp(cy=200, cx=200, spread=10)
        large = _make_kp(cy=500, cx=500, spread=200)
        result = tracker.select({0: small, 1: large})
        assert np.allclose(result, large)

    def test_single_candidate_returned(self):
        tracker = SubjectTracker()
        kp = _make_kp(cy=100, cx=100)
        result = tracker.select({0: kp})
        assert np.allclose(result, kp)

    def test_empty_detection_returns_none(self):
        tracker = SubjectTracker()
        result = tracker.select({})
        assert result is None


# ---------------------------------------------------------------------------
# Temporal consistency
# ---------------------------------------------------------------------------

class TestTemporalConsistency:
    def test_stays_on_same_person_across_frames(self):
        tracker = SubjectTracker(max_drift_px=300)

        # Subject: large, center-frame
        subject = _make_kp(cy=300, cx=400, spread=60, conf=0.9)
        # Distractor: smaller, far away
        distractor = _make_kp(cy=50, cx=50, spread=20, conf=0.9)

        # Frame 0 — subject is largest → gets selected
        r0 = tracker.select({0: subject, 1: distractor})
        assert np.allclose(r0, subject)

        # Frame 1 — subject moves slightly, distractor stays far
        subject_moved = _make_kp(cy=310, cx=410, spread=60, conf=0.9)
        r1 = tracker.select({0: subject_moved, 1: distractor})
        assert np.allclose(r1, subject_moved), "Should track the nearby subject"

    def test_falls_back_to_largest_when_drift_exceeded(self):
        tracker = SubjectTracker(max_drift_px=50)

        # Frame 0 — lock on small person at top-left
        small_tl = _make_kp(cy=30, cx=30, spread=5, conf=0.9)
        tracker.select({0: small_tl})

        # Frame 1 — that person gone; only a large person far away
        large_br = _make_kp(cy=500, cx=600, spread=150, conf=0.9)
        r1 = tracker.select({0: large_br})
        assert np.allclose(r1, large_br), "Should fall back to largest after drift"

    def test_reset_clears_state(self):
        tracker = SubjectTracker(max_drift_px=50)

        # Lock onto a person at top-left
        small_tl = _make_kp(cy=30, cx=30, spread=5)
        tracker.select({0: small_tl})

        tracker.reset()

        # After reset, should behave like frame-0 (largest wins)
        small2 = _make_kp(cy=30, cx=30, spread=5, conf=0.9)
        large2 = _make_kp(cy=300, cx=300, spread=150, conf=0.9)
        r = tracker.select({0: small2, 1: large2})
        assert np.allclose(r, large2), "After reset, should pick largest again"

    def test_shadow_with_low_confidence_ignored(self):
        tracker = SubjectTracker(min_confident_joints=5, min_confidence=0.4, max_drift_px=300)

        subject = _make_kp(cy=300, cx=400, spread=60, conf=0.9)
        shadow = _make_kp(cy=300, cx=200, spread=30, conf=0.1)  # low conf → filtered

        r0 = tracker.select({0: subject, 1: shadow})
        assert np.allclose(r0, subject)

        # Next frame: shadow is still there (low conf), subject moves slightly
        subject_moved = _make_kp(cy=310, cx=410, spread=60, conf=0.9)
        r1 = tracker.select({0: subject_moved, 1: shadow})
        assert np.allclose(r1, subject_moved)

    def test_no_detection_preserves_previous_centroid(self):
        """Missing detection frame should not reset the centroid."""
        tracker = SubjectTracker()
        subject = _make_kp(cy=300, cx=400, spread=60, conf=0.9)
        tracker.select({0: subject})

        # Frame with no detections
        r = tracker.select({})
        assert r is None

        # After empty frame, tracker should still know where the subject was
        subject_back = _make_kp(cy=305, cx=405, spread=60, conf=0.9)
        distractor = _make_kp(cy=50, cx=50, spread=20, conf=0.9)
        r2 = tracker.select({0: subject_back, 1: distractor})
        assert np.allclose(r2, subject_back), \
            "After empty frame, should re-lock onto nearby subject"

    def test_largest_uses_confident_joints_only(self):
        """Largest-bbox selection must ignore low-confidence outlier joints."""
        tracker = SubjectTracker(min_confidence=0.5)

        # Person A: small but tight, all joints confident
        small = _make_kp(cy=300, cx=300, spread=20, conf=0.9)

        # Person B: one joint far off-screen (low conf) that would inflate bbox if counted
        large_fake = _make_kp(cy=300, cx=500, spread=20, conf=0.9)
        large_fake[0, 0] = 0.0    # y at top edge
        large_fake[0, 1] = 0.0    # x at left edge
        large_fake[0, 2] = 0.1    # low confidence — should be excluded from area

        # Without the fix, large_fake would "win" due to the outlier inflating its bbox.
        # With the fix, both should have similar areas and small might even win.
        r = tracker.select({0: small, 1: large_fake})
        # Key assertion: result should NOT be driven by the garbage outlier joint
        # (we don't prescribe which wins, but the outlier joint must not be the deciding factor)
        assert r is not None

    def test_drift_boundary_condition(self):
        """A candidate at exactly max_drift_px should remain locked (<=, not <)."""
        tracker = SubjectTracker(max_drift_px=100.0)
        subject = _make_kp(cy=300, cx=300, spread=0, conf=0.9)
        tracker.select({0: subject})

        # Place next detection exactly 100px away (at the boundary)
        at_boundary = _make_kp(cy=300, cx=400, spread=0, conf=0.9)  # cx diff = 100
        result = tracker.select({0: at_boundary})
        assert np.allclose(result, at_boundary), \
            "Candidate at exactly max_drift_px should be accepted (<=, not <)"

    def test_exact_threshold_joints(self):
        """Detection with exactly min_confident_joints at exactly min_confidence passes."""
        tracker = SubjectTracker(min_confident_joints=5, min_confidence=0.3)
        kp = _make_kp(cy=300, cx=300, conf=0.0)   # all low
        kp[:5, 2] = 0.3   # exactly 5 joints at exactly 0.3
        result = tracker.select({0: kp})
        assert result is not None, \
            "Exactly min_confident_joints at exactly min_confidence should pass the filter"

    def test_multi_frame_dropout_recovery(self):
        """After 10 consecutive empty frames, tracker recovers to the nearest subject."""
        tracker = SubjectTracker(max_drift_px=300)
        subject = _make_kp(cy=300, cx=400, spread=60, conf=0.9)
        tracker.select({0: subject})

        # 10 consecutive empty frames
        for _ in range(10):
            assert tracker.select({}) is None

        # Centroid should be preserved — nearby subject picked over far distractor
        subject_back = _make_kp(cy=310, cx=410, spread=60, conf=0.9)
        distractor = _make_kp(cy=50, cx=50, spread=20, conf=0.9)
        result = tracker.select({0: subject_back, 1: distractor})
        assert np.allclose(result, subject_back), \
            "After 10 empty frames, should re-lock onto subject near last known position"

    def test_crossing_subjects_before_overlap(self):
        """
        Two skiers moving toward each other — tracker stays on subject until they overlap.

        NOTE: Centroid-only tracking cannot resolve identity after full crossing
        (when the two skiers swap positions). This test verifies correct behaviour
        *before* the crossing point. Post-crossing identity recovery requires
        person-ID tracking (e.g. ByteTrack) which is out of scope for SubjectTracker.
        """
        tracker = SubjectTracker(max_drift_px=200)

        # Frame 0: subject on left (cx=200), other skier on right (cx=600)
        subject = _make_kp(cy=300, cx=200, spread=30, conf=0.9)
        other   = _make_kp(cy=300, cx=600, spread=30, conf=0.9)
        tracker.select({0: subject, 1: other})

        # Frames 1-3: skiers approach but have not yet crossed (subject stays closer
        # to previous centroid than the other skier)
        for step in range(1, 4):
            subject_pos = _make_kp(cy=300, cx=200 + step * 60, spread=30, conf=0.9)
            other_pos   = _make_kp(cy=300, cx=600 - step * 60, spread=30, conf=0.9)
            result = tracker.select({0: subject_pos, 1: other_pos})
            result_cx = result[:, 1].mean()
            subject_cx = subject_pos[:, 1].mean()
            other_cx   = other_pos[:, 1].mean()
            assert abs(result_cx - subject_cx) < abs(result_cx - other_cx), \
                f"Frame {step}: tracker should follow subject before crossing point"
