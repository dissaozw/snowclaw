"""Tests for core.schemas — shared data structures."""

import numpy as np
import pytest
from pydantic import ValidationError

from core.schemas import (
    Discipline,
    Frame,
    Pose3D,
    SessionMetrics,
    TurnPhase,
    TurnPhaseLabel,
)

# ── Helpers ──────────────────────────────────────────────────

ZERO = [0.0, 0.0, 0.0]


def _make_pose(**overrides) -> Pose3D:
    """Create a Pose3D with all joints at origin unless overridden."""
    defaults = {
        "head": [0, 1.8, 0],
        "neck": [0, 1.6, 0],
        "left_shoulder": [-0.2, 1.5, 0],
        "right_shoulder": [0.2, 1.5, 0],
        "left_elbow": [-0.3, 1.2, 0],
        "right_elbow": [0.3, 1.2, 0],
        "left_wrist": [-0.3, 1.0, 0],
        "right_wrist": [0.3, 1.0, 0],
        "left_hip": [-0.1, 1.0, 0],
        "right_hip": [0.1, 1.0, 0],
        "left_knee": [-0.1, 0.5, 0],
        "right_knee": [0.1, 0.5, 0],
        "left_ankle": [-0.1, 0.0, 0],
        "right_ankle": [0.1, 0.0, 0],
    }
    defaults.update(overrides)
    return Pose3D(**defaults)


# ── Discipline enum ──────────────────────────────────────────

class TestDiscipline:
    def test_ski_value(self):
        assert Discipline.SKI == "ski"

    def test_snowboard_value(self):
        assert Discipline.SNOWBOARD == "snowboard"


# ── TurnPhaseLabel enum ─────────────────────────────────────

class TestTurnPhaseLabel:
    def test_all_values(self):
        assert set(TurnPhaseLabel) == {
            TurnPhaseLabel.INITIATION,
            TurnPhaseLabel.FALL_LINE,
            TurnPhaseLabel.COMPLETION,
            TurnPhaseLabel.TRANSITION,
        }

    def test_initiation_value(self):
        assert TurnPhaseLabel.INITIATION == "initiation"


# ── Pose3D ───────────────────────────────────────────────────

class TestPose3D:
    def test_construction(self):
        pose = _make_pose()
        assert pose.head == [0, 1.8, 0]

    def test_requires_3_elements(self):
        with pytest.raises(ValidationError):
            _make_pose(head=[0, 1])

    def test_to_np(self):
        pose = _make_pose()
        arr = pose.to_np("head")
        np.testing.assert_array_almost_equal(arr, [0, 1.8, 0])

    def test_to_np_missing_optional(self):
        pose = _make_pose()
        with pytest.raises(ValueError, match="not set"):
            pose.to_np("left_ski_tip")

    def test_com(self):
        pose = _make_pose()
        com = pose.com
        assert com.shape == (3,)
        # COM y should be between ankle (0) and head (1.8)
        assert 0 < com[1] < 1.8

    def test_hip_midpoint(self):
        pose = _make_pose()
        np.testing.assert_array_almost_equal(pose.hip_midpoint, [0, 1.0, 0])

    def test_shoulder_midpoint(self):
        pose = _make_pose()
        np.testing.assert_array_almost_equal(pose.shoulder_midpoint, [0, 1.5, 0])

    def test_ankle_midpoint(self):
        pose = _make_pose()
        np.testing.assert_array_almost_equal(pose.ankle_midpoint, [0, 0.0, 0])

    def test_body_height(self):
        pose = _make_pose()
        assert pytest.approx(pose.body_height, abs=0.01) == 1.8

    def test_confidence_optional(self):
        pose = _make_pose(confidence={"head": 0.95, "neck": 0.9})
        assert pose.confidence["head"] == 0.95

    def test_equipment_keypoints_optional(self):
        pose = _make_pose(left_ski_tip=[0, 0, 1], right_ski_tip=[0, 0, 1])
        assert pose.left_ski_tip == [0, 0, 1]
        assert pose.board_nose is None

    def test_validates_floats(self):
        """Integer inputs are coerced to float."""
        pose = _make_pose(head=[0, 2, 0])
        assert pose.head == [0.0, 2.0, 0.0]


# ── Frame ────────────────────────────────────────────────────

class TestFrame:
    def test_construction(self):
        frame = Frame(frame_idx=0, timestamp_s=0.0, pose=_make_pose())
        assert frame.frame_idx == 0

    def test_negative_frame_idx_rejected(self):
        with pytest.raises(ValidationError):
            Frame(frame_idx=-1, timestamp_s=0.0, pose=_make_pose())

    def test_snow_surface_normal_normalized(self):
        frame = Frame(
            frame_idx=0,
            timestamp_s=0.0,
            pose=_make_pose(),
            snow_surface_normal=[0, 10, 0],
        )
        np.testing.assert_array_almost_equal(frame.snow_surface_normal, [0, 1, 0])

    def test_zero_normal_rejected(self):
        with pytest.raises(ValidationError):
            Frame(
                frame_idx=0,
                timestamp_s=0.0,
                pose=_make_pose(),
                snow_surface_normal=[0, 0, 0],
            )

    def test_default_discipline(self):
        frame = Frame(frame_idx=0, timestamp_s=0.0, pose=_make_pose())
        assert frame.discipline == Discipline.SKI


# ── TurnPhase ────────────────────────────────────────────────

class TestTurnPhase:
    def test_construction(self):
        tp = TurnPhase(
            label=TurnPhaseLabel.INITIATION,
            start_frame=0,
            end_frame=10,
            direction="left",
        )
        assert tp.label == TurnPhaseLabel.INITIATION

    def test_end_before_start_rejected(self):
        with pytest.raises(ValidationError):
            TurnPhase(
                label=TurnPhaseLabel.FALL_LINE,
                start_frame=10,
                end_frame=5,
            )

    def test_same_start_end_ok(self):
        tp = TurnPhase(
            label=TurnPhaseLabel.TRANSITION,
            start_frame=5,
            end_frame=5,
        )
        assert tp.start_frame == tp.end_frame


# ── SessionMetrics ───────────────────────────────────────────

class TestSessionMetrics:
    def test_construction(self):
        sm = SessionMetrics(
            discipline=Discipline.SKI,
            frame_count=100,
            duration_s=3.33,
        )
        assert sm.turns_detected == 0
        assert sm.avg_edge_angle_deg is None

    def test_snowboard_metrics(self):
        sm = SessionMetrics(
            discipline=Discipline.SNOWBOARD,
            frame_count=200,
            duration_s=6.67,
            avg_board_tilt_deg=25.0,
        )
        assert sm.avg_board_tilt_deg == 25.0
