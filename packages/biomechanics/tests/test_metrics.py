"""
Unit tests for the biomechanics metrics module.

Covers: normal cases, edge cases (parallel/zero vectors),
boundary conditions, and round-trip sanity checks.
"""

import numpy as np
import pytest

from packages.biomechanics.metrics import (
    edge_angle,
    inclination_angle,
    angulation,
    knee_flex_angle,
    hip_flex_angle,
    fore_aft_balance,
    lateral_balance,
    upper_lower_separation,
    com_height_pct,
    turn_radius_estimate,
    speed_estimate,
    board_tilt_angle,
    fore_aft_weight_distribution,
    counter_rotation,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

UP = np.array([0.0, 1.0, 0.0])
RIGHT = np.array([1.0, 0.0, 0.0])
FORWARD = np.array([0.0, 0.0, 1.0])


# ──────────────────────────────────────────────────────────────
# edge_angle
# ──────────────────────────────────────────────────────────────

class TestEdgeAngle:
    def test_flat_ski_zero_degrees(self):
        """Flat ski — ski normal aligned with snow normal."""
        assert edge_angle(UP, UP) == pytest.approx(0.0, abs=1e-6)

    def test_fully_on_edge_90_degrees(self):
        """Ski fully on edge — ski normal perpendicular to snow."""
        assert edge_angle(RIGHT, UP) == pytest.approx(90.0, abs=1e-6)

    def test_45_degrees(self):
        diag = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        assert edge_angle(diag, UP) == pytest.approx(45.0, abs=1e-4)

    def test_returns_at_most_90(self):
        """Edge angle should never exceed 90°."""
        for _ in range(50):
            v = np.random.randn(3)
            v /= np.linalg.norm(v)
            result = edge_angle(v, UP)
            assert 0.0 <= result <= 90.0

    def test_flipped_normal_same_result(self):
        """Flipping the ski normal should give the same edge angle."""
        n = np.array([0.5, 0.866, 0.0])
        assert edge_angle(n, UP) == pytest.approx(edge_angle(-n, UP), abs=1e-6)


# ──────────────────────────────────────────────────────────────
# inclination_angle
# ──────────────────────────────────────────────────────────────

class TestInclinationAngle:
    def test_upright_zero_degrees(self):
        assert inclination_angle(UP) == pytest.approx(0.0, abs=1e-6)

    def test_horizontal_90_degrees(self):
        assert inclination_angle(FORWARD) == pytest.approx(90.0, abs=1e-6)

    def test_45_degree_lean(self):
        v = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        assert inclination_angle(v) == pytest.approx(45.0, abs=1e-4)

    def test_custom_vertical(self):
        """Non-standard vertical reference (tilted slope)."""
        slope_normal = np.array([0.0, 0.866, 0.5])
        assert inclination_angle(slope_normal, slope_normal) == pytest.approx(0.0, abs=1e-4)


# ──────────────────────────────────────────────────────────────
# knee_flex_angle
# ──────────────────────────────────────────────────────────────

class TestKneeFlexAngle:
    def test_straight_leg_180_degrees(self):
        hip = np.array([0.0, 2.0, 0.0])
        knee = np.array([0.0, 1.0, 0.0])
        ankle = np.array([0.0, 0.0, 0.0])
        assert knee_flex_angle(hip, knee, ankle) == pytest.approx(180.0, abs=1e-4)

    def test_right_angle_90_degrees(self):
        hip = np.array([0.0, 1.0, 0.0])
        knee = np.array([0.0, 0.0, 0.0])
        ankle = np.array([1.0, 0.0, 0.0])
        assert knee_flex_angle(hip, knee, ankle) == pytest.approx(90.0, abs=1e-4)

    def test_ski_athletic_stance(self):
        """Typical ski stance: ~145° knee flex."""
        # Slight forward bend
        hip = np.array([0.0, 0.9, -0.1])
        knee = np.array([0.0, 0.5, 0.05])
        ankle = np.array([0.0, 0.0, 0.0])
        angle = knee_flex_angle(hip, knee, ankle)
        assert 100.0 <= angle <= 180.0  # valid athletic range

    def test_symmetry_left_right(self):
        """Left and right knee at same flex should return same angle."""
        hip = np.array([0.0, 1.0, 0.0])
        knee = np.array([0.0, 0.5, 0.0])
        ankle = np.array([0.0, 0.0, 0.0])
        hip_r = np.array([0.3, 1.0, 0.0])
        knee_r = np.array([0.3, 0.5, 0.0])
        ankle_r = np.array([0.3, 0.0, 0.0])
        assert knee_flex_angle(hip, knee, ankle) == pytest.approx(
            knee_flex_angle(hip_r, knee_r, ankle_r), abs=1e-6
        )


# ──────────────────────────────────────────────────────────────
# hip_flex_angle
# ──────────────────────────────────────────────────────────────

class TestHipFlexAngle:
    def test_upright_180_degrees(self):
        spine = np.array([0.0, 2.0, 0.0])
        hip = np.array([0.0, 1.0, 0.0])
        knee = np.array([0.0, 0.0, 0.0])
        assert hip_flex_angle(spine, hip, knee) == pytest.approx(180.0, abs=1e-4)

    def test_right_angle(self):
        spine = np.array([1.0, 1.0, 0.0])
        hip = np.array([0.0, 1.0, 0.0])
        knee = np.array([0.0, 0.0, 0.0])
        assert hip_flex_angle(spine, hip, knee) == pytest.approx(90.0, abs=1e-4)


# ──────────────────────────────────────────────────────────────
# fore_aft_balance
# ──────────────────────────────────────────────────────────────

class TestForeAftBalance:
    def test_neutral_position(self):
        com = np.array([0.0, 1.0, 0.0])
        boot = np.array([0.0, 0.0, 0.0])
        ski_axis = FORWARD
        assert fore_aft_balance(com, boot, ski_axis) == pytest.approx(0.0, abs=1e-6)

    def test_fully_forward(self):
        com = np.array([0.0, 1.0, 0.85])  # 0.85 m forward, ski_length=1.7
        boot = np.array([0.0, 0.0, 0.0])
        assert fore_aft_balance(com, boot, FORWARD) == pytest.approx(1.0, abs=1e-4)

    def test_fully_back(self):
        com = np.array([0.0, 1.0, -0.85])
        boot = np.array([0.0, 0.0, 0.0])
        assert fore_aft_balance(com, boot, FORWARD) == pytest.approx(-1.0, abs=1e-4)

    def test_clamped_beyond_range(self):
        com = np.array([0.0, 1.0, 5.0])  # extreme forward
        boot = np.array([0.0, 0.0, 0.0])
        result = fore_aft_balance(com, boot, FORWARD)
        assert result == pytest.approx(1.0, abs=1e-6)


# ──────────────────────────────────────────────────────────────
# upper_lower_separation
# ──────────────────────────────────────────────────────────────

class TestUpperLowerSeparation:
    def test_no_separation(self):
        ls = np.array([-0.3, 1.5, 0.0])
        rs = np.array([0.3, 1.5, 0.0])
        lh = np.array([-0.2, 0.9, 0.0])
        rh = np.array([0.2, 0.9, 0.0])
        assert upper_lower_separation(ls, rs, lh, rh) == pytest.approx(0.0, abs=1e-4)

    def test_positive_rotation(self):
        """Right shoulder rotated backward relative to hips → positive angle (CCW around Y)."""
        # Hips: pointing right [+X]; Shoulders: right+backward [+X, -Z]
        # cross(hip_horiz, shoulder_horiz) points in +Y → positive signed angle
        ls = np.array([-0.3, 1.5, 0.1])
        rs = np.array([0.3, 1.5, -0.1])
        lh = np.array([-0.2, 0.9, 0.0])
        rh = np.array([0.2, 0.9, 0.0])
        sep = upper_lower_separation(ls, rs, lh, rh)
        assert sep > 0

    def test_antisymmetry(self):
        """Swapping L/R hips (reversing the hip reference vector) flips the sign."""
        ls = np.array([-0.3, 1.5, 0.1])
        rs = np.array([0.3, 1.5, -0.1])
        lh = np.array([-0.2, 0.9, 0.0])
        rh = np.array([0.2, 0.9, 0.0])
        pos = upper_lower_separation(ls, rs, lh, rh)
        neg = upper_lower_separation(ls, rs, rh, lh)
        # Reversing the hip reference reverses the signed angle direction
        assert pos > 0 and neg < 0


# ──────────────────────────────────────────────────────────────
# com_height_pct
# ──────────────────────────────────────────────────────────────

class TestComHeightPct:
    def test_full_height(self):
        com = np.array([0.0, 1.8, 0.0])  # 1.8 m above snow
        snow_pt = np.array([0.0, 0.0, 0.0])
        assert com_height_pct(com, snow_pt, UP, body_height=1.8) == pytest.approx(1.0, abs=1e-6)

    def test_half_height(self):
        com = np.array([0.0, 0.9, 0.0])
        snow_pt = np.array([0.0, 0.0, 0.0])
        assert com_height_pct(com, snow_pt, UP, body_height=1.8) == pytest.approx(0.5, abs=1e-4)

    def test_at_snow_level(self):
        com = np.array([0.0, 0.0, 0.0])
        snow_pt = np.array([0.0, 0.0, 0.0])
        assert com_height_pct(com, snow_pt, UP, body_height=1.8) == pytest.approx(0.0, abs=1e-6)

    def test_clamped_negative(self):
        """COM below snow surface should clamp to 0."""
        com = np.array([0.0, -0.5, 0.0])
        snow_pt = np.array([0.0, 0.0, 0.0])
        assert com_height_pct(com, snow_pt, UP, body_height=1.8) == pytest.approx(0.0, abs=1e-6)

    def test_zero_body_height(self):
        com = np.array([0.0, 1.0, 0.0])
        snow_pt = np.array([0.0, 0.0, 0.0])
        assert com_height_pct(com, snow_pt, UP, body_height=0.0) == pytest.approx(0.0, abs=1e-6)


# ──────────────────────────────────────────────────────────────
# turn_radius_estimate
# ──────────────────────────────────────────────────────────────

class TestTurnRadiusEstimate:
    def test_circle_radius_1(self):
        t = np.linspace(0, np.pi, 30)
        pts = np.column_stack([np.cos(t), np.zeros(30), np.sin(t)])
        r = turn_radius_estimate(pts)
        assert abs(r - 1.0) < 0.05

    def test_circle_radius_10(self):
        t = np.linspace(0, np.pi, 50)
        pts = np.column_stack([10 * np.cos(t), np.zeros(50), 10 * np.sin(t)])
        r = turn_radius_estimate(pts)
        assert abs(r - 10.0) < 0.5

    def test_straight_line_returns_inf(self):
        pts = np.column_stack([np.zeros(10), np.zeros(10), np.linspace(0, 10, 10)])
        r = turn_radius_estimate(pts)
        assert r == float("inf") or r > 1e6

    def test_too_few_points(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert turn_radius_estimate(pts) == float("inf")


# ──────────────────────────────────────────────────────────────
# speed_estimate
# ──────────────────────────────────────────────────────────────

class TestSpeedEstimate:
    def test_constant_speed(self):
        """Points spaced 1 m apart at 30 fps → speed = 30 m/s."""
        pts = np.column_stack([np.linspace(0, 9, 10), np.zeros(10), np.zeros(10)])
        assert speed_estimate(pts, fps=30.0) == pytest.approx(30.0, rel=1e-4)

    def test_zero_fps(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert speed_estimate(pts, fps=0.0) == pytest.approx(0.0)

    def test_single_frame(self):
        pts = np.array([[0.0, 0.0, 0.0]])
        assert speed_estimate(pts, fps=30.0) == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────
# board_tilt_angle (snowboard)
# ──────────────────────────────────────────────────────────────

class TestBoardTiltAngle:
    def test_flat_board(self):
        assert board_tilt_angle(UP, UP) == pytest.approx(0.0, abs=1e-6)

    def test_on_heelside_edge(self):
        assert board_tilt_angle(RIGHT, UP) == pytest.approx(90.0, abs=1e-6)

    def test_equivalent_to_edge_angle(self):
        n = np.array([0.6, 0.8, 0.0])
        assert board_tilt_angle(n, UP) == pytest.approx(edge_angle(n, UP), abs=1e-6)


# ──────────────────────────────────────────────────────────────
# counter_rotation (snowboard)
# ──────────────────────────────────────────────────────────────

class TestCounterRotation:
    def test_no_rotation(self):
        """Upper body aligned with board heading → 0° counter-rotation."""
        ls = np.array([-0.3, 1.5, 0.0])
        rs = np.array([0.3, 1.5, 0.0])
        board_heading = RIGHT
        assert counter_rotation(ls, rs, board_heading) == pytest.approx(0.0, abs=1e-4)

    def test_90_degree_rotation(self):
        """Upper body perpendicular to board → 90° or -90°."""
        ls = np.array([0.0, 1.5, -0.3])
        rs = np.array([0.0, 1.5, 0.3])
        board_heading = RIGHT
        cr = counter_rotation(ls, rs, board_heading)
        assert abs(cr) == pytest.approx(90.0, abs=1e-4)
