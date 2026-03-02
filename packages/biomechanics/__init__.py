"""
SnowClaw Biomechanics Engine
=============================
Pure-numpy biomechanical metrics for skiing and snowboarding analysis.
No ML dependencies — operates on 3D keypoints from any upstream pose model.

Coordinate system: Y-up, right-hand (X=right, Y=up, Z=toward camera).
Snow surface normal ≈ [0, 1, 0] on flat terrain.
All angles are in degrees unless noted.
"""

from .metrics import (
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
from .snow_iq import SnowIQCalculator, SkillScores, SnowIQResult, Level
# Re-export shared types from core for backward compatibility
from core.schemas import Pose3D, Frame, TurnPhase, SessionMetrics
from .turn_segmentation import segment_turns

__all__ = [
    "edge_angle", "inclination_angle", "angulation",
    "knee_flex_angle", "hip_flex_angle",
    "fore_aft_balance", "lateral_balance",
    "upper_lower_separation", "com_height_pct",
    "turn_radius_estimate", "speed_estimate",
    "board_tilt_angle", "fore_aft_weight_distribution", "counter_rotation",
    "SnowIQCalculator", "SkillScores", "SnowIQResult", "Level",
    "Pose3D", "Frame", "TurnPhase", "SessionMetrics",
    "segment_turns",
]
