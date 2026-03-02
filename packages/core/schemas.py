"""
Shared Pydantic v2 data models for the SnowClaw pipeline.

Coordinate system: Y-up, right-hand (X=right, Y=up, Z=toward camera).
All angles in degrees. All distances in meters.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class Discipline(str, Enum):
    SKI = "ski"
    SNOWBOARD = "snowboard"


class TurnPhaseLabel(str, Enum):
    INITIATION = "initiation"
    FALL_LINE = "fall_line"
    COMPLETION = "completion"
    TRANSITION = "transition"


class Pose3D(BaseModel):
    """
    3D keypoint set for a single person at a single frame.

    All positions in meters relative to the scene origin.
    Confidence values in [0, 1]; use to weight downstream metrics.
    """

    # Core body joints (Y-up coordinate system)
    head: list[float] = Field(..., min_length=3, max_length=3)
    neck: list[float] = Field(..., min_length=3, max_length=3)
    left_shoulder: list[float] = Field(..., min_length=3, max_length=3)
    right_shoulder: list[float] = Field(..., min_length=3, max_length=3)
    left_elbow: list[float] = Field(..., min_length=3, max_length=3)
    right_elbow: list[float] = Field(..., min_length=3, max_length=3)
    left_wrist: list[float] = Field(..., min_length=3, max_length=3)
    right_wrist: list[float] = Field(..., min_length=3, max_length=3)
    left_hip: list[float] = Field(..., min_length=3, max_length=3)
    right_hip: list[float] = Field(..., min_length=3, max_length=3)
    left_knee: list[float] = Field(..., min_length=3, max_length=3)
    right_knee: list[float] = Field(..., min_length=3, max_length=3)
    left_ankle: list[float] = Field(..., min_length=3, max_length=3)
    right_ankle: list[float] = Field(..., min_length=3, max_length=3)

    # Optional equipment keypoints
    left_ski_tip: Optional[list[float]] = Field(None, min_length=3, max_length=3)
    right_ski_tip: Optional[list[float]] = Field(None, min_length=3, max_length=3)
    left_ski_tail: Optional[list[float]] = Field(None, min_length=3, max_length=3)
    right_ski_tail: Optional[list[float]] = Field(None, min_length=3, max_length=3)
    board_nose: Optional[list[float]] = Field(None, min_length=3, max_length=3)
    board_tail: Optional[list[float]] = Field(None, min_length=3, max_length=3)

    # Per-joint confidence scores (optional)
    confidence: Optional[dict[str, float]] = None

    @field_validator("head", "neck", "left_shoulder", "right_shoulder",
                     "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                     "left_hip", "right_hip", "left_knee", "right_knee",
                     "left_ankle", "right_ankle", mode="before")
    @classmethod
    def validate_3d_point(cls, v: list) -> list:
        if len(v) != 3:
            raise ValueError("Each joint must be a 3-element [x, y, z] list")
        return [float(x) for x in v]

    def to_np(self, joint: str) -> np.ndarray:
        """Return a joint as a numpy array shape (3,)."""
        val = getattr(self, joint)
        if val is None:
            raise ValueError(f"Joint '{joint}' is not set in this Pose3D")
        return np.array(val, dtype=float)

    @property
    def com(self) -> np.ndarray:
        """Approximate center of mass from 14 body joints (equal weights)."""
        joints = [
            self.head, self.neck,
            self.left_shoulder, self.right_shoulder,
            self.left_hip, self.right_hip,
            self.left_knee, self.right_knee,
            self.left_ankle, self.right_ankle,
            self.left_elbow, self.right_elbow,
            self.left_wrist, self.right_wrist,
        ]
        return np.mean(np.array(joints, dtype=float), axis=0)

    @property
    def hip_midpoint(self) -> np.ndarray:
        return (np.array(self.left_hip) + np.array(self.right_hip)) / 2

    @property
    def shoulder_midpoint(self) -> np.ndarray:
        return (np.array(self.left_shoulder) + np.array(self.right_shoulder)) / 2

    @property
    def ankle_midpoint(self) -> np.ndarray:
        return (np.array(self.left_ankle) + np.array(self.right_ankle)) / 2

    @property
    def body_height(self) -> float:
        """Approximate body height = distance from head to average ankle."""
        return float(np.linalg.norm(
            np.array(self.head) - self.ankle_midpoint
        ))


class Frame(BaseModel):
    """A single video frame with pose, timestamp, and optional scene data."""

    frame_idx: int = Field(..., ge=0)
    timestamp_s: float = Field(..., ge=0.0, description="Seconds from video start")
    pose: Pose3D
    snow_surface_normal: list[float] = Field(
        default=[0.0, 1.0, 0.0],
        min_length=3, max_length=3,
        description="Unit normal of snow surface at this frame (Y-up default)",
    )
    discipline: Discipline = Discipline.SKI

    @field_validator("snow_surface_normal", mode="before")
    @classmethod
    def normalize_normal(cls, v: list) -> list:
        arr = np.array(v, dtype=float)
        norm = np.linalg.norm(arr)
        if norm < 1e-8:
            raise ValueError("Snow surface normal must be non-zero")
        return (arr / norm).tolist()


class TurnPhase(BaseModel):
    """A detected turn phase segment within a session."""

    label: TurnPhaseLabel
    start_frame: int
    end_frame: int
    direction: Optional[str] = Field(None, description="'left' or 'right'")
    avg_edge_angle_deg: Optional[float] = None
    avg_com_height_pct: Optional[float] = None
    peak_speed_ms: Optional[float] = None

    @model_validator(mode="after")
    def check_frame_order(self) -> "TurnPhase":
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be >= start_frame")
        return self


class SessionMetrics(BaseModel):
    """Aggregate biomechanical metrics for a full analysis session."""

    discipline: Discipline
    frame_count: int
    duration_s: float

    # Skiing CSIA skill metrics (mean ± std per session)
    avg_edge_angle_deg: Optional[float] = None
    std_edge_angle_deg: Optional[float] = None
    avg_inclination_deg: Optional[float] = None
    avg_angulation_deg: Optional[float] = None
    avg_knee_flex_left_deg: Optional[float] = None
    avg_knee_flex_right_deg: Optional[float] = None
    avg_fore_aft_balance: Optional[float] = None
    avg_lateral_balance: Optional[float] = None
    avg_upper_lower_separation_deg: Optional[float] = None
    avg_com_height_pct: Optional[float] = None
    avg_turn_radius_m: Optional[float] = None
    avg_speed_ms: Optional[float] = None

    # Snowboard CASI metrics
    avg_board_tilt_deg: Optional[float] = None
    avg_weight_distribution: Optional[float] = None
    avg_counter_rotation_deg: Optional[float] = None

    # Turn analysis
    turns_detected: int = 0
    left_turns: int = 0
    right_turns: int = 0
    turn_symmetry_pct: Optional[float] = Field(
        None, description="0=perfect symmetry, 100=max asymmetry"
    )
