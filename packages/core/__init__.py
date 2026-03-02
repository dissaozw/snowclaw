"""
SnowClaw Core — shared data structures for the SnowClaw pipeline.

All packages import schema types from here to keep the dependency graph clean.
"""

from .schemas import (
    Discipline,
    Frame,
    Pose3D,
    SessionMetrics,
    TurnPhase,
    TurnPhaseLabel,
)

__all__ = [
    "Discipline",
    "Frame",
    "Pose3D",
    "SessionMetrics",
    "TurnPhase",
    "TurnPhaseLabel",
]
