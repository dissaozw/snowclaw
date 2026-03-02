"""
SnowIQ Scoring System

Converts per-skill scores (0–100) into a single SnowIQ number on a 0–200 scale,
aligned with the CSIA (skiing) and CASI (snowboarding) level frameworks.

Skill weights:
  - Rotary:       0.20
  - Edging:       0.25
  - Balance:      0.25
  - Pressure:     0.15
  - Coordination: 0.15

Level mapping (0–200 scale):
  0–50:   Beginner     (CSIA L1 student)
  51–100: Intermediate (CSIA L1 instructor standard)
  101–140: Advanced    (CSIA L2 standard)
  141–170: Expert      (CSIA L3 standard)
  171–200: Elite       (CSIA L4 / racing standard)
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

SKILL_WEIGHTS: dict[str, float] = {
    "rotary": 0.20,
    "edging": 0.25,
    "balance": 0.25,
    "pressure": 0.15,
    "coordination": 0.15,
}

assert abs(sum(SKILL_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

LEVEL_THRESHOLDS = [
    (171, "Elite",        "CSIA L4 / racing standard"),
    (141, "Expert",       "CSIA L3 standard"),
    (101, "Advanced",     "CSIA L2 standard"),
    (51,  "Intermediate", "CSIA L1 instructor standard"),
    (0,   "Beginner",     "CSIA L1 student level"),
]


class Level(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"
    ELITE = "Elite"


# ──────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────

class SkillScores(BaseModel):
    """Per-skill scores on a 0–100 scale."""

    rotary: float = Field(..., ge=0.0, le=100.0, description="Rotary skill score (0–100)")
    edging: float = Field(..., ge=0.0, le=100.0, description="Edging skill score (0–100)")
    balance: float = Field(..., ge=0.0, le=100.0, description="Balance skill score (0–100)")
    pressure: float = Field(..., ge=0.0, le=100.0, description="Pressure skill score (0–100)")
    coordination: float = Field(..., ge=0.0, le=100.0, description="Coordination skill score (0–100)")

    @field_validator("rotary", "edging", "balance", "pressure", "coordination", mode="before")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(float(v), 2)

    def weakest_skill(self) -> str:
        """Return the name of the lowest-scoring skill."""
        return min(SKILL_WEIGHTS.keys(), key=lambda s: getattr(self, s))

    def strongest_skill(self) -> str:
        """Return the name of the highest-scoring skill."""
        return max(SKILL_WEIGHTS.keys(), key=lambda s: getattr(self, s))

    def delta_to_next_level(self, current_snow_iq: float) -> dict[str, float]:
        """
        Estimate how much each skill must improve to reach the next SnowIQ level.
        Returns a dict mapping skill name → required improvement (0 if already sufficient).
        """
        next_threshold = _next_level_threshold(current_snow_iq)
        if next_threshold is None:
            return {s: 0.0 for s in SKILL_WEIGHTS}

        result = {}
        for skill, weight in SKILL_WEIGHTS.items():
            current_weighted = getattr(self, skill) * weight
            # Required contribution from this skill if all others stay constant
            needed_total_weighted = next_threshold / 2.0  # SnowIQ = weighted_avg * 2
            needed_skill = needed_total_weighted / weight
            delta = max(0.0, needed_skill - getattr(self, skill))
            result[skill] = round(min(delta, 100.0 - getattr(self, skill)), 2)
        return result


class SnowIQResult(BaseModel):
    """Full SnowIQ scoring result."""

    skills: SkillScores
    snow_iq: float = Field(..., ge=0.0, le=200.0)
    level: Level
    level_description: str
    weakest_skill: str
    strongest_skill: str
    percentile_in_level: float = Field(
        ..., ge=0.0, le=100.0,
        description="How far through the current level (0=just entered, 100=about to advance)"
    )

    @model_validator(mode="after")
    def check_consistency(self) -> "SnowIQResult":
        expected_level = _classify_level(self.snow_iq)
        if self.level != expected_level:
            raise ValueError(
                f"Level {self.level} inconsistent with SnowIQ {self.snow_iq:.1f} "
                f"(expected {expected_level})"
            )
        return self


# ──────────────────────────────────────────────────────────────
# Core calculator
# ──────────────────────────────────────────────────────────────

class SnowIQCalculator:
    """
    Compute SnowIQ from per-skill scores.

    Usage::
        calc = SnowIQCalculator()
        skills = SkillScores(rotary=70, edging=65, balance=80, pressure=58, coordination=72)
        result = calc.score(skills)
        print(result.snow_iq)       # e.g. 136.4
        print(result.level)         # Level.ADVANCED
        print(result.weakest_skill) # 'pressure'
    """

    def score(self, skills: SkillScores) -> SnowIQResult:
        """
        Compute SnowIQ from per-skill scores.

        Formula::
            weighted_avg = Σ (skill_score_i × weight_i)
            SnowIQ       = weighted_avg × 2   → maps [0,100] weighted avg to [0,200]

        Args:
            skills: SkillScores with each skill rated 0–100.

        Returns:
            SnowIQResult with score, level, and diagnostic information.
        """
        weighted_avg = sum(
            getattr(skills, skill) * weight
            for skill, weight in SKILL_WEIGHTS.items()
        )
        snow_iq = float(np.clip(weighted_avg * 2.0, 0.0, 200.0))
        level = _classify_level(snow_iq)
        level_desc = _level_description(level)
        percentile = _percentile_in_level(snow_iq, level)

        return SnowIQResult(
            skills=skills,
            snow_iq=round(snow_iq, 2),
            level=level,
            level_description=level_desc,
            weakest_skill=skills.weakest_skill(),
            strongest_skill=skills.strongest_skill(),
            percentile_in_level=round(percentile, 1),
        )

    def score_from_dict(self, scores: dict[str, float]) -> SnowIQResult:
        """Convenience method — pass scores as a plain dict."""
        return self.score(SkillScores(**scores))

    def compare_sessions(
        self,
        baseline: SkillScores,
        current: SkillScores,
    ) -> dict[str, float]:
        """
        Return per-skill deltas (current − baseline) and overall SnowIQ delta.

        Positive values indicate improvement.

        Returns:
            Dict with keys: 'rotary', 'edging', 'balance', 'pressure',
            'coordination', 'snow_iq_delta'.
        """
        baseline_result = self.score(baseline)
        current_result = self.score(current)
        deltas: dict[str, float] = {}
        for skill in SKILL_WEIGHTS:
            deltas[skill] = round(getattr(current, skill) - getattr(baseline, skill), 2)
        deltas["snow_iq_delta"] = round(current_result.snow_iq - baseline_result.snow_iq, 2)
        return deltas


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _classify_level(snow_iq: float) -> Level:
    for threshold, label, _ in LEVEL_THRESHOLDS:
        if snow_iq >= threshold:
            return Level(label)
    return Level.BEGINNER


def _level_description(level: Level) -> str:
    for _, label, desc in LEVEL_THRESHOLDS:
        if label == level.value:
            return desc
    return ""


def _percentile_in_level(snow_iq: float, level: Level) -> float:
    """0 = just entered this level, 100 = about to advance to next."""
    bands = {
        Level.BEGINNER:     (0.0,   50.0),
        Level.INTERMEDIATE: (51.0,  100.0),
        Level.ADVANCED:     (101.0, 140.0),
        Level.EXPERT:       (141.0, 170.0),
        Level.ELITE:        (171.0, 200.0),
    }
    lo, hi = bands[level]
    span = hi - lo
    if span < 1e-8:
        return 100.0
    return float(np.clip((snow_iq - lo) / span * 100.0, 0.0, 100.0))


def _next_level_threshold(snow_iq: float) -> float | None:
    """Return the SnowIQ value needed to enter the next level, or None if Elite."""
    breakpoints = [50.0, 100.0, 140.0, 170.0, 200.0]
    for bp in breakpoints:
        if snow_iq < bp:
            return bp
    return None
