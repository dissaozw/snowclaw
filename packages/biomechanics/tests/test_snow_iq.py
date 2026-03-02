"""Unit tests for the SnowIQ scoring system."""

import pytest

from packages.biomechanics.snow_iq import (
    SkillScores,
    SnowIQCalculator,
    Level,
    SKILL_WEIGHTS,
)


@pytest.fixture
def calc() -> SnowIQCalculator:
    return SnowIQCalculator()


# ──────────────────────────────────────────────────────────────
# Weight invariants
# ──────────────────────────────────────────────────────────────

class TestWeights:
    def test_weights_sum_to_one(self):
        assert sum(SKILL_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)

    def test_all_weights_positive(self):
        assert all(w > 0 for w in SKILL_WEIGHTS.values())


# ──────────────────────────────────────────────────────────────
# Boundary scores
# ──────────────────────────────────────────────────────────────

class TestBoundaryScores:
    def test_all_zero_gives_zero_snow_iq(self, calc):
        skills = SkillScores(rotary=0, edging=0, balance=0, pressure=0, coordination=0)
        result = calc.score(skills)
        assert result.snow_iq == pytest.approx(0.0, abs=1e-6)
        assert result.level == Level.BEGINNER

    def test_all_100_gives_200_snow_iq(self, calc):
        skills = SkillScores(rotary=100, edging=100, balance=100, pressure=100, coordination=100)
        result = calc.score(skills)
        assert result.snow_iq == pytest.approx(200.0, abs=1e-4)
        assert result.level == Level.ELITE

    def test_all_50_gives_100_snow_iq(self, calc):
        skills = SkillScores(rotary=50, edging=50, balance=50, pressure=50, coordination=50)
        result = calc.score(skills)
        assert result.snow_iq == pytest.approx(100.0, abs=1e-4)
        assert result.level == Level.INTERMEDIATE


# ──────────────────────────────────────────────────────────────
# Level classification
# ──────────────────────────────────────────────────────────────

class TestLevelClassification:
    @pytest.mark.parametrize("score,expected_level", [
        (0.0,   Level.BEGINNER),
        (25.0,  Level.BEGINNER),
        (50.0,  Level.BEGINNER),
        (51.0,  Level.INTERMEDIATE),
        (75.0,  Level.INTERMEDIATE),
        (100.0, Level.INTERMEDIATE),
        (101.0, Level.ADVANCED),
        (120.0, Level.ADVANCED),
        (140.0, Level.ADVANCED),
        (141.0, Level.EXPERT),
        (155.0, Level.EXPERT),
        (170.0, Level.EXPERT),
        (171.0, Level.ELITE),
        (185.0, Level.ELITE),
        (200.0, Level.ELITE),
    ])
    def test_level_thresholds(self, calc, score, expected_level):
        # Reverse-engineer skill scores that give the desired SnowIQ
        raw_skill = score / 2.0  # weighted_avg = snow_iq / 2
        skills = SkillScores(
            rotary=raw_skill,
            edging=raw_skill,
            balance=raw_skill,
            pressure=raw_skill,
            coordination=raw_skill,
        )
        result = calc.score(skills)
        assert result.snow_iq == pytest.approx(score, abs=1e-3)
        assert result.level == expected_level


# ──────────────────────────────────────────────────────────────
# Weakest / strongest skill
# ──────────────────────────────────────────────────────────────

class TestSkillDiagnostics:
    def test_weakest_skill_identified(self, calc):
        skills = SkillScores(rotary=70, edging=65, balance=80, pressure=55, coordination=72)
        result = calc.score(skills)
        assert result.weakest_skill == "pressure"

    def test_strongest_skill_identified(self, calc):
        skills = SkillScores(rotary=70, edging=65, balance=85, pressure=55, coordination=72)
        result = calc.score(skills)
        assert result.strongest_skill == "balance"

    def test_all_equal_any_skill_is_weakest(self):
        skills = SkillScores(rotary=60, edging=60, balance=60, pressure=60, coordination=60)
        # Any skill is equally weak — just check it returns a valid key
        assert skills.weakest_skill() in SKILL_WEIGHTS


# ──────────────────────────────────────────────────────────────
# Session comparison
# ──────────────────────────────────────────────────────────────

class TestSessionComparison:
    def test_improvement_positive_delta(self, calc):
        baseline = SkillScores(rotary=60, edging=55, balance=65, pressure=50, coordination=58)
        current = SkillScores(rotary=70, edging=65, balance=75, pressure=60, coordination=68)
        deltas = calc.compare_sessions(baseline, current)
        assert deltas["snow_iq_delta"] > 0
        assert deltas["balance"] == pytest.approx(10.0, abs=1e-4)

    def test_no_change_zero_delta(self, calc):
        skills = SkillScores(rotary=70, edging=70, balance=70, pressure=70, coordination=70)
        deltas = calc.compare_sessions(skills, skills)
        assert deltas["snow_iq_delta"] == pytest.approx(0.0, abs=1e-4)

    def test_regression_negative_delta(self, calc):
        baseline = SkillScores(rotary=80, edging=80, balance=80, pressure=80, coordination=80)
        current = SkillScores(rotary=60, edging=60, balance=60, pressure=60, coordination=60)
        deltas = calc.compare_sessions(baseline, current)
        assert deltas["snow_iq_delta"] < 0


# ──────────────────────────────────────────────────────────────
# Percentile in level
# ──────────────────────────────────────────────────────────────

class TestPercentileInLevel:
    def test_just_entered_level_low_percentile(self, calc):
        # All skills = 26 → weighted_avg = 26 → SnowIQ = 52 → just entered Intermediate
        skills = SkillScores(rotary=26, edging=26, balance=26, pressure=26, coordination=26)
        result = calc.score(skills)
        assert result.level == Level.INTERMEDIATE
        assert result.percentile_in_level < 10.0  # near bottom of Intermediate

    def test_about_to_advance_high_percentile(self, calc):
        # SnowIQ ≈ 140 = top of Advanced
        skills = SkillScores(rotary=70, edging=70, balance=70, pressure=70, coordination=70)
        result = calc.score(skills)
        assert result.level == Level.ADVANCED
        assert result.percentile_in_level > 50.0

    def test_elite_max_percentile(self, calc):
        skills = SkillScores(rotary=100, edging=100, balance=100, pressure=100, coordination=100)
        result = calc.score(skills)
        assert result.percentile_in_level == pytest.approx(100.0, abs=1e-4)


# ──────────────────────────────────────────────────────────────
# from_dict convenience method
# ──────────────────────────────────────────────────────────────

class TestFromDict:
    def test_dict_matches_direct(self, calc):
        d = {"rotary": 72, "edging": 68, "balance": 79, "pressure": 61, "coordination": 70}
        result_dict = calc.score_from_dict(d)
        result_direct = calc.score(SkillScores(**d))
        assert result_dict.snow_iq == pytest.approx(result_direct.snow_iq, abs=1e-6)

    def test_invalid_score_raises(self):
        with pytest.raises(Exception):
            SkillScores(rotary=101, edging=50, balance=50, pressure=50, coordination=50)
