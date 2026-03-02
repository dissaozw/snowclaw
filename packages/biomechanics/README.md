# `packages/biomechanics` — SnowClaw Biomechanics Engine

Pure-numpy biomechanical analysis for skiing and snowboarding.
**No ML dependencies** — operates on 3D keypoints from any upstream pose model.

---

## Overview

This package implements the metric computation layer described in the
[SnowCoach AI Technical Plan](../../PLAN.md) — Sections 4.3 (Biomechanical Analysis)
and 4.5 (SnowIQ Scoring).

It is intentionally decoupled from the video/pose pipeline: you feed it numpy arrays
of 3D keypoints (output of MotionBERT, BioPose, or any other model), and it returns
structured metrics and coaching scores.

---

## Coordinate System

| Axis | Direction        |
|------|-----------------|
| X    | Right            |
| Y    | Up (gravity ↓)   |
| Z    | Toward camera    |

Snow surface normal ≈ `[0, 1, 0]` on flat terrain.
All angles are in **degrees** unless noted. All distances in **metres**.

---

## Installation

```bash
pip install -e ".[dev]"   # from repo root
```

Dependencies: `numpy>=1.26`, `pydantic>=2.0`, `scipy>=1.11`

---

## Quick Start

```python
import numpy as np
from packages.biomechanics import (
    edge_angle, knee_flex_angle, com_height_pct,
    SnowIQCalculator, SkillScores,
    Pose3D, segment_turns,
)

# ── Compute a single metric ─────────────────────────────────────
ski_normal  = np.array([0.766, 0.643, 0.0])  # ski tilted ~40° from flat
snow_normal = np.array([0.0,   1.0,   0.0])
print(f"Edge angle: {edge_angle(ski_normal, snow_normal):.1f}°")  # → 40.0°

# ── Knee flex from 3D keypoints ─────────────────────────────────
hip   = np.array([0.0, 0.95, 0.0])
knee  = np.array([0.0, 0.52, 0.04])
ankle = np.array([0.0, 0.05, 0.0])
print(f"Knee flex: {knee_flex_angle(hip, knee, ankle):.1f}°")  # → ~150°

# ── SnowIQ scoring ──────────────────────────────────────────────
calc   = SnowIQCalculator()
skills = SkillScores(rotary=72, edging=65, balance=80, pressure=58, coordination=70)
result = calc.score(skills)
print(f"SnowIQ: {result.snow_iq:.1f}  Level: {result.level.value}")
# → SnowIQ: 136.4  Level: Advanced
print(f"Weakest skill: {result.weakest_skill}")
# → pressure

# ── Turn segmentation ───────────────────────────────────────────
# com_trajectory shape: (N, 3) — COM positions from pose model output
com_trajectory = np.random.randn(120, 3)  # replace with real data
turns = segment_turns(com_trajectory, fps=60.0)
for t in turns:
    print(f"  {t.label.value:12s} frames {t.start_frame}–{t.end_frame}  dir={t.direction}")
```

---

## Modules

### `metrics.py` — Biomechanical Metrics

All functions accept numpy arrays and return scalars (float).

#### Skiing Metrics (CSIA Five Skills)

| Function | CSIA Skill | Description |
|----------|-----------|-------------|
| `edge_angle(ski_plane_normal, snow_surface_normal)` | Edging | Ski tilt relative to snow [0°–90°] |
| `inclination_angle(body_axis, vertical?)` | Balance | Whole-body lean from vertical |
| `angulation(upper_plane_normal, lower_plane_normal)` | Edging + Balance | Upper/lower body lateral separation |
| `knee_flex_angle(hip, knee, ankle)` | Pressure | Knee joint flexion [0°–180°] |
| `hip_flex_angle(spine_base, hip, knee)` | Pressure | Hip joint flexion [0°–180°] |
| `fore_aft_balance(com, boot_midpoint, ski_axis)` | Balance | COM fore-aft position [-1, 1] |
| `lateral_balance(com, boot_midpoint, ski_perp)` | Balance | COM lateral position [-1, 1] |
| `upper_lower_separation(l_shoulder, r_shoulder, l_hip, r_hip)` | Rotary | Shoulder vs hip rotation [-180°, 180°] |
| `com_height_pct(com, snow_pt, snow_normal, body_height)` | Pressure | COM height fraction [0, 1] |
| `turn_radius_estimate(com_trajectory)` | Coordination | Turn radius in metres |
| `speed_estimate(com_trajectory, fps)` | — | Average speed in m/s |

#### Snowboard Metrics (CASI Framework)

| Function | CASI Skill | Description |
|----------|-----------|-------------|
| `board_tilt_angle(board_normal, snow_normal)` | Edging | Board edge angle [0°–90°] |
| `fore_aft_weight_distribution(com, board_center, board_axis)` | Balance | Board weight distribution [-1, 1] |
| `counter_rotation(l_shoulder, r_shoulder, board_heading)` | Rotary | Upper body vs board rotation [-180°, 180°] |

---

### `snow_iq.py` — SnowIQ Scoring

```
SnowIQ = Σ(skill_score_i × weight_i) × 2
```

| Skill | Weight |
|-------|--------|
| Edging | 0.25 |
| Balance | 0.25 |
| Rotary | 0.20 |
| Coordination | 0.15 |
| Pressure | 0.15 |

**Scale (0–200):**

| Range | Level | CSIA Equivalent |
|-------|-------|----------------|
| 0–50 | Beginner | L1 student |
| 51–100 | Intermediate | L1 instructor standard |
| 101–140 | Advanced | L2 standard |
| 141–170 | Expert | L3 standard |
| 171–200 | Elite | L4 / racing standard |

---

### `turn_segmentation.py` — Turn Phase Detection

Detects turn phases from a COM trajectory using lateral velocity zero-crossings
and Savitzky-Golay smoothing.

**Phases:** `INITIATION` → `FALL_LINE` → `COMPLETION` → `TRANSITION`

```python
turns = segment_turns(com_trajectory, fps=60.0)
# Returns list[TurnPhase] ordered by frame index
```

---

### `schemas.py` — Pydantic v2 Data Models

- **`Pose3D`** — 3D keypoint set for a single frame (14 body joints + optional equipment)
- **`Frame`** — Frame with pose + timestamp + snow surface
- **`TurnPhase`** — A labelled turn segment
- **`SessionMetrics`** — Aggregate metrics for a full session

---

## Running Tests

```bash
# From repo root
pytest packages/biomechanics/tests/ -v

# With coverage
pytest packages/biomechanics/tests/ -v --cov=packages/biomechanics --cov-report=term-missing
```

---

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

When adding a new metric:
1. Add the function to `metrics.py` with full docstring (units, coord system, example)
2. Add corresponding tests to `tests/test_metrics.py` (normal + edge cases)
3. Export from `__init__.py`
4. Document in this README
