"""
Turn phase segmentation from COM trajectory.

Detects left/right turns and labels each frame with:
  - INITIATION   (entering the fall line)
  - FALL_LINE    (crossing the fall line)
  - COMPLETION   (exiting the turn)
  - TRANSITION   (between turns)

Algorithm:
  1. Project COM trajectory onto the XZ (horizontal) plane.
  2. Apply Savitzky-Golay smoothing to remove noise.
  3. Compute the horizontal velocity direction (heading angle).
  4. Detect sign changes in the lateral (X) velocity to identify turn reversals.
  5. Label phases within each turn based on heading relative to fall line.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter  # type: ignore

from core.schemas import TurnPhase, TurnPhaseLabel


def segment_turns(
    com_trajectory: np.ndarray,
    fps: float,
    fall_line_direction: np.ndarray | None = None,
    min_turn_frames: int = 10,
    smoothing_window: int = 15,
    smoothing_poly: int = 3,
) -> list[TurnPhase]:
    """
    Detect and label turn phases from a COM trajectory.

    Args:
        com_trajectory:    COM positions over time, shape (N, 3). Y-up convention.
        fps:               Video frame rate.
        fall_line_direction: Unit vector of the fall line in XZ plane.
                            Defaults to [0, 0, 1] (toward camera = downhill).
        min_turn_frames:   Minimum frames for a turn to be counted (noise filter).
        smoothing_window:  Savitzky-Golay window length (must be odd, > smoothing_poly).
        smoothing_poly:    Savitzky-Golay polynomial order.

    Returns:
        List of TurnPhase objects ordered by frame index.

    Raises:
        ValueError: If com_trajectory has fewer than 3 frames or invalid shape.
    """
    if com_trajectory.ndim != 2 or com_trajectory.shape[1] != 3:
        raise ValueError("com_trajectory must have shape (N, 3)")
    n = len(com_trajectory)
    if n < 3:
        raise ValueError("Need at least 3 frames to segment turns")

    if fall_line_direction is None:
        fall_line_direction = np.array([0.0, 0.0, 1.0])
    fall_line_direction = fall_line_direction / (np.linalg.norm(fall_line_direction) + 1e-8)

    # ── 1. Smooth the XZ trajectory ──────────────────────────
    window = min(smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1, n)
    if window <= smoothing_poly:
        window = smoothing_poly + (1 if smoothing_poly % 2 == 0 else 2)
    window = min(window, n if n % 2 == 1 else n - 1)

    x_smooth = savgol_filter(com_trajectory[:, 0], window_length=window, polyorder=smoothing_poly)
    z_smooth = savgol_filter(com_trajectory[:, 2], window_length=window, polyorder=smoothing_poly)

    # ── 2. Compute lateral velocity (X direction) ─────────────
    vx = np.gradient(x_smooth, 1.0 / fps)

    # ── 3. Find zero crossings of lateral velocity ────────────
    sign_changes = np.where(np.diff(np.sign(vx)))[0]

    # ── 4. Build turn segments ────────────────────────────────
    boundaries = [0] + list(sign_changes + 1) + [n]
    turns: list[TurnPhase] = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1] - 1
        length = end - start + 1

        if length < min_turn_frames:
            continue

        # Determine turn direction from sign of vx in this segment
        vx_seg = vx[start:end + 1]
        direction = "right" if np.mean(vx_seg) > 0 else "left"

        # ── 5. Label sub-phases within this turn ─────────────
        # Phase boundaries: first 25% = initiation, 25-75% = fall line, last 25% = completion
        q1 = start + length // 4
        q3 = start + 3 * length // 4

        _add_phase(turns, TurnPhaseLabel.INITIATION, start, q1 - 1, direction)
        _add_phase(turns, TurnPhaseLabel.FALL_LINE, q1, q3 - 1, direction)
        _add_phase(turns, TurnPhaseLabel.COMPLETION, q3, end, direction)

        # Add transition after turn (if not last)
        if i < len(boundaries) - 2:
            next_start = boundaries[i + 1]
            next_end = boundaries[i + 2] - 1
            trans_len = next_end - next_start + 1
            if trans_len >= min_turn_frames // 2:
                _add_phase(turns, TurnPhaseLabel.TRANSITION, next_start, next_end, None)

    return turns


def _add_phase(
    turns: list[TurnPhase],
    label: TurnPhaseLabel,
    start: int,
    end: int,
    direction: str | None,
) -> None:
    if end >= start:
        turns.append(TurnPhase(label=label, start_frame=start, end_frame=end, direction=direction))
