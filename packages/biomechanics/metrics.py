"""
Biomechanical metric computations for skiing and snowboarding.

All inputs are numpy arrays. All outputs in degrees (angles) or meters (distances)
unless explicitly stated otherwise.

Coordinate system: Y-up, right-hand (X=right, Y=up, Z=toward camera).
Snow surface normal ≈ [0, 1, 0] on flat terrain.
"""

from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

_UP = np.array([0.0, 1.0, 0.0])
_EPS = 1e-8


def _unit(v: np.ndarray) -> np.ndarray:
    """Return unit vector; returns zero vector if magnitude < EPS."""
    n = np.linalg.norm(v)
    return v / n if n >= _EPS else np.zeros_like(v)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle in degrees between two vectors (0–180°)."""
    cos = np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _signed_angle_around_axis(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """
    Signed angle in degrees from vector *a* to vector *b* around *axis*.
    Positive = counter-clockwise when viewed from tip of axis.
    """
    cross = np.cross(a, b)
    sin = np.dot(cross, _unit(axis))
    cos = np.dot(a, b)
    return float(np.degrees(np.arctan2(sin, cos)))


def _plane_normal(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Unit normal of the plane defined by three points."""
    return _unit(np.cross(p2 - p1, p3 - p1))


# ──────────────────────────────────────────────────────────────
# Skiing metrics (CSIA Five Skills)
# ──────────────────────────────────────────────────────────────

def edge_angle(ski_plane_normal: np.ndarray, snow_surface_normal: np.ndarray) -> float:
    """
    Edging skill — angle between the ski's base plane normal and the snow surface.

    A flat ski has edge_angle ≈ 0°; full edge engagement ≈ 60–90° for expert turns.

    Args:
        ski_plane_normal:    Unit normal of the ski base plane, shape (3,).
        snow_surface_normal: Unit normal of the snow surface, shape (3,).

    Returns:
        Edge angle in degrees [0°, 90°].

    Example::
        >>> edge_angle(np.array([0,1,0]), np.array([0,1,0]))
        0.0   # flat ski
        >>> edge_angle(np.array([1,0,0]), np.array([0,1,0]))
        90.0  # fully on edge
    """
    raw = _angle_between(ski_plane_normal, snow_surface_normal)
    # Normalise to [0, 90]: beyond 90° means the normal flipped
    return min(raw, 180.0 - raw)


def inclination_angle(body_axis: np.ndarray, vertical: np.ndarray | None = None) -> float:
    """
    Balance skill — lean of the whole body relative to vertical.

    Computed as angle between the body's long axis and the vertical (Y-up).
    Expert carvers often reach 30–45° inclination in aggressive turns.

    Args:
        body_axis: Vector from ankle midpoint to head, shape (3,).
        vertical:  Reference vertical, defaults to [0, 1, 0].

    Returns:
        Inclination angle in degrees [0°, 90°].

    Example::
        >>> inclination_angle(np.array([0,1,0]))
        0.0   # perfectly upright
        >>> inclination_angle(np.array([1,0,0]))
        90.0  # horizontal
    """
    if vertical is None:
        vertical = _UP
    return _angle_between(body_axis, vertical)


def angulation(
    upper_body_plane_normal: np.ndarray,
    lower_body_plane_normal: np.ndarray,
) -> float:
    """
    Edging + Balance skill — lateral separation between upper and lower body planes.

    Angulation is the sideways bend at the hip/waist that allows the upper body
    to remain more upright while the lower body carves. Expert skiers exhibit
    15–35° of angulation in carved turns.

    Args:
        upper_body_plane_normal: Normal of the plane (head, L-shoulder, R-shoulder).
        lower_body_plane_normal: Normal of the plane (L-hip, R-hip, midpoint).

    Returns:
        Angulation angle in degrees [0°, 90°].
    """
    return edge_angle(upper_body_plane_normal, lower_body_plane_normal)


def knee_flex_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    """
    Pressure skill — flexion angle at the knee joint.

    Full extension = 180°; squatting = ~90°; athletic ski stance = 130–160°.

    Args:
        hip:   Hip joint position, shape (3,).
        knee:  Knee joint position, shape (3,).
        ankle: Ankle joint position, shape (3,).

    Returns:
        Knee flexion angle in degrees [0°, 180°].

    Example::
        >>> knee_flex_angle(np.array([0,2,0]), np.array([0,1,0]), np.array([0,0,0]))
        180.0  # straight leg
    """
    thigh = hip - knee
    shin = ankle - knee
    return _angle_between(thigh, shin)


def hip_flex_angle(spine_base: np.ndarray, hip: np.ndarray, knee: np.ndarray) -> float:
    """
    Pressure skill — flexion angle at the hip joint.

    Measured as angle between the torso vector and thigh vector.
    Upright posture = ~180°; forward lean = <180°; typical ski stance ≈ 150–170°.

    Args:
        spine_base: Lower spine / pelvis point, shape (3,).
        hip:        Hip joint position, shape (3,).
        knee:       Knee joint position, shape (3,).

    Returns:
        Hip flexion angle in degrees [0°, 180°].
    """
    torso = spine_base - hip
    thigh = knee - hip
    return _angle_between(torso, thigh)


def fore_aft_balance(
    com: np.ndarray,
    boot_midpoint: np.ndarray,
    ski_axis: np.ndarray,
    ski_length: float = 1.7,
) -> float:
    """
    Balance skill — fore-aft COM position relative to boot midpoint along ski axis.

    Returns a value normalised by half the ski length:
      +1.0 = COM fully over ski tip  (forward)
       0.0 = COM directly over boot
      -1.0 = COM fully over ski tail (back seat)

    Args:
        com:          Centre of mass, shape (3,).
        boot_midpoint: Midpoint between boot bindings, shape (3,).
        ski_axis:     Unit vector along ski from tail to tip, shape (3,).
        ski_length:   Approximate ski length in metres (default 1.7 m).

    Returns:
        Fore-aft balance ratio in [-1, 1].
    """
    projection = np.dot(com - boot_midpoint, _unit(ski_axis))
    return float(np.clip(projection / (ski_length / 2), -1.0, 1.0))


def lateral_balance(
    com: np.ndarray,
    boot_midpoint: np.ndarray,
    ski_perp_axis: np.ndarray,
    ski_width: float = 0.1,
) -> float:
    """
    Balance skill — lateral COM position relative to boot midpoint.

    Returns a value normalised by half the ski width:
      +1.0 = COM toward inside of turn  (uphill edge)
      -1.0 = COM toward outside of turn (downhill edge)

    Args:
        com:            Centre of mass, shape (3,).
        boot_midpoint:  Midpoint between boot bindings, shape (3,).
        ski_perp_axis:  Unit vector perpendicular to ski axis (XZ plane), shape (3,).
        ski_width:      Approximate ski waist width in metres (default 0.1 m).

    Returns:
        Lateral balance ratio in [-1, 1].
    """
    projection = np.dot(com - boot_midpoint, _unit(ski_perp_axis))
    return float(np.clip(projection / (ski_width / 2), -1.0, 1.0))


def upper_lower_separation(
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
    left_hip: np.ndarray,
    right_hip: np.ndarray,
) -> float:
    """
    Rotary skill — rotation difference between the shoulder plane and hip plane,
    projected onto the vertical (Y) axis. Positive = shoulders rotated toward
    the right; negative = shoulders rotated toward the left.

    Expert skiers use upper/lower body separation (counter-rotation) to maintain
    balance and control in challenging terrain. Typical range: ±20–40°.

    Args:
        left_shoulder:  Left shoulder position, shape (3,).
        right_shoulder: Right shoulder position, shape (3,).
        left_hip:       Left hip position, shape (3,).
        right_hip:      Right hip position, shape (3,).

    Returns:
        Separation angle in degrees [-180°, 180°].
    """
    shoulder_vec = right_shoulder - left_shoulder
    hip_vec = right_hip - left_hip
    # Project onto horizontal plane
    shoulder_horiz = np.array([shoulder_vec[0], 0.0, shoulder_vec[2]])
    hip_horiz = np.array([hip_vec[0], 0.0, hip_vec[2]])
    return _signed_angle_around_axis(hip_horiz, shoulder_horiz, _UP)


def com_height_pct(
    com: np.ndarray,
    snow_surface_point: np.ndarray,
    snow_surface_normal: np.ndarray,
    body_height: float,
) -> float:
    """
    Pressure skill — height of the COM above the snow surface, normalised by body height.

    1.0 = COM at full standing height above snow
    0.0 = COM at snow level (fully compressed)
    Typical skiing: 0.50–0.75.

    Args:
        com:                  Centre of mass position, shape (3,).
        snow_surface_point:   Any point on the snow surface, shape (3,).
        snow_surface_normal:  Unit normal of snow surface, shape (3,).
        body_height:          Athlete body height in metres.

    Returns:
        COM height fraction in [0, 1] (clamped).
    """
    if body_height < _EPS:
        return 0.0
    height_above = np.dot(com - snow_surface_point, _unit(snow_surface_normal))
    return float(np.clip(height_above / body_height, 0.0, 1.0))


def turn_radius_estimate(com_trajectory: np.ndarray) -> float:
    """
    Coordination skill — estimate turn radius from COM trajectory using a
    circle fit to the XZ (horizontal) plane.

    Uses the Menger curvature formula on consecutive triplets of points and
    returns the median radius as a robust estimate.

    Args:
        com_trajectory: COM positions over time, shape (N, 3). N >= 3 required.

    Returns:
        Estimated turn radius in metres. Returns np.inf for a straight line.

    Example::
        >>> pts = np.array([[np.cos(t), 0, np.sin(t)] for t in np.linspace(0, np.pi, 20)])
        >>> abs(turn_radius_estimate(pts) - 1.0) < 0.05
        True
    """
    if len(com_trajectory) < 3:
        return float("inf")

    traj = com_trajectory[:, [0, 2]]  # XZ plane
    radii = []
    for i in range(len(traj) - 2):
        a, b, c = traj[i], traj[i + 1], traj[i + 2]
        # Menger curvature: κ = 4·Area / (|ab|·|bc|·|ca|)
        # Extend to 3D for np.cross compatibility (numpy 2.0+)
        def _to3d(v2: np.ndarray) -> np.ndarray:
            return np.array([v2[0], v2[1], 0.0])
        area = abs(np.cross(_to3d(b - a), _to3d(c - a))[2]) / 2.0
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ca = np.linalg.norm(a - c)
        denom = ab * bc * ca
        if denom < _EPS or area < _EPS:
            continue  # colinear / degenerate
        radii.append(denom / (4.0 * area))

    return float(np.median(radii)) if radii else float("inf")


def speed_estimate(com_trajectory: np.ndarray, fps: float) -> float:
    """
    Estimate average speed from COM trajectory.

    Args:
        com_trajectory: COM positions over time, shape (N, 3).
        fps:            Frames per second of the source video.

    Returns:
        Average speed in m/s. Returns 0.0 if fewer than 2 frames.
    """
    if len(com_trajectory) < 2 or fps <= 0:
        return 0.0
    displacements = np.linalg.norm(np.diff(com_trajectory, axis=0), axis=1)
    return float(np.mean(displacements) * fps)


# ──────────────────────────────────────────────────────────────
# Snowboard metrics (CASI framework)
# ──────────────────────────────────────────────────────────────

def board_tilt_angle(board_plane_normal: np.ndarray, snow_surface_normal: np.ndarray) -> float:
    """
    Edging skill (snowboard) — tilt of the board relative to the snow surface.

    Equivalent to edge_angle for skiing; 0° = flat board, 90° = board on rail.

    Args:
        board_plane_normal:  Unit normal of the snowboard base plane, shape (3,).
        snow_surface_normal: Unit normal of the snow surface, shape (3,).

    Returns:
        Board tilt angle in degrees [0°, 90°].
    """
    return edge_angle(board_plane_normal, snow_surface_normal)


def fore_aft_weight_distribution(
    com: np.ndarray,
    board_center: np.ndarray,
    board_axis: np.ndarray,
    board_length: float = 1.55,
) -> float:
    """
    Balance skill (snowboard) — fore-aft weight distribution along the board.

    +1.0 = all weight over nose, -1.0 = all weight over tail.
    Neutral riding: near 0.0 to +0.1 (slightly nose-weighted).

    Args:
        com:          Centre of mass, shape (3,).
        board_center: Centre of the snowboard, shape (3,).
        board_axis:   Unit vector from tail to nose, shape (3,).
        board_length: Board length in metres (default 1.55 m).

    Returns:
        Weight distribution ratio in [-1, 1].
    """
    return fore_aft_balance(com, board_center, board_axis, ski_length=board_length)


def counter_rotation(
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
    board_heading: np.ndarray,
) -> float:
    """
    Rotary skill (snowboard) — angle between the upper body orientation and
    the board heading, projected onto the horizontal plane.

    Positive = upper body rotated more toward the nose (open stance).
    Negative = upper body counter-rotated behind board heading (closed stance).

    Skilled riders maintain slight counter-rotation (+5° to +20°) on toeside
    turns and moderate counter-rotation on heelside turns.

    Args:
        left_shoulder:  Left shoulder position, shape (3,).
        right_shoulder: Right shoulder position, shape (3,).
        board_heading:  Unit vector from board tail to nose in XZ plane, shape (3,).

    Returns:
        Counter-rotation angle in degrees [-180°, 180°].
    """
    shoulder_vec = right_shoulder - left_shoulder
    shoulder_horiz = np.array([shoulder_vec[0], 0.0, shoulder_vec[2]])
    board_horiz = np.array([board_heading[0], 0.0, board_heading[2]])
    return _signed_angle_around_axis(board_horiz, shoulder_horiz, _UP)
