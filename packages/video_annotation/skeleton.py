"""Skeleton drawing utilities — joints, bones, COM plumb line, metrics text."""

from __future__ import annotations

import cv2
import numpy as np

from core.schemas import Pose3D

# Bone connections (pairs of joint names)
BONES = [
    ("head", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("neck", "left_hip"),
    ("neck", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]

# Joint names for iteration
JOINT_NAMES = [
    "head", "neck",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def _confidence_color(conf: float) -> tuple[int, int, int]:
    """Return BGR color based on confidence: green >= 0.7, yellow 0.3-0.7, red < 0.3."""
    if conf >= 0.7:
        return (0, 255, 0)     # Green
    elif conf >= 0.3:
        return (0, 255, 255)   # Yellow
    else:
        return (0, 0, 255)     # Red


def _project_to_2d(
    point_3d: np.ndarray,
    image_size: tuple[int, int],
    cam_center: np.ndarray | None = None,
    scale: float | None = None,
) -> tuple[int, int]:
    """
    Simple orthographic projection from 3D to 2D pixel coordinates.

    Args:
        point_3d: 3D position (x, y, z) in Y-up system.
        image_size: (height, width) of the output image.
        cam_center: 3D center for the projection (default: auto).
        scale: Pixels per meter (default: auto).

    Returns:
        (px, py) pixel coordinates.
    """
    h, w = image_size
    # X maps to horizontal, Y maps to vertical (inverted)
    if scale is None:
        scale = min(w, h) * 0.4  # Rough default
    if cam_center is None:
        cam_center = np.array([0.0, 0.9, 0.0])  # Approximate body center

    px = int(w / 2 + (point_3d[0] - cam_center[0]) * scale)
    py = int(h / 2 - (point_3d[1] - cam_center[1]) * scale)
    return (px, py)


def draw_skeleton(
    frame: np.ndarray,
    pose: Pose3D,
    joint_radius: int = 5,
    bone_thickness: int = 2,
) -> np.ndarray:
    """
    Draw skeleton overlay on a frame — joint dots colored by confidence, bone lines.

    Args:
        frame: BGR image (H, W, 3), uint8. Modified in place.
        pose: 3D pose data.
        joint_radius: Radius of joint circles in pixels.
        bone_thickness: Thickness of bone lines in pixels.

    Returns:
        The annotated frame.
    """
    h, w = frame.shape[:2]
    image_size = (h, w)
    com = pose.com
    cam_center = com.copy()

    # Estimate pixels-per-meter from body height in 3D (head Y - ankle Y).
    # Fall back to a fixed default if geometry is degenerate.
    try:
        head_y = pose.to_np("head")[1]
        ankle_y = (pose.to_np("left_ankle")[1] + pose.to_np("right_ankle")[1]) / 2
        body_height_m = abs(head_y - ankle_y)
        if body_height_m > 0.2:
            # Target body height ≈ 40% of min(frame_dim) in pixels
            scale = (min(h, w) * 0.35) / body_height_m
        else:
            scale = min(w, h) * 0.4
    except Exception:
        scale = min(w, h) * 0.4

    # Determine 2D screen anchor: use anchor_px (hip midpoint) if available,
    # otherwise fall back to frame center.
    if pose.anchor_px is not None:
        anchor_x, anchor_y = float(pose.anchor_px[0]), float(pose.anchor_px[1])
    else:
        anchor_x, anchor_y = w / 2.0, h / 2.0

    # The hip midpoint in 3D projects to anchor_px in 2D.
    hip_3d = (pose.to_np("left_hip") + pose.to_np("right_hip")) / 2.0

    def project(point_3d: np.ndarray) -> tuple[int, int]:
        """Project 3D point to 2D pixel, anchored to skier's hip position.

        MotionBERT outputs in Y-down / camera convention (more negative Y = higher).
        Screen Y also increases downward, so dy maps directly: py = anchor_y + dy*scale.
        """
        dx = point_3d[0] - hip_3d[0]
        dy = point_3d[1] - hip_3d[1]
        px = int(anchor_x + dx * scale)
        py = int(anchor_y + dy * scale)  # Y-down (MotionBERT) → screen Y-down
        return (px, py)

    # Draw bones first (behind joints)
    for j1_name, j2_name in BONES:
        try:
            p1 = pose.to_np(j1_name)
            p2 = pose.to_np(j2_name)
        except ValueError:
            continue
        px1 = project(p1)
        px2 = project(p2)
        cv2.line(frame, px1, px2, (200, 200, 200), bone_thickness)

    # Draw joints
    for joint_name in JOINT_NAMES:
        try:
            pos = pose.to_np(joint_name)
        except ValueError:
            continue
        px = project(pos)

        conf = 0.5  # Default
        if pose.confidence and joint_name in pose.confidence:
            conf = pose.confidence[joint_name]

        color = _confidence_color(conf)
        cv2.circle(frame, px, joint_radius, color, -1)

    return frame


def draw_com_plumb_line(
    frame: np.ndarray,
    pose: Pose3D,
    line_length_m: float = 1.5,
    color: tuple[int, int, int] = (255, 165, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a vertical plumb line from the center of mass downward.

    Args:
        frame: BGR image (H, W, 3), uint8.
        pose: 3D pose data.
        line_length_m: Length of the plumb line in meters.
        color: BGR color for the line.
        thickness: Line thickness in pixels.

    Returns:
        The annotated frame.
    """
    h, w = frame.shape[:2]
    image_size = (h, w)
    com = pose.com

    # Use anchor_px for consistent positioning with draw_skeleton
    if pose.anchor_px is not None:
        anchor_x, anchor_y = float(pose.anchor_px[0]), float(pose.anchor_px[1])
    else:
        anchor_x, anchor_y = w / 2.0, h / 2.0

    hip_3d = (pose.to_np("left_hip") + pose.to_np("right_hip")) / 2.0
    try:
        head_y = pose.to_np("head")[1]
        ankle_y = (pose.to_np("left_ankle")[1] + pose.to_np("right_ankle")[1]) / 2
        body_height_m = abs(head_y - ankle_y)
        scale = (min(h, w) * 0.35) / body_height_m if body_height_m > 0.2 else min(w, h) * 0.4
    except Exception:
        scale = min(w, h) * 0.4

    def project(point_3d: np.ndarray) -> tuple[int, int]:
        dx = point_3d[0] - hip_3d[0]
        dy = point_3d[1] - hip_3d[1]
        return (int(anchor_x + dx * scale), int(anchor_y + dy * scale))

    com_px = project(com)
    ground_point = com.copy()
    ground_point[1] += line_length_m  # Y-down: positive = lower on screen
    ground_px = project(ground_point)

    cv2.line(frame, com_px, ground_px, color, thickness)
    return frame


def draw_metrics_text(
    frame: np.ndarray,
    pose: Pose3D,
    position: tuple[int, int] = (15, 30),
    font_scale: float = 0.6,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw biomechanical metrics text overlay on the frame.

    Displays: knee flex angle (L/R), inclination, COM height %.

    Args:
        frame: BGR image (H, W, 3), uint8.
        pose: 3D pose data.
        position: Top-left (x, y) for the text block.
        font_scale: Font size multiplier.
        color: BGR text color.
        thickness: Text thickness.

    Returns:
        The annotated frame.
    """
    from biomechanics.metrics import knee_flex_angle, inclination_angle, com_height_pct

    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position
    line_height = int(25 * font_scale / 0.6)

    # Compute metrics
    try:
        left_knee = knee_flex_angle(
            pose.to_np("left_hip"), pose.to_np("left_knee"), pose.to_np("left_ankle")
        )
        right_knee = knee_flex_angle(
            pose.to_np("right_hip"), pose.to_np("right_knee"), pose.to_np("right_ankle")
        )
        body_axis = pose.to_np("head") - pose.ankle_midpoint
        incl = inclination_angle(body_axis)
        com_h = com_height_pct(
            pose.com,
            pose.ankle_midpoint,
            np.array([0.0, 1.0, 0.0]),
            pose.body_height,
        )
    except Exception:
        return frame

    lines = [
        f"Knee L: {left_knee:.0f} deg  R: {right_knee:.0f} deg",
        f"Inclination: {incl:.1f} deg",
        f"COM Height: {com_h * 100:.0f}%",
    ]

    # Draw background rectangle for readability
    max_text_w = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    )
    bg_h = len(lines) * line_height + 10
    cv2.rectangle(
        frame,
        (x - 5, y - line_height),
        (x + max_text_w + 10, y + bg_h - line_height),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        frame,
        (x - 5, y - line_height),
        (x + max_text_w + 10, y + bg_h - line_height),
        (100, 100, 100),
        1,
    )

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * line_height), font, font_scale, color, thickness)

    return frame


def format_metrics(pose: Pose3D) -> dict[str, float]:
    """
    Compute and return metrics as a dict for display or JSON export.

    Returns dict with keys: left_knee_deg, right_knee_deg, inclination_deg, com_height_pct.
    """
    from biomechanics.metrics import knee_flex_angle, inclination_angle, com_height_pct

    left_knee = knee_flex_angle(
        pose.to_np("left_hip"), pose.to_np("left_knee"), pose.to_np("left_ankle")
    )
    right_knee = knee_flex_angle(
        pose.to_np("right_hip"), pose.to_np("right_knee"), pose.to_np("right_ankle")
    )
    body_axis = pose.to_np("head") - pose.ankle_midpoint
    incl = inclination_angle(body_axis)
    com_h = com_height_pct(
        pose.com,
        pose.ankle_midpoint,
        np.array([0.0, 1.0, 0.0]),
        pose.body_height,
    )

    return {
        "left_knee_deg": round(left_knee, 1),
        "right_knee_deg": round(right_knee, 1),
        "inclination_deg": round(incl, 1),
        "com_height_pct": round(com_h, 3),
    }
