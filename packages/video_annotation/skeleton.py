"""Skeleton drawing utilities — joints, bones, COM plumb line, metrics text."""

from __future__ import annotations

import cv2
import numpy as np

from core.schemas import Pose3D
from pose_estimation.interfaces import Keypoints2D

# COCO keypoint indices (17 joints):
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

# Bone connections using COCO indices (skip face joints for clean skeleton)
COCO_BONES = [
    (0, 5),    # nose → left_shoulder (approximate head-neck)
    (0, 6),    # nose → right_shoulder
    (5, 6),    # left_shoulder → right_shoulder
    (5, 7),    # left_shoulder → left_elbow
    (7, 9),    # left_elbow → left_wrist
    (6, 8),    # right_shoulder → right_elbow
    (8, 10),   # right_elbow → right_wrist
    (5, 11),   # left_shoulder → left_hip
    (6, 12),   # right_shoulder → right_hip
    (11, 12),  # left_hip → right_hip
    (11, 13),  # left_hip → left_knee
    (13, 15),  # left_knee → left_ankle
    (12, 14),  # right_hip → right_knee
    (14, 16),  # right_knee → right_ankle
]

# COCO joint indices to draw (skip eyes/ears for cleaner overlay)
COCO_DRAW_JOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Bone connections for Pose3D (pairs of joint names) — used by COM plumb line
BONES_3D = [
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

# Joint names for Pose3D iteration
JOINT_NAMES = [
    "head", "neck",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Confidence threshold below which joints/bones are not drawn
MIN_CONFIDENCE = 0.3


def _confidence_color(conf: float) -> tuple[int, int, int]:
    """Return BGR color based on confidence: green >= 0.7, yellow 0.3-0.7, red < 0.3."""
    if conf >= 0.7:
        return (0, 255, 0)     # Green
    elif conf >= 0.3:
        return (0, 255, 255)   # Yellow
    else:
        return (0, 0, 255)     # Red


def draw_skeleton(
    frame: np.ndarray,
    keypoints: Keypoints2D,
    joint_radius: int = 5,
    bone_thickness: int = 2,
) -> np.ndarray:
    """
    Draw skeleton overlay using original 2D keypoints from ViTPose.

    Uses the pixel-accurate 2D detections directly instead of projecting
    3D poses back to 2D, which avoids alignment errors from the roundtrip.

    Args:
        frame: BGR image (H, W, 3), uint8. Modified in place.
        keypoints: Original 2D keypoint detections (COCO 17-joint format).
        joint_radius: Radius of joint circles in pixels.
        bone_thickness: Thickness of bone lines in pixels.

    Returns:
        The annotated frame.
    """
    points = keypoints.points    # (17, 2) as (x, y) pixels
    conf = keypoints.confidence  # (17,)

    # Draw bones first (behind joints)
    for j1_idx, j2_idx in COCO_BONES:
        c1, c2 = conf[j1_idx], conf[j2_idx]
        if c1 < MIN_CONFIDENCE or c2 < MIN_CONFIDENCE:
            continue
        p1 = (int(points[j1_idx, 0]), int(points[j1_idx, 1]))
        p2 = (int(points[j2_idx, 0]), int(points[j2_idx, 1]))
        # Bone color: average confidence of endpoints
        bone_color = _confidence_color((c1 + c2) / 2)
        cv2.line(frame, p1, p2, bone_color, bone_thickness, cv2.LINE_AA)

    # Draw joints
    for j_idx in COCO_DRAW_JOINTS:
        c = conf[j_idx]
        if c < MIN_CONFIDENCE:
            continue
        px = (int(points[j_idx, 0]), int(points[j_idx, 1]))
        color = _confidence_color(c)
        cv2.circle(frame, px, joint_radius, color, -1, cv2.LINE_AA)

    return frame


def draw_com_plumb_line(
    frame: np.ndarray,
    keypoints: Keypoints2D,
    pose: Pose3D,
    line_length_px: int = 200,
    color: tuple[int, int, int] = (255, 165, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a vertical plumb line from the 2D hip midpoint (COM proxy) downward.

    Uses the original 2D keypoints for positioning to stay aligned with the
    skeleton overlay. The line extends straight down from the hip midpoint.

    Args:
        frame: BGR image (H, W, 3), uint8.
        keypoints: Original 2D keypoint detections (COCO 17-joint format).
        pose: 3D pose data (used for COM offset if needed in future).
        line_length_px: Length of the plumb line in pixels.
        color: BGR color for the line.
        thickness: Line thickness in pixels.

    Returns:
        The annotated frame.
    """
    points = keypoints.points
    conf = keypoints.confidence

    # Hip midpoint in 2D (COCO indices 11, 12)
    if conf[11] < MIN_CONFIDENCE or conf[12] < MIN_CONFIDENCE:
        return frame

    hip_px = (points[11] + points[12]) / 2.0
    start = (int(hip_px[0]), int(hip_px[1]))
    end = (int(hip_px[0]), int(hip_px[1]) + line_length_px)

    cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)
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
