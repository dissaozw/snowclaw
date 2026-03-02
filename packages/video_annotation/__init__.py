"""
SnowClaw Video Annotation — skeleton overlay and metrics rendering.

Draws skeleton, COM plumb line, and metric text onto video frames.
"""

from .renderer import annotate_frames, annotate_video
from .skeleton import draw_skeleton, draw_com_plumb_line, draw_metrics_text

__all__ = [
    "annotate_frames",
    "annotate_video",
    "draw_skeleton",
    "draw_com_plumb_line",
    "draw_metrics_text",
]
