"""Tests for video annotation — skeleton drawing, metrics text, renderer."""

from __future__ import annotations

import numpy as np
import pytest

from core.schemas import Pose3D
from video_annotation.skeleton import (
    _confidence_color,
    draw_com_plumb_line,
    draw_metrics_text,
    draw_skeleton,
    format_metrics,
)
from video_annotation.renderer import annotate_frames


# ── Helpers ──────────────────────────────────────────────────

def _make_pose(**overrides) -> Pose3D:
    defaults = {
        "head": [0, 1.8, 0],
        "neck": [0, 1.6, 0],
        "left_shoulder": [-0.2, 1.5, 0],
        "right_shoulder": [0.2, 1.5, 0],
        "left_elbow": [-0.3, 1.2, 0],
        "right_elbow": [0.3, 1.2, 0],
        "left_wrist": [-0.3, 1.0, 0],
        "right_wrist": [0.3, 1.0, 0],
        "left_hip": [-0.1, 1.0, 0],
        "right_hip": [0.1, 1.0, 0],
        "left_knee": [-0.1, 0.5, 0],
        "right_knee": [0.1, 0.5, 0],
        "left_ankle": [-0.1, 0.0, 0],
        "right_ankle": [0.1, 0.0, 0],
    }
    defaults.update(overrides)
    return Pose3D(**defaults)


def _make_frame(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── Confidence colors ────────────────────────────────────────

class TestConfidenceColor:
    def test_high_confidence_green(self):
        assert _confidence_color(0.9) == (0, 255, 0)

    def test_medium_confidence_yellow(self):
        assert _confidence_color(0.5) == (0, 255, 255)

    def test_low_confidence_red(self):
        assert _confidence_color(0.1) == (0, 0, 255)

    def test_boundary_0_7(self):
        assert _confidence_color(0.7) == (0, 255, 0)

    def test_boundary_0_3(self):
        assert _confidence_color(0.3) == (0, 255, 255)


# ── Skeleton drawing ─────────────────────────────────────────

class TestDrawSkeleton:
    def test_modifies_frame(self):
        frame = _make_frame()
        pose = _make_pose()
        result = draw_skeleton(frame, pose)
        # Frame should have non-zero pixels (skeleton drawn)
        assert result.sum() > 0

    def test_with_confidence(self):
        pose = _make_pose(
            confidence={"head": 0.95, "left_ankle": 0.2}
        )
        frame = _make_frame()
        result = draw_skeleton(frame, pose)
        assert result.sum() > 0


# ── COM plumb line ───────────────────────────────────────────

class TestDrawComPlumbLine:
    def test_draws_line(self):
        frame = _make_frame()
        pose = _make_pose()
        result = draw_com_plumb_line(frame, pose)
        assert result.sum() > 0


# ── Metrics text ─────────────────────────────────────────────

class TestDrawMetricsText:
    def test_draws_text(self):
        frame = _make_frame()
        pose = _make_pose()
        result = draw_metrics_text(frame, pose)
        assert result.sum() > 0


class TestFormatMetrics:
    def test_returns_expected_keys(self):
        pose = _make_pose()
        metrics = format_metrics(pose)
        assert "left_knee_deg" in metrics
        assert "right_knee_deg" in metrics
        assert "inclination_deg" in metrics
        assert "com_height_pct" in metrics

    def test_knee_angles_reasonable(self):
        pose = _make_pose()
        metrics = format_metrics(pose)
        assert 0 <= metrics["left_knee_deg"] <= 180
        assert 0 <= metrics["right_knee_deg"] <= 180


# ── Renderer ─────────────────────────────────────────────────

class TestAnnotateFrames:
    def test_annotates_multiple_frames(self):
        frames = [_make_frame() for _ in range(3)]
        poses = [_make_pose() for _ in range(3)]
        result = annotate_frames(frames, poses)
        assert len(result) == 3
        for frame in result:
            assert frame.sum() > 0

    def test_preserves_originals(self):
        frames = [_make_frame()]
        poses = [_make_pose()]
        original_sum = frames[0].sum()
        annotate_frames(frames, poses)
        assert frames[0].sum() == original_sum

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            annotate_frames([_make_frame()], [_make_pose(), _make_pose()])
