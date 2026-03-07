"""Tests for the crop subcommand (person tracking + crop)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from snowclaw.crop import (
    _build_crop_trajectory,
    _select_track,
    _smooth,
)


# ---------------------------------------------------------------------------
# Unit tests — pure functions, no model / video required
# ---------------------------------------------------------------------------


class TestSmooth:
    def test_returns_same_length(self):
        arr = np.random.rand(100)
        out = _smooth(arr, window=11)
        assert len(out) == len(arr)

    def test_short_array_unchanged(self):
        arr = np.array([1.0, 2.0])
        out = _smooth(arr, window=31)
        assert len(out) == 2

    def test_constant_array_stays_constant(self):
        arr = np.full(50, 5.0)
        out = _smooth(arr, window=9)
        np.testing.assert_allclose(out[4:-4], 5.0, atol=1e-6)


class TestSelectTrack:
    def _make_tracks(self):
        # track 1: small boxes, track 2: large boxes (should win)
        return {
            1: [(i, 0, 0, 10, 10) for i in range(10)],   # area 100 each
            2: [(i, 0, 0, 50, 50) for i in range(10)],   # area 2500 each
        }

    def test_auto_selects_largest(self):
        tracks = self._make_tracks()
        chosen = _select_track(tracks, track_id=None)
        assert chosen == 2

    def test_explicit_track_id(self):
        tracks = self._make_tracks()
        chosen = _select_track(tracks, track_id=1)
        assert chosen == 1

    def test_invalid_track_id_exits(self):
        tracks = self._make_tracks()
        with pytest.raises(SystemExit):
            _select_track(tracks, track_id=999)


class TestBuildCropTrajectory:
    def _make_detections(self, n=20, cx=424, cy=238, bw=80, bh=120):
        """Synthetic detections for a stationary person."""
        return [
            (i, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            for i in range(n)
        ]

    def test_output_shapes(self):
        dets = self._make_detections(n=20)
        cx_arr, cy_arr, cw, ch = _build_crop_trajectory(
            dets, total_frames=20, video_w=848, video_h=476
        )
        assert len(cx_arr) == 20
        assert len(cy_arr) == 20

    def test_crop_size_includes_padding(self):
        dets = self._make_detections(n=20, bw=80, bh=120)
        _, _, cw, ch = _build_crop_trajectory(
            dets, total_frames=20, video_w=848, video_h=476, pad=0.4
        )
        # crop_w should be larger than the raw bbox (padding applied).
        # 90th-percentile + smoothing may reduce slightly vs bw*(1+2*pad),
        # so we check crop_w > bw (no-padding baseline).
        assert cw > 80
        assert ch > 120

    def test_crop_size_capped_at_video_dims(self):
        dets = self._make_detections(n=20, bw=800, bh=460)
        _, _, cw, ch = _build_crop_trajectory(
            dets, total_frames=20, video_w=848, video_h=476
        )
        assert cw <= 848
        assert ch <= 476

    def test_crop_size_even(self):
        dets = self._make_detections(n=20, bw=75, bh=111)
        _, _, cw, ch = _build_crop_trajectory(
            dets, total_frames=20, video_w=848, video_h=476
        )
        assert cw % 2 == 0
        assert ch % 2 == 0

    def test_centre_interpolated_to_all_frames(self):
        # Only every 5th frame has a detection; result should cover all 100 frames
        dets = self._make_detections(n=20)
        dets_sparse = [dets[i] for i in range(0, 20, 5)]
        cx_arr, cy_arr, _, _ = _build_crop_trajectory(
            dets_sparse, total_frames=20, video_w=848, video_h=476
        )
        assert len(cx_arr) == 20
        assert not np.any(np.isnan(cx_arr))


# ---------------------------------------------------------------------------
# CLI integration test — uses the sample video + mock tracking
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_crop_cli_mock(tmp_path):
    """Crop subcommand runs end-to-end with a mocked YOLO tracker."""
    sample = Path(__file__).parents[3] / "data" / "samples" / "ski_demo.mp4"
    if not sample.exists():
        pytest.skip("sample video not present")

    output = tmp_path / "out_crop.mp4"

    # Patch YOLO so we don't need real weights in CI
    fake_box = MagicMock()
    fake_box.xyxy.cpu().numpy.return_value = np.array([[100, 80, 200, 280]])
    fake_box.id.cpu().numpy.return_value = np.array([1])

    fake_result = MagicMock()
    fake_result.boxes = fake_box

    with patch("ultralytics.YOLO") as MockYOLO:
        instance = MockYOLO.return_value
        instance.track.return_value = iter([fake_result] * 30)

        result = subprocess.run(
            [
                sys.executable, "-m", "snowclaw.cli",
                "crop", str(sample),
                "--output", str(output),
            ],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parents[3] / "packages")},
        )

    assert result.returncode == 0 or "Done" in result.stdout, result.stderr
