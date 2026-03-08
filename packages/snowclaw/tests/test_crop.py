"""Tests for the crop subcommand (person tracking + crop)."""

from __future__ import annotations

import argparse
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from snowclaw.crop import (
    _build_continuous_detections,
    _build_crop_trajectory,
    _render_crop_opencv,
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
        np.testing.assert_allclose(out, 5.0, atol=1e-6)

    def test_no_edge_drift(self):
        """Edge values should not drift toward zero (regression for zero-pad bug)."""
        arr = np.full(60, 400.0)   # simulates a crop centre held at x=400
        out = _smooth(arr, window=31)
        # All values — including the first and last 15 frames — must stay near 400
        assert out[0] > 390, f"First frame drifted to {out[0]:.1f} (zero-pad edge bug)"
        assert out[-1] > 390, f"Last frame drifted to {out[-1]:.1f} (zero-pad edge bug)"


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


class TestBuildContinuousDetections:
    def test_stitches_fragmented_track_ids(self):
        # Same person fragmented into 3 IDs over time
        tracks = {
            1: [(0, 0, 0, 10, 10), (1, 1, 0, 11, 10)],
            2: [(2, 2, 0, 12, 10), (3, 3, 0, 13, 10)],
            3: [(4, 4, 0, 14, 10)],
        }
        dets = _build_continuous_detections(tracks, total_frames=5, seed_track_id=2)
        frames = [d[0] for d in dets]
        assert frames == [0, 1, 2, 3, 4]


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

    def test_crop_preserves_input_aspect_ratio(self):
        dets = self._make_detections(n=20, bw=120, bh=200)
        _, _, cw, ch = _build_crop_trajectory(
            dets, total_frames=20, video_w=1920, video_h=1080
        )
        in_ar = 1920 / 1080
        out_ar = cw / ch
        assert abs(out_ar - in_ar) < 0.03

    def test_centre_interpolated_to_all_frames(self):
        # Only every 5th frame has a detection; result should cover all 100 frames
        dets = self._make_detections(n=20)
        dets_sparse = [dets[i] for i in range(0, 20, 5)]
        cx_arr, cy_arr, _, _ = _build_crop_trajectory(
            dets_sparse, total_frames=20, video_w=848, video_h=476
        )
        assert len(cx_arr) == 20
        assert not np.any(np.isnan(cx_arr))

    def test_tiny_input_clamps_crop_to_video(self):
        dets = self._make_detections(n=5, bw=20, bh=20)
        _, _, cw, ch = _build_crop_trajectory(
            dets, total_frames=5, video_w=63, video_h=47
        )
        assert cw <= 63
        assert ch <= 47
        assert cw > 0
        assert ch > 0


class TestRenderCropOpenCV:
    def test_raises_runtime_error_with_ffmpeg_stderr_and_cleans_up(self, tmp_path):
        released = {"value": False}
        stdin_closed = {"value": False}

        class FakeVideoCapture:
            def __init__(self, _path):
                self._reads = 0

            def read(self):
                if self._reads == 0:
                    self._reads += 1
                    return True, np.zeros((100, 100, 3), dtype=np.uint8)
                return False, None

            def release(self):
                released["value"] = True

        class FakeStdin:
            def __init__(self):
                self.closed = False

            def write(self, _data):
                raise BrokenPipeError("ffmpeg died")

            def close(self):
                self.closed = True
                stdin_closed["value"] = True

        fake_stdin = FakeStdin()
        fake_proc = MagicMock()
        fake_proc.stdin = fake_stdin
        fake_proc.wait.return_value = 1
        fake_proc.stderr.read.return_value = "encoder unavailable"

        fake_cv2 = types.SimpleNamespace(
            VideoCapture=FakeVideoCapture,
            resize=lambda frame, size, interpolation=None: frame,
            INTER_LANCZOS4=1,
        )

        with patch("snowclaw.crop.subprocess.Popen", return_value=fake_proc):
            with patch.dict("sys.modules", {"cv2": fake_cv2}):
                with pytest.raises(RuntimeError, match="encoder unavailable"):
                    _render_crop_opencv(
                        video_path=tmp_path / "in.mp4",
                        cx_arr=np.array([50.0]),
                        cy_arr=np.array([50.0]),
                        crop_w=20,
                        crop_h=20,
                        video_w=100,
                        video_h=100,
                        output_path=tmp_path / "out.mp4",
                        out_w=10,
                        out_h=10,
                        fps=30.0,
                    )

        assert released["value"] is True
        assert stdin_closed["value"] is True
        assert fake_proc.wait.called


# ---------------------------------------------------------------------------
# CLI integration test — uses the sample video + mock tracking
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_crop_end_to_end_mock(tmp_path):
    """Crop pipeline runs end-to-end with a mocked YOLO tracker."""
    sample = Path(__file__).parents[3] / "data" / "samples" / "ski_demo.mp4"
    if not sample.exists():
        pytest.skip("sample video not present")

    output = tmp_path / "out_crop.mp4"

    args = argparse.Namespace(
        video=str(sample),
        output=str(output),
        track_id=None,
        padding=0.4,
        smooth=15,
        out_width=848,
        out_height=476,
    )

    with patch("snowclaw.crop._track_persons") as mock_track:
        mock_track.return_value = (
            {1: [(i, 100.0, 80.0, 300.0, 380.0) for i in range(30)]},
            30,
        )
        from snowclaw.crop import run_crop
        rc = run_crop(args)

    assert rc == 0
    assert output.exists()


def test_run_crop_warns_for_frame_count_mismatch(tmp_path):
    from snowclaw.crop import run_crop

    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"placeholder")
    output = tmp_path / "output.mp4"

    fake_cap = MagicMock()
    fake_cap.get.side_effect = [640, 360, 10, 30.0]
    args = argparse.Namespace(
        video=str(video_path),
        output=str(output),
        track_id=None,
        padding=0.4,
        smooth=15,
        out_width=320,
        out_height=180,
    )

    with patch("cv2.VideoCapture", return_value=fake_cap):
        with patch("snowclaw.crop._track_persons", return_value=({1: [(i, 0, 0, 30, 30) for i in range(12)]}, 12)):
            with patch("snowclaw.crop._render_crop_opencv") as mock_render:
                with pytest.warns(RuntimeWarning, match="Frame count mismatch"):
                    rc = run_crop(args)

    assert rc == 0
    assert mock_render.called
