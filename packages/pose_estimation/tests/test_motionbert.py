"""Tests for MotionBERT backend with mock PyTorch model."""

from __future__ import annotations

import numpy as np
import pytest

from core.schemas import Pose3D
from pose_estimation.interfaces import Keypoints2D
from pose_estimation.motionbert_backend import (
    MOTIONBERT_NUM_JOINTS,
    MOTIONBERT_WINDOW_SIZE,
    MotionBERTBackend,
    _apply_temporal_smoothing,
    _assemble_temporal_window,
    _normalize_keypoints_2d,
)


# ── Normalization ────────────────────────────────────────────

class TestNormalizeKeypoints2D:
    def test_center_of_image_maps_to_zero(self):
        kp = Keypoints2D(
            points=np.array([[320.0, 240.0]], dtype=np.float32),
            confidence=np.array([1.0]),
            image_size=(480, 640),
        )
        norm = _normalize_keypoints_2d(kp)
        assert norm[0, 0] == pytest.approx(0.0, abs=0.01)
        assert norm[0, 1] == pytest.approx(0.0, abs=0.01)

    def test_corners_map_to_extremes(self):
        kp = Keypoints2D(
            points=np.array([[0.0, 0.0], [640.0, 480.0]], dtype=np.float32),
            confidence=np.array([1.0, 1.0]),
            image_size=(480, 640),
        )
        norm = _normalize_keypoints_2d(kp)
        assert norm[0, 0] == pytest.approx(-1.0, abs=0.01)
        assert norm[0, 1] == pytest.approx(-1.0, abs=0.01)
        assert norm[1, 0] == pytest.approx(1.0, abs=0.01)
        assert norm[1, 1] == pytest.approx(1.0, abs=0.01)


# ── Temporal window assembly ─────────────────────────────────

class TestAssembleTemporalWindow:
    def test_basic_window(self):
        kps = [np.ones((17, 2)) * i for i in range(30)]
        window = _assemble_temporal_window(kps, window_size=5, center_idx=15)
        assert window.shape == (5, 17, 2)
        # Center should be frame 15
        assert window[2, 0, 0] == pytest.approx(15.0)

    def test_edge_padding_start(self):
        kps = [np.ones((17, 2)) * i for i in range(10)]
        window = _assemble_temporal_window(kps, window_size=5, center_idx=0)
        assert window.shape == (5, 17, 2)
        # First frames should be padded with frame 0
        assert window[0, 0, 0] == pytest.approx(0.0)
        assert window[1, 0, 0] == pytest.approx(0.0)

    def test_edge_padding_end(self):
        kps = [np.ones((17, 2)) * i for i in range(10)]
        window = _assemble_temporal_window(kps, window_size=5, center_idx=9)
        assert window.shape == (5, 17, 2)
        # Last frames should be padded with frame 9
        assert window[4, 0, 0] == pytest.approx(9.0)


# ── Temporal smoothing ───────────────────────────────────────

class TestApplyTemporalSmoothing:
    def test_smooth_reduces_jitter(self):
        # Create noisy trajectory (seeded for deterministic test behavior)
        n = 30
        rng = np.random.default_rng(0)
        base = np.linspace(0, 1, n)
        noise = rng.normal(0.0, 0.1, size=(n, 17, 3))
        joints = np.zeros((n, 17, 3))
        for j in range(17):
            for c in range(3):
                joints[:, j, c] = base + noise[:, j, c]

        smoothed = _apply_temporal_smoothing(joints, window_length=7, polyorder=3)
        assert smoothed.shape == joints.shape

        # Smoothed should have less variance
        original_var = np.var(np.diff(joints, axis=0))
        smoothed_var = np.var(np.diff(smoothed, axis=0))
        assert smoothed_var < original_var

    def test_too_few_frames(self):
        rng = np.random.default_rng(1)
        joints = rng.normal(size=(3, 17, 3)).astype(np.float32)
        smoothed = _apply_temporal_smoothing(joints, window_length=15, polyorder=3)
        # n <= polyorder -> smoothing should no-op
        assert smoothed.shape == (3, 17, 3)
        np.testing.assert_array_equal(smoothed, joints)

    def test_single_frame_no_crash(self):
        rng = np.random.default_rng(2)
        joints = rng.normal(size=(1, 17, 3)).astype(np.float32)
        smoothed = _apply_temporal_smoothing(joints, window_length=15, polyorder=3)
        # Should return unchanged (can't smooth 1 frame)
        np.testing.assert_array_equal(smoothed, joints)

    def test_short_sequences_do_not_crash(self):
        rng = np.random.default_rng(3)
        for n in (1, 2, 3):
            joints = rng.normal(size=(n, 17, 3)).astype(np.float32)
            smoothed = _apply_temporal_smoothing(joints, window_length=15, polyorder=3)
            assert smoothed.shape == joints.shape
            # For n <= polyorder, smoothing must be no-op
            np.testing.assert_array_equal(smoothed, joints)


# ── MotionBERT Backend ───────────────────────────────────────

class TestMotionBERTBackend:
    def test_invalid_smoothing_params(self):
        with pytest.raises(ValueError, match="smoothing_window"):
            MotionBERTBackend(smoothing_window=0)
        with pytest.raises(ValueError, match="smoothing_poly"):
            MotionBERTBackend(smoothing_poly=-1)

    def _create_mock_model(self):
        """Create a mock PyTorch model returning 3D joints in Y-up system."""
        import torch

        class MockModel(torch.nn.Module):
            def forward(self, input_tensor):
                output = torch.zeros(
                    (1, MOTIONBERT_WINDOW_SIZE, 17, 3),
                    dtype=torch.float32,
                    device=input_tensor.device,
                )
                for j in range(17):
                    output[0, :, j, 1] = float(j) * 0.1  # Y = stacked upward
                return output

        return MockModel()

    def _make_keypoints_2d(self, n_frames: int) -> list[Keypoints2D]:
        return [
            Keypoints2D(
                points=np.random.rand(17, 2).astype(np.float32) * 640,
                confidence=np.ones(17, dtype=np.float32) * 0.85,
                image_size=(480, 640),
            )
            for _ in range(n_frames)
        ]

    def test_lift_produces_pose3d(self):
        backend = MotionBERTBackend(model_path="/fake/path.bin")
        backend._model = self._create_mock_model()
        backend._device = "cpu"

        keypoints = self._make_keypoints_2d(5)
        poses = backend.lift(keypoints)

        assert len(poses) == 5
        for pose in poses:
            assert isinstance(pose, Pose3D)

    def test_confidence_propagation(self):
        backend = MotionBERTBackend(model_path="/fake/path.bin")
        backend._model = self._create_mock_model()
        backend._device = "cpu"

        keypoints = self._make_keypoints_2d(3)
        poses = backend.lift(keypoints)

        for pose in poses:
            assert pose.confidence is not None
            assert pose.confidence["head"] == pytest.approx(0.85)

    def test_empty_input(self):
        backend = MotionBERTBackend(model_path="/fake/path.bin")
        backend._model = self._create_mock_model()
        backend._device = "cpu"
        assert backend.lift([]) == []

    def test_coordinate_system_y_up(self):
        backend = MotionBERTBackend(model_path="/fake/path.bin", smoothing_window=3)
        backend._model = self._create_mock_model()
        backend._device = "cpu"

        keypoints = self._make_keypoints_2d(5)
        poses = backend.lift(keypoints)

        # Joints should be stacked upward (Y increases with joint index)
        # Head is COCO joint 0 -> y = 0.0
        # Right ankle is COCO joint 16 -> y = 1.6
        for pose in poses:
            head_y = pose.head[1]
            ankle_y = pose.right_ankle[1]
            # Ankle (COCO 16) should have higher Y than head (COCO 0)
            assert ankle_y > head_y
