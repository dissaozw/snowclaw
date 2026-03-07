"""Tests for pose estimation interfaces, joint mapping, and model cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.schemas import Pose3D
from pose_estimation.interfaces import Keypoints2D, PoseEstimator2D, PoseLifter3D
from pose_estimation.joint_mapping import COCO_TO_POSE3D, coco_keypoints_to_pose3d
from pose_estimation.model_cache import download_model


# ── Keypoints2D ──────────────────────────────────────────────

class TestKeypoints2D:
    def test_construction(self):
        kp = Keypoints2D(
            points=np.zeros((17, 2)),
            confidence=np.ones(17),
            image_size=(480, 640),
        )
        assert kp.points.shape == (17, 2)
        assert kp.confidence.shape == (17,)
        assert kp.image_size == (480, 640)

    def test_mismatched_joints_raises(self):
        with pytest.raises(AssertionError, match="same number of joints"):
            Keypoints2D(
                points=np.zeros((17, 2)),
                confidence=np.ones(10),  # wrong size
                image_size=(480, 640),
            )

    def test_wrong_points_shape_raises(self):
        with pytest.raises(AssertionError, match="must be"):
            Keypoints2D(
                points=np.zeros((17, 3)),  # should be (J, 2)
                confidence=np.ones(17),
                image_size=(480, 640),
            )


# ── Abstract classes ─────────────────────────────────────────

class TestAbstractClasses:
    def test_cannot_instantiate_estimator_2d(self):
        with pytest.raises(TypeError):
            PoseEstimator2D()

    def test_cannot_instantiate_lifter_3d(self):
        with pytest.raises(TypeError):
            PoseLifter3D()

    def test_concrete_estimator(self):
        class DummyEstimator(PoseEstimator2D):
            def predict(self, frames):
                return [
                    Keypoints2D(
                        points=np.zeros((17, 2)),
                        confidence=np.ones(17),
                        image_size=(frames[0].shape[0], frames[0].shape[1]),
                    )
                    for _ in frames
                ]

        est = DummyEstimator()
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = est.predict(frames)
        assert len(results) == 1
        assert results[0].points.shape == (17, 2)

    def test_concrete_lifter(self):
        class DummyLifter(PoseLifter3D):
            def lift(self, keypoints_2d):
                return [None] * len(keypoints_2d)

        lifter = DummyLifter()
        result = lifter.lift([None])
        assert len(result) == 1


# ── Joint mapping ────────────────────────────────────────────

class TestJointMapping:
    def test_mapping_covers_14_joints(self):
        assert len(COCO_TO_POSE3D) == 14

    def test_all_coco_indices_valid(self):
        for joint, indices in COCO_TO_POSE3D.items():
            for idx in indices:
                assert 0 <= idx < 17, f"Invalid COCO index {idx} for {joint}"

    def test_coco_to_pose3d_conversion(self):
        # Create synthetic COCO 3D joints (17 joints, each at a different Y height)
        joints = np.zeros((17, 3))
        for i in range(17):
            joints[i] = [0.0, float(i) / 16.0, 0.0]

        confidence = np.ones(17) * 0.9

        pose = coco_keypoints_to_pose3d(joints, confidence)
        assert isinstance(pose, Pose3D)
        # Head = nose (index 0) -> y = 0.0
        assert pose.head[1] == pytest.approx(0.0)
        # Neck = midpoint of shoulders (indices 5, 6)
        expected_neck_y = (5.0 / 16.0 + 6.0 / 16.0) / 2.0
        assert pose.neck[1] == pytest.approx(expected_neck_y)
        assert pose.confidence is not None
        assert pose.confidence["head"] == pytest.approx(0.9)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="Expected shape"):
            coco_keypoints_to_pose3d(np.zeros((14, 3)))

    def test_no_confidence(self):
        joints = np.zeros((17, 3))
        pose = coco_keypoints_to_pose3d(joints, confidence=None)
        assert pose.confidence is None


# ── Model cache ──────────────────────────────────────────────

class TestModelCache:
    def test_download_and_cache(self, tmp_path):
        """Test caching with a local file:// URL."""
        # Create a fake model file
        src = tmp_path / "source_model.onnx"
        src.write_bytes(b"fake model data")

        cache_dir = tmp_path / "cache"
        path = download_model(
            url=f"file://{src}",
            filename="test_model.onnx",
            cache_dir=cache_dir,
        )
        assert path.exists()
        assert path.name == "test_model.onnx"
        assert path.read_bytes() == b"fake model data"

    def test_uses_cache_on_second_call(self, tmp_path):
        src = tmp_path / "source_model.onnx"
        src.write_bytes(b"fake model data")

        cache_dir = tmp_path / "cache"
        path1 = download_model(
            url=f"file://{src}",
            filename="test_model.onnx",
            cache_dir=cache_dir,
        )
        # Delete source — second call should use cache
        src.unlink()
        path2 = download_model(
            url=f"file://{src}",
            filename="test_model.onnx",
            cache_dir=cache_dir,
        )
        assert path1 == path2
        assert path2.exists()


class TestMotionBERTYAxisFlip:
    """
    Regression tests for the MotionBERT Y-axis flip.

    MotionBERT outputs camera-space Y (Y-down). The backend must negate Y so
    that downstream code (com_height_pct etc.) works in world Y-up convention.
    """

    def _make_fake_kp2d(self, n_frames: int = 5) -> list:
        """Generate minimal Keypoints2D list for lifting."""
        import numpy as np
        from pose_estimation.interfaces import Keypoints2D

        rng = np.random.default_rng(0)
        return [
            Keypoints2D(
                points=rng.uniform(100, 900, size=(17, 2)).astype(np.float32),
                confidence=np.full(17, 0.9, dtype=np.float32),
                image_size=(1080, 1920),
            )
            for _ in range(n_frames)
        ]

    def test_y_up_after_lift(self, monkeypatch):
        """
        After lifting, head Y must be greater than ankle Y (Y-up).
        Simulates MotionBERT returning camera-space Y (inverted) and asserts the
        backend flips it correctly.
        """
        import numpy as np
        import torch
        from pose_estimation.motionbert_backend import MotionBERTBackend, MOTIONBERT_WINDOW_SIZE

        # Build a fake model output that mimics MotionBERT camera-Y convention:
        # head (idx 0 / nose) has MORE NEGATIVE Y than ankles (idx 15, 16).
        fake_joints_camera_y = np.zeros((17, 3), dtype=np.float32)
        fake_joints_camera_y[0, 1] = -0.7   # nose — high in world → negative camera Y
        fake_joints_camera_y[15, 1] = -0.05  # left_ankle — low in world → near-zero camera Y
        fake_joints_camera_y[16, 1] = -0.05  # right_ankle

        class FakeModel(torch.nn.Module):
            def forward(self, x):
                B, T, J, C = x.shape
                out = torch.zeros(B, T, J, 3)
                # Fill every frame with our fake camera-Y skeleton
                out[:, :, :, :] = torch.from_numpy(fake_joints_camera_y)
                return out

        backend = MotionBERTBackend.__new__(MotionBERTBackend)
        backend._model = FakeModel()
        backend._smoothing_window = 3
        backend._smoothing_poly = 2
        backend._device = "cpu"

        kp2d = self._make_fake_kp2d(n_frames=5)
        poses = backend.lift(kp2d)

        for pose in poses:
            assert pose.head[1] > pose.ankle_midpoint[1], (
                f"After Y-flip, head Y ({pose.head[1]:.3f}) should be above "
                f"ankle Y ({pose.ankle_midpoint[1]:.3f}). "
                "MotionBERT Y-axis flip may be missing."
            )

    def test_com_height_pct_nonzero_after_lift(self, monkeypatch):
        """com_height_pct must not be 0 after the Y-axis fix."""
        import numpy as np
        import torch
        from pose_estimation.motionbert_backend import MotionBERTBackend
        from biomechanics.metrics import com_height_pct

        fake_joints_camera_y = np.zeros((17, 3), dtype=np.float32)
        fake_joints_camera_y[0, 1] = -0.70   # nose
        fake_joints_camera_y[5, 1] = -0.55   # left_shoulder
        fake_joints_camera_y[6, 1] = -0.55   # right_shoulder
        fake_joints_camera_y[11, 1] = -0.30  # left_hip
        fake_joints_camera_y[12, 1] = -0.30  # right_hip
        fake_joints_camera_y[13, 1] = -0.15  # left_knee
        fake_joints_camera_y[14, 1] = -0.15  # right_knee
        fake_joints_camera_y[15, 1] = -0.02  # left_ankle
        fake_joints_camera_y[16, 1] = -0.02  # right_ankle

        class FakeModel(torch.nn.Module):
            def forward(self, x):
                B, T, J, _ = x.shape
                out = torch.zeros(B, T, J, 3)
                out[:, :, :, :] = torch.from_numpy(fake_joints_camera_y)
                return out

        backend = MotionBERTBackend.__new__(MotionBERTBackend)
        backend._model = FakeModel()
        backend._smoothing_window = 3
        backend._smoothing_poly = 2
        backend._device = "cpu"

        kp2d = self._make_fake_kp2d(n_frames=5)
        poses = backend.lift(kp2d)

        for pose in poses:
            h = com_height_pct(
                pose.com,
                pose.ankle_midpoint,
                np.array([0.0, 1.0, 0.0]),
                pose.body_height,
            )
            assert h > 0.0, f"com_height_pct={h} should be > 0 after Y-axis flip"
