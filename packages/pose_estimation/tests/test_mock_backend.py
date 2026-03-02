"""Tests for MockViTPoseBackend and MockMotionBERTBackend."""

import numpy as np
import pytest

from pose_estimation.mock_backend import MockViTPoseBackend, MockMotionBERTBackend
from pose_estimation.interfaces import Keypoints2D
from core.schemas import Pose3D


@pytest.fixture
def fake_frames() -> list[np.ndarray]:
    """5 synthetic RGB frames (240×320)."""
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(5)]


@pytest.fixture
def mock_estimator() -> MockViTPoseBackend:
    return MockViTPoseBackend()


@pytest.fixture
def mock_lifter() -> MockMotionBERTBackend:
    return MockMotionBERTBackend()


# ──────────────────────────────────────────────────────────────
# MockViTPoseBackend
# ──────────────────────────────────────────────────────────────

class TestMockViTPoseBackend:
    def test_returns_one_result_per_frame(self, mock_estimator, fake_frames):
        results = mock_estimator.predict(fake_frames)
        assert len(results) == len(fake_frames)

    def test_result_type_is_keypoints2d(self, mock_estimator, fake_frames):
        results = mock_estimator.predict(fake_frames)
        for r in results:
            assert isinstance(r, Keypoints2D)

    def test_points_shape(self, mock_estimator, fake_frames):
        results = mock_estimator.predict(fake_frames)
        for r in results:
            assert r.points.shape == (17, 2)

    def test_confidence_shape(self, mock_estimator, fake_frames):
        results = mock_estimator.predict(fake_frames)
        for r in results:
            assert r.confidence.shape == (17,)

    def test_confidence_in_range(self, mock_estimator, fake_frames):
        results = mock_estimator.predict(fake_frames)
        for r in results:
            assert np.all(r.confidence >= 0.0)
            assert np.all(r.confidence <= 1.0)

    def test_points_within_image_bounds(self, mock_estimator, fake_frames):
        """
        Most projected keypoints should land within the image frame.
        A generous margin is allowed because the skeleton is orthographically
        projected and head/ankle joints can clip slightly outside short frames.
        """
        results = mock_estimator.predict(fake_frames)
        h, w = fake_frames[0].shape[:2]
        margin = min(h, w) * 0.2  # 20% margin of the shorter dimension
        for r in results:
            assert np.all(r.points[:, 0] > -margin), "x coord too far left"
            assert np.all(r.points[:, 0] < w + margin), "x coord too far right"
            assert np.all(r.points[:, 1] > -margin), "y coord too far up"
            assert np.all(r.points[:, 1] < h + margin), "y coord too far down"

    def test_deterministic_per_frame(self, mock_estimator, fake_frames):
        """Same frame index always produces same keypoints."""
        r1 = mock_estimator.predict(fake_frames)
        r2 = mock_estimator.predict(fake_frames)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a.points, b.points)

    def test_image_size_stored(self, mock_estimator, fake_frames):
        results = mock_estimator.predict(fake_frames)
        h, w = fake_frames[0].shape[:2]
        for r in results:
            assert r.image_size == (h, w)

    def test_single_frame(self, mock_estimator):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = mock_estimator.predict([frame])
        assert len(results) == 1
        assert results[0].points.shape == (17, 2)

    def test_empty_input(self, mock_estimator):
        results = mock_estimator.predict([])
        assert results == []

    def test_custom_jitter_zero(self):
        """With jitter=0, results should be perfectly deterministic across runs."""
        est = MockViTPoseBackend(jitter_px=0.0)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        r1 = est.predict([frame])
        r2 = est.predict([frame])
        np.testing.assert_array_equal(r1[0].points, r2[0].points)


# ──────────────────────────────────────────────────────────────
# MockMotionBERTBackend
# ──────────────────────────────────────────────────────────────

class TestMockMotionBERTBackend:
    def test_returns_one_pose_per_frame(self, mock_estimator, mock_lifter, fake_frames):
        kp2d = mock_estimator.predict(fake_frames)
        poses = mock_lifter.lift(kp2d)
        assert len(poses) == len(fake_frames)

    def test_result_type_is_pose3d(self, mock_estimator, mock_lifter, fake_frames):
        kp2d = mock_estimator.predict(fake_frames)
        poses = mock_lifter.lift(kp2d)
        for p in poses:
            assert isinstance(p, Pose3D)

    def test_joints_are_3d(self, mock_estimator, mock_lifter, fake_frames):
        kp2d = mock_estimator.predict(fake_frames)
        poses = mock_lifter.lift(kp2d)
        for pose in poses:
            assert len(pose.head) == 3
            assert len(pose.left_knee) == 3

    def test_y_up_coordinate_system(self, mock_estimator, mock_lifter, fake_frames):
        """Head should be higher than hips in Y-up convention."""
        kp2d = mock_estimator.predict(fake_frames)
        poses = mock_lifter.lift(kp2d)
        for pose in poses:
            assert pose.head[1] > pose.hip_midpoint[1], \
                f"Head Y={pose.head[1]:.2f} should be above hip Y={pose.hip_midpoint[1]:.2f}"

    def test_hips_above_ankles(self, mock_estimator, mock_lifter, fake_frames):
        kp2d = mock_estimator.predict(fake_frames)
        poses = mock_lifter.lift(kp2d)
        for pose in poses:
            assert pose.hip_midpoint[1] > pose.ankle_midpoint[1]

    def test_body_height_plausible(self, mock_estimator, mock_lifter, fake_frames):
        """Mock skeleton should be roughly human-sized (1.5–2.1 m)."""
        kp2d = mock_estimator.predict(fake_frames)
        poses = mock_lifter.lift(kp2d)
        for pose in poses:
            assert 1.5 < pose.body_height < 2.1, \
                f"Body height {pose.body_height:.2f}m outside plausible range"

    def test_com_reasonable(self, mock_estimator, mock_lifter, fake_frames):
        """COM should be roughly at torso height (0.8–1.4 m above ground)."""
        kp2d = mock_estimator.predict(fake_frames)
        poses = mock_lifter.lift(kp2d)
        for pose in poses:
            com_y = pose.com[1]
            assert 0.8 < com_y < 1.4, f"COM Y={com_y:.2f}m outside expected range"

    def test_per_frame_variation(self, mock_estimator, mock_lifter, fake_frames):
        """Consecutive frames should have slightly different poses (oscillation)."""
        if len(fake_frames) < 2:
            pytest.skip("Need at least 2 frames")
        kp2d = mock_estimator.predict(fake_frames)
        poses = mock_lifter.lift(kp2d)
        # The oscillation offset means consecutive frames differ
        p0 = np.array(poses[0].head)
        p1 = np.array(poses[1].head)
        assert not np.allclose(p0, p1), "Consecutive frames should have slightly different poses"

    def test_empty_input(self, mock_lifter):
        poses = mock_lifter.lift([])
        assert poses == []

    def test_confidence_propagated(self, mock_estimator, mock_lifter, fake_frames):
        kp2d = mock_estimator.predict(fake_frames)
        poses = mock_lifter.lift(kp2d)
        for pose in poses:
            assert pose.confidence is not None
            assert "head" in pose.confidence
            assert 0.0 <= pose.confidence["head"] <= 1.0


# ──────────────────────────────────────────────────────────────
# Full pipeline round-trip
# ──────────────────────────────────────────────────────────────

class TestMockPipelineRoundtrip:
    def test_full_mock_pipeline(self, fake_frames):
        """End-to-end: frames → 2D keypoints → 3D poses."""
        estimator = MockViTPoseBackend()
        lifter = MockMotionBERTBackend()

        kp2d = estimator.predict(fake_frames)
        poses_3d = lifter.lift(kp2d)

        assert len(poses_3d) == len(fake_frames)
        for pose in poses_3d:
            assert isinstance(pose, Pose3D)
            assert pose.head[1] > pose.ankle_midpoint[1]  # Y-up sanity check
