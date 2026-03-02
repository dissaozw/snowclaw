"""Tests for ViTPose+ backend with mock ONNX session."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pose_estimation.vitpose_backend import (
    VITPOSE_INPUT_SIZE,
    VITPOSE_NUM_KEYPOINTS,
    ViTPoseBackend,
    _decode_heatmaps,
    _preprocess_frame,
)


class TestPreprocessFrame:
    def test_output_shape(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = _preprocess_frame(frame, VITPOSE_INPUT_SIZE)
        assert out.shape == (3, 256, 192)
        assert out.dtype == np.float32

    def test_normalization_range(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        out = _preprocess_frame(frame, VITPOSE_INPUT_SIZE)
        # Should be centered around 0 after ImageNet normalization
        assert -3.0 < out.mean() < 3.0


class TestDecodeHeatmaps:
    def test_output_shapes(self):
        # Create a heatmap with a known peak
        heatmaps = np.zeros((17, 64, 48), dtype=np.float32)
        for i in range(17):
            heatmaps[i, 32, 24] = 1.0  # Peak at center
        points, conf = _decode_heatmaps(heatmaps, (480, 640))
        assert points.shape == (17, 2)
        assert conf.shape == (17,)

    def test_peak_at_center(self):
        heatmaps = np.zeros((1, 64, 48), dtype=np.float32)
        heatmaps[0, 32, 24] = 1.0
        points, conf = _decode_heatmaps(heatmaps, (480, 640))
        # Peak at heatmap center (24/48*640, 32/64*480) = (320, 240)
        assert points[0, 0] == pytest.approx(320.0)
        assert points[0, 1] == pytest.approx(240.0)
        assert conf[0] == pytest.approx(1.0)

    def test_confidence_propagation(self):
        heatmaps = np.zeros((17, 64, 48), dtype=np.float32)
        heatmaps[0, 10, 10] = 0.85
        heatmaps[1, 20, 20] = 0.65
        _, conf = _decode_heatmaps(heatmaps, (480, 640))
        assert conf[0] == pytest.approx(0.85)
        assert conf[1] == pytest.approx(0.65)


class TestViTPoseBackend:
    def _create_mock_session(self, batch_size: int = 1):
        """Create a mock ONNX session that returns proper shaped output."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]

        def mock_run(output_names, feed_dict):
            input_data = feed_dict["input"]
            bs = input_data.shape[0]
            # Return heatmaps: (batch, 17, 64, 48)
            heatmaps = np.zeros((bs, 17, 64, 48), dtype=np.float32)
            for i in range(bs):
                for j in range(17):
                    heatmaps[i, j, 32, 24] = 0.9  # Peak at center
            return [heatmaps]

        mock_session.run = mock_run
        return mock_session

    def test_predict_single_frame(self):
        backend = ViTPoseBackend(model_path="/fake/path.onnx")
        backend._session = self._create_mock_session()

        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)]
        results = backend.predict(frames)

        assert len(results) == 1
        assert results[0].points.shape == (17, 2)
        assert results[0].confidence.shape == (17,)
        assert results[0].image_size == (480, 640)

    def test_predict_multiple_frames(self):
        backend = ViTPoseBackend(model_path="/fake/path.onnx", batch_size=4)
        backend._session = self._create_mock_session()

        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(7)]
        results = backend.predict(frames)

        assert len(results) == 7

    def test_batching(self):
        backend = ViTPoseBackend(model_path="/fake/path.onnx", batch_size=3)
        mock_session = self._create_mock_session()
        backend._session = mock_session

        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(7)]
        results = backend.predict(frames)

        assert len(results) == 7
