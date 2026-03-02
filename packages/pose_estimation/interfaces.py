"""Abstract base classes and data types for pose estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Keypoints2D:
    """2D keypoint detection result for a single frame.

    Attributes:
        points: Jx2 ndarray of (x, y) pixel coordinates.
        confidence: J-length ndarray of per-joint confidence scores in [0, 1].
        image_size: (height, width) tuple of the source image.
    """

    points: np.ndarray  # shape (J, 2)
    confidence: np.ndarray  # shape (J,)
    image_size: tuple[int, int]  # (H, W)

    def __post_init__(self):
        assert self.points.ndim == 2 and self.points.shape[1] == 2, (
            f"points must be (J, 2), got {self.points.shape}"
        )
        assert self.confidence.ndim == 1, (
            f"confidence must be (J,), got {self.confidence.shape}"
        )
        assert self.points.shape[0] == self.confidence.shape[0], (
            f"points ({self.points.shape[0]}) and confidence ({self.confidence.shape[0]}) "
            f"must have the same number of joints"
        )


class PoseEstimator2D(ABC):
    """Abstract base class for 2D keypoint detection models."""

    @abstractmethod
    def predict(self, frames: list[np.ndarray]) -> list[Keypoints2D]:
        """
        Detect 2D keypoints in a list of video frames.

        Args:
            frames: List of RGB images as numpy arrays, shape (H, W, 3), dtype uint8.

        Returns:
            List of Keypoints2D, one per input frame.
        """
        ...


class PoseLifter3D(ABC):
    """Abstract base class for 2D-to-3D pose lifting models."""

    @abstractmethod
    def lift(self, keypoints_2d: list[Keypoints2D]) -> list:
        """
        Lift 2D keypoints to 3D poses.

        Args:
            keypoints_2d: List of Keypoints2D from a 2D estimator.

        Returns:
            List of core.Pose3D, one per input frame.
        """
        ...
