## ADDED Requirements

### Requirement: Abstract 2D pose estimator interface
The system SHALL define a `PoseEstimator2D` abstract base class with a `predict` method that accepts a list of RGB frames (NumPy arrays) and returns a list of `Keypoints2D` objects.

#### Scenario: Implementing a custom backend
- **WHEN** a developer subclasses `PoseEstimator2D` and implements `predict`
- **THEN** the subclass SHALL be usable anywhere the base class is accepted

### Requirement: ViTPose+ backend
The system SHALL provide a `ViTPoseBackend` implementation of `PoseEstimator2D` that runs ViTPose+ inference via ONNX Runtime.

#### Scenario: Predict keypoints from video frames
- **WHEN** a list of RGB frames is passed to `ViTPoseBackend.predict()`
- **THEN** a list of `Keypoints2D` objects SHALL be returned, one per frame, each containing 17 COCO-format keypoints with pixel coordinates and confidence scores

#### Scenario: Batch processing
- **WHEN** more frames than `batch_size` are provided
- **THEN** frames SHALL be processed in batches and results concatenated transparently

### Requirement: Keypoints2D data structure
The system SHALL define a `Keypoints2D` dataclass with fields: `points` (ndarray shape Jx2, pixel coords), `confidence` (ndarray shape J, values in 0-1), and `image_size` (height, width tuple).

#### Scenario: Access keypoint data
- **WHEN** a `Keypoints2D` object is returned from any backend
- **THEN** `points` SHALL contain pixel-space (x, y) coordinates and `confidence` SHALL contain per-joint scores between 0 and 1

### Requirement: Model weight management
The system SHALL download model weights on first use to `~/.cache/snowclaw/models/` and reuse cached weights on subsequent runs.

#### Scenario: First-time model load
- **WHEN** `ViTPoseBackend` is instantiated and weights are not cached
- **THEN** weights SHALL be downloaded with a progress indicator and saved to the cache directory

#### Scenario: Cached model load
- **WHEN** `ViTPoseBackend` is instantiated and weights already exist in cache
- **THEN** weights SHALL be loaded from cache without any network request

### Requirement: Device selection
The system SHALL support running inference on CPU or CUDA GPU, selectable via a `device` parameter (default: auto-detect).

#### Scenario: GPU available
- **WHEN** `device="auto"` and a CUDA GPU is available
- **THEN** inference SHALL run on GPU

#### Scenario: CPU fallback
- **WHEN** `device="auto"` and no GPU is available
- **THEN** inference SHALL run on CPU without error
