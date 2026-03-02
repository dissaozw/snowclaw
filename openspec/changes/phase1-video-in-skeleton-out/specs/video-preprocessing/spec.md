## ADDED Requirements

### Requirement: Frame extraction from video files
The system SHALL extract frames from video files (MP4, MOV, AVI, MKV) at a configurable target FPS using FFmpeg. Extracted frames SHALL be returned as a list of NumPy arrays in RGB format with shape (H, W, 3).

#### Scenario: Extract frames at default FPS
- **WHEN** a valid MP4 video is provided with no FPS override
- **THEN** frames SHALL be extracted at the video's native FPS

#### Scenario: Extract frames at custom FPS
- **WHEN** a valid video and `target_fps=10` are provided
- **THEN** frames SHALL be extracted at approximately 10 frames per second

#### Scenario: Invalid video file
- **WHEN** a non-existent or corrupt file path is provided
- **THEN** the system SHALL raise a `VideoProcessingError` with a descriptive message

### Requirement: Resolution normalization
The system SHALL resize extracted frames to a configurable maximum dimension (default 1920px on the long edge) while preserving aspect ratio.

#### Scenario: Downscale a 4K video
- **WHEN** a 3840x2160 video is processed with `max_dimension=1920`
- **THEN** extracted frames SHALL have dimensions 1920x1080

#### Scenario: No upscaling of small videos
- **WHEN** a 1280x720 video is processed with `max_dimension=1920`
- **THEN** frames SHALL retain their original 1280x720 resolution

### Requirement: Video metadata extraction
The system SHALL extract and return video metadata including: duration (seconds), native FPS, resolution (width x height), codec, and total frame count.

#### Scenario: Extract metadata from a standard video
- **WHEN** a valid video file is provided
- **THEN** a `VideoMetadata` object SHALL be returned with all fields populated

### Requirement: Keyframe selection
The system SHALL provide a keyframe selection mode that identifies frames with significant visual change (scene cuts, large motion) using frame differencing.

#### Scenario: Keyframe extraction from a ski video
- **WHEN** keyframe selection is enabled with `threshold=0.3`
- **THEN** only frames where the normalized frame difference exceeds the threshold SHALL be returned

### Requirement: FFmpeg availability check
The system SHALL verify that FFmpeg is available on the system PATH at initialization time.

#### Scenario: FFmpeg not installed
- **WHEN** the `ffmpeg` binary is not found on PATH
- **THEN** the system SHALL raise a `DependencyError` with instructions for installing FFmpeg
