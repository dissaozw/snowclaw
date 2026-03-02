## ADDED Requirements

### Requirement: Skeleton overlay rendering
The system SHALL draw a skeleton overlay on each video frame consisting of: colored dots at each detected joint, lines connecting adjacent joints (bone segments), and a vertical plumb line from the center of mass downward (balance reference).

#### Scenario: Render skeleton on a frame
- **WHEN** a video frame and its corresponding `core.Pose3D` (projected to 2D) are provided
- **THEN** the renderer SHALL draw joint dots and bone lines on the frame image

#### Scenario: Confidence-based coloring
- **WHEN** a joint has confidence >= 0.7
- **THEN** the dot SHALL be green
- **WHEN** a joint has confidence between 0.3 and 0.7
- **THEN** the dot SHALL be yellow
- **WHEN** a joint has confidence < 0.3
- **THEN** the dot SHALL be red

### Requirement: COM plumb line
The system SHALL draw a vertical line from the estimated center of mass straight downward to indicate balance alignment, similar to SkiPro AI's reference line.

#### Scenario: Draw plumb line
- **WHEN** a frame is annotated
- **THEN** a vertical line SHALL be drawn from the COM position downward to the bottom of the frame

### Requirement: Metrics text overlay
The system SHALL overlay basic metric values as text on each frame: knee flex angle (left/right), inclination angle, and COM height percentage.

#### Scenario: Display metrics on frame
- **WHEN** a frame has computed metrics
- **THEN** metric values SHALL be displayed as readable text in a fixed position on the frame (e.g., top-left corner)

### Requirement: Annotated video output
The system SHALL encode all annotated frames into an output video file (MP4, H.264) at the original video's FPS using FFmpeg.

#### Scenario: Generate annotated video
- **WHEN** all frames have been annotated
- **THEN** the system SHALL produce an MP4 file with the skeleton overlay baked in

#### Scenario: Preserve original audio
- **WHEN** the input video has an audio track
- **THEN** the output video SHALL include the original audio

### Requirement: Temporal smoothing for stable overlay
The system SHALL apply temporal smoothing (Savitzky-Golay filter) to joint positions before drawing, so the skeleton does not jitter frame-to-frame.

#### Scenario: Smooth skeleton across frames
- **WHEN** a sequence of frames is annotated
- **THEN** the drawn joint positions SHALL be temporally smooth with no single-frame jumps
