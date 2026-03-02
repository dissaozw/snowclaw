## ADDED Requirements

### Requirement: Video upload endpoint
The system SHALL provide a `POST /api/upload` endpoint that accepts a video file and returns a job ID for tracking the processing pipeline.

#### Scenario: Upload a valid video
- **WHEN** a user POSTs a video file (MP4, MOV, AVI, MKV) to `/api/upload`
- **THEN** the system SHALL return a JSON response with a `job_id` and status `"queued"`

#### Scenario: Upload an invalid file
- **WHEN** a user POSTs a non-video file
- **THEN** the system SHALL return a 400 error with a descriptive message

### Requirement: Async pipeline execution
The system SHALL process videos asynchronously via a Celery task queue backed by Redis, so the API remains responsive during long-running ML inference.

#### Scenario: Video processing runs in background
- **WHEN** a video is uploaded
- **THEN** the ML pipeline (frame extraction → 2D pose → 3D lifting → annotation) SHALL execute as a background Celery task

### Requirement: WebSocket progress updates
The system SHALL provide a WebSocket endpoint `GET /api/ws/status/{job_id}` that streams real-time progress updates during processing.

#### Scenario: Receive progress updates
- **WHEN** a client connects to the WebSocket for an active job
- **THEN** the client SHALL receive messages with stage name and progress percentage (e.g., `{"stage": "pose_2d", "progress": 45}`)

#### Scenario: Job completes
- **WHEN** processing finishes
- **THEN** the WebSocket SHALL send a final message with status `"complete"` and result URLs

### Requirement: Annotated video download
The system SHALL provide a `GET /api/results/{job_id}/video` endpoint that returns the annotated video file.

#### Scenario: Download annotated video
- **WHEN** a client requests the video for a completed job
- **THEN** the system SHALL return the MP4 file with skeleton overlay

#### Scenario: Job not complete
- **WHEN** a client requests results for a still-processing job
- **THEN** the system SHALL return a 202 status with the current progress

### Requirement: 3D skeleton data endpoint
The system SHALL provide a `GET /api/results/{job_id}/poses` endpoint that returns per-frame 3D pose data as JSON, consumable by the 3D viewer.

#### Scenario: Fetch pose data for viewer
- **WHEN** a client requests poses for a completed job
- **THEN** the system SHALL return a JSON array of per-frame `core.Pose3D` data with timestamps

### Requirement: Result cleanup
The system SHALL automatically clean up job results (video files, pose data) after a configurable retention period (default: 24 hours).

#### Scenario: Expired results
- **WHEN** a job's results are older than the retention period
- **THEN** the associated files SHALL be deleted and the endpoint SHALL return 410 Gone
