## ADDED Requirements

### Requirement: 3D skeleton rendering
The system SHALL render a 3D skeleton in the browser using React Three Fiber, with spheres at each of the 14 core joint positions and cylinders connecting adjacent joints (bone segments).

#### Scenario: View skeleton at a specific frame
- **WHEN** the user navigates to a frame in the viewer
- **THEN** the skeleton SHALL be rendered with joints and bones at the 3D positions from `core.Pose3D`

### Requirement: Orbit controls
The system SHALL provide orbit controls allowing the user to rotate (drag), zoom (scroll), and pan (right-drag) around the skeleton.

#### Scenario: Rotate around the skeleton
- **WHEN** the user clicks and drags in the viewer
- **THEN** the camera SHALL orbit around the skeleton's center of mass

### Requirement: Frame scrubber / timeline
The system SHALL provide a timeline scrubber that allows the user to navigate to any frame in the video.

#### Scenario: Scrub to a specific frame
- **WHEN** the user drags the timeline scrubber to frame 150
- **THEN** the skeleton SHALL update to show the pose at frame 150

#### Scenario: Play/pause animation
- **WHEN** the user clicks play
- **THEN** the skeleton SHALL animate through frames at the video's original FPS

### Requirement: Ground reference plane
The system SHALL render a simple flat grid plane at y=0 as a spatial reference. This is NOT a reconstructed snow surface — just a visual anchor.

#### Scenario: View skeleton with ground reference
- **WHEN** the skeleton is rendered
- **THEN** a flat grid plane SHALL be visible at y=0 to provide spatial context

### Requirement: Metric overlay in viewer
The system SHALL display the current frame's basic metrics (knee angle, inclination, COM height) as a text panel alongside the 3D view.

#### Scenario: View metrics while orbiting
- **WHEN** the user orbits the skeleton at frame N
- **THEN** the metrics panel SHALL show the metric values for frame N

### Requirement: Skeleton-only scope
The viewer SHALL render skeleton wireframe only. Body mesh, snow surface reconstruction, and ski/snowboard geometry are explicitly out of scope for Phase 1.

#### Scenario: No mesh or equipment rendered
- **WHEN** the viewer is loaded
- **THEN** only joint spheres, bone cylinders, and the ground grid SHALL be rendered — no body mesh, no skis, no snow surface
