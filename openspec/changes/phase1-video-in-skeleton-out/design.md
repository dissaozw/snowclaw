## Context

SnowCoach AI has a biomechanics engine but no user-facing product. Phase 1 ("Video In, Skeleton Out") delivers the minimal end-to-end experience: upload a ski video, get back (1) an annotated video with skeleton and metrics, and (2) a simple 3D viewer to orbit the skeleton at any frame.

The existing `biomechanics` package defines shared types (`Pose3D`, `Frame`) that other packages need. These must move to `packages/core/` so the dependency graph flows cleanly without cross-package coupling.

**Current state:** Video → _(nothing happens)_
**Target state:** Video → Annotated video + 3D skeleton viewer

## Goals / Non-Goals

**Goals:**
- Create `packages/core/` with shared data structures
- Extract frames from video with FFmpeg
- Detect 2D keypoints (ViTPose+) and lift to 3D (MotionBERT) with pluggable backends
- Render annotated output video: skeleton overlay, COM plumb line, basic metrics (knee angle, inclination, COM height)
- Simple 3D skeleton viewer: orbit controls, frame scrubber, skeleton wireframe
- FastAPI backend: upload video → async pipeline → download annotated video + view 3D

**Non-Goals:**
- SMPL-X body mesh (Phase 2)
- Snow surface reconstruction / depth estimation (Phase 2)
- Ski/snowboard equipment detection and rendering (Phase 2)
- Snowboard-specific mode (Phase 2)
- LLM coaching / SnowIQ scoring (Phase 3)
- Multi-person tracking
- Real-time / streaming inference
- Model training or fine-tuning

## Decisions

### 1. Shared types in `packages/core/`

**Decision:** Extract `Pose3D`, `Frame`, `Discipline`, `TurnPhaseLabel`, `TurnPhase`, `SessionMetrics` into `packages/core/schemas.py`. Biomechanics re-exports for backward compatibility.

**Rationale:** Dependency graph must flow downward: `core` ← everything else. Pose estimation is upstream of biomechanics — it should not import from it.

### 2. Two deliverables: annotated video + 3D viewer

**Decision:** Phase 1 produces two outputs, not one:
1. **Annotated video** (MP4) — skeleton drawn on each frame with metrics. Downloadable, shareable, works everywhere.
2. **3D skeleton viewer** (web) — orbit around the skeleton at any frame. Skeleton wireframe only — no mesh, no slope, no equipment.

**Rationale:** The annotated video is the fast, shareable output (like SkiPro AI). The 3D viewer is the differentiator but starts simple — skeleton only. Phase 2 upgrades it with body mesh, snow surface, and ski geometry.

### 3. Video annotation with OpenCV + FFmpeg

**Decision:** Draw skeleton overlays using OpenCV (`cv2.circle`, `cv2.line`, `cv2.putText`) on each frame, then re-encode with FFmpeg.

**Rationale:** OpenCV is lightweight, fast, and purpose-built for image annotation. No need for a heavier rendering engine for 2D overlays. FFmpeg handles encoding with proper codec support.

**Overlay elements:**
- Joint dots (colored by confidence: green = high, yellow = medium, red = low)
- Skeleton lines connecting adjacent joints
- COM plumb line (vertical line from center of mass to ground — balance reference)
- Metric text overlay: knee flex angle, inclination angle, COM height %

### 4. Simple 3D viewer with React Three Fiber

**Decision:** Skeleton-only viewer using React Three Fiber. Joints as spheres, bones as cylinders. Orbit controls + frame scrubber timeline.

**Rationale:** React Three Fiber is already in the planned tech stack (PLAN.md §5). Starting with skeleton-only keeps scope manageable. Phase 2 adds body mesh (SMPL-X), snow surface, and ski geometry.

**What the viewer shows:**
- Spheres at each of the 14 joint positions
- Cylinders connecting adjacent joints (bone segments)
- Ground plane at y=0 as a simple spatial reference
- Orbit controls (drag to rotate, scroll to zoom)
- Frame scrubber / timeline to step through the video
- Current frame's metric values displayed as text overlay

### 5. FFmpeg via subprocess

**Decision:** Call FFmpeg directly via `subprocess.run` with typed wrappers. No `ffmpeg-python` library.

**Rationale:** `ffmpeg-python` is poorly maintained. Direct subprocess calls are transparent and zero-dependency.

### 6. Abstract base classes for model backends

**Decision:** `PoseEstimator2D` (ABC) and `PoseLifter3D` (ABC) interfaces with ViTPose+ and MotionBERT as shipped implementations.

**Rationale:** PLAN.md calls for pluggable backends so the community can swap in newer models.

### 7. ONNX Runtime as default inference engine

**Decision:** Ship models as ONNX with PyTorch fallback via optional `[torch]` extra.

**Rationale:** ONNX Runtime is lightweight (~50MB vs ~2GB for PyTorch), runs on CPU and GPU.

### 8. FastAPI backend with async pipeline

**Decision:** FastAPI server with: (1) POST `/upload` → returns job ID, (2) WebSocket `/ws/status/{job_id}` → progress updates, (3) GET `/results/{job_id}/video` → annotated video, (4) GET `/results/{job_id}/poses` → 3D skeleton data as JSON for the viewer.

**Rationale:** Video processing is slow (minutes). Async pipeline with progress updates is essential UX. FastAPI is the planned tech (PLAN.md §5) with native async + auto-docs.

**Job queue:** Celery + Redis for async ML pipeline execution.

### 9. Temporal smoothing for stable skeleton

**Decision:** Apply Savitzky-Golay filter to 3D joint trajectories after MotionBERT lifting.

**Rationale:** Even with MotionBERT's temporal consistency, some frame-to-frame jitter remains. A light smoothing pass produces the stable, non-jittery skeleton the user expects — a key improvement over SkiPro AI's visible jitter.

## Risks / Trade-offs

**[Model accuracy on ski clothing]** → ViTPose+ not fine-tuned on winter sports. **Mitigation:** Confidence-based coloring shows users where detection is uncertain. Fine-tuning is a later concern.

**[FFmpeg as system dependency]** → Not pip-installable. **Mitigation:** Clear error message + install instructions. Docker images include it.

**[Large model downloads]** → ~500MB total. **Mitigation:** Lazy download with progress bar, cached in `~/.cache/snowclaw/`.

**[3D viewer scope creep]** → Temptation to add mesh/slope in Phase 1. **Mitigation:** Hard boundary — skeleton only. Phase 2 is explicitly for mesh + scene.

**[Single-person assumption]** → Multi-person videos produce incorrect results. **Mitigation:** Document constraint. Add largest-bbox person selection as pre-filter.
