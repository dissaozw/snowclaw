## Context

SnowCoach AI has a complete biomechanics engine (`packages/biomechanics/`) that computes skiing/snowboarding metrics from `Pose3D` objects, but no way to produce those objects from raw video. The ML pipeline defined in PLAN.md §3-4 has six stages; this change implements stages 1-3 (video preprocessing, 2D pose, 3D lifting) and wires them together.

Shared data structures (`Pose3D`, `Frame`, `Discipline`, etc.) currently live in `biomechanics.schemas`. This creates unwanted coupling — `pose-estimation` would need to import from `biomechanics` just to reference `Pose3D`, even though pose estimation has nothing to do with biomechanical analysis. The AGENTS.md repo structure already plans a `packages/core/` for "Shared types, configs, utilities." This change realizes that plan.

**Current state:** Video → _(gap)_ → `Pose3D` (defined in biomechanics) → Biomechanics metrics
**Target state:** Video → Frames → 2D Keypoints → `core.Pose3D` → _(any consumer: biomechanics, viewer, API, etc.)_

## Goals / Non-Goals

**Goals:**
- Create `packages/core/` with shared data structures that all packages can import independently
- Refactor `biomechanics` to import schemas from `core` (no logic changes)
- Extract frames from video files at configurable FPS with resolution normalization
- Detect 2D human keypoints from frames using a pluggable model backend
- Lift 2D keypoint sequences to 3D with temporal consistency
- Output `core.Pose3D` objects — consumable by biomechanics, viewer, API, or any other package
- Define abstract interfaces so model backends are swappable without changing downstream code
- Provide a single `Pipeline` entry point: `video path → list[core.Pose3D]`

**Non-Goals:**
- SMPL-X mesh recovery (Phase 2)
- Scene reconstruction / depth estimation (Phase 2)
- Multi-person tracking or person re-identification
- Real-time / streaming inference (batch processing only for Phase 1)
- Model training or fine-tuning — we wrap pretrained checkpoints
- FastAPI backend or 3D viewer (separate changes)
- Baking biomechanical analysis into the pipeline — consumers compose it themselves

## Decisions

### 1. Shared types in `packages/core/`, not in `biomechanics`

**Decision:** Extract `Pose3D`, `Frame`, `Discipline`, `TurnPhaseLabel`, `TurnPhase`, `SessionMetrics` into `packages/core/schemas.py`. The `biomechanics` package re-exports them from `core` for backward compatibility, then drops its own definitions.

**Rationale:** The dependency graph must flow downward: `core` ← `biomechanics`, `core` ← `pose-estimation`, `core` ← `api`, etc. No package should depend on `biomechanics` just to get `Pose3D`. AGENTS.md already lists `core` as "Shared types, configs, utilities."

**Alternative considered:** Keep schemas in biomechanics and have pose-estimation depend on biomechanics — rejected because it creates circular conceptual coupling (pose estimation is upstream of biomechanics in the pipeline, not downstream).

### 2. Three separate packages, not one

**Decision:** `core`, `video-pipeline`, and `pose-estimation` are distinct packages under `packages/`.

**Rationale:** `video-pipeline` has zero Python dependencies (just FFmpeg subprocess calls). `pose-estimation` needs ONNX/torch. `core` needs only numpy + pydantic. Bundling them would force unnecessary dependencies. This follows the existing pattern where `biomechanics` is pure NumPy.

**Alternative considered:** Single `ml-pipeline` package — rejected because it violates the minimal-deps-per-package convention.

### 3. FFmpeg via subprocess, not ffmpeg-python

**Decision:** Call FFmpeg directly via `subprocess.run` with typed wrapper functions rather than using the `ffmpeg-python` binding library.

**Rationale:** `ffmpeg-python` is poorly maintained (last release 2022) and adds an abstraction layer that obscures the actual FFmpeg commands being run. Direct subprocess calls are transparent, debuggable, and have zero additional dependencies.

**Alternative considered:** `ffmpeg-python` — rejected due to maintenance status. `PyAV` — rejected as overkill; we only need basic frame extraction.

### 4. Abstract base classes for model backends

**Decision:** Define `PoseEstimator2D` (ABC) and `PoseLifter3D` (ABC) interfaces. Ship with `ViTPoseBackend` and `MotionBERTBackend` as concrete implementations.

```
class PoseEstimator2D(ABC):
    @abstractmethod
    def predict(self, frames: list[np.ndarray]) -> list[Keypoints2D]: ...

class PoseLifter3D(ABC):
    @abstractmethod
    def lift(self, keypoints_2d: list[Keypoints2D]) -> list[Pose3D]: ...
```

Where `Pose3D` is imported from `core`, not `biomechanics`.

**Rationale:** PLAN.md explicitly calls for "pluggable model backends" so the community can swap in newer models (RTMPose, Context-Aware PoseFormer, etc.) without modifying downstream code.

### 5. Model weights via lazy download

**Decision:** Model weights are not bundled. On first use, each backend downloads its checkpoint to `~/.cache/snowclaw/models/` and caches it. A standalone `download_models.py` script pre-fetches all weights.

**Rationale:** Keeps the repo and package size small. Lazy download means users only fetch what they actually use.

### 6. ONNX Runtime as default inference engine

**Decision:** Ship ViTPose+ and MotionBERT as ONNX models for inference, with a PyTorch fallback.

**Rationale:** ONNX Runtime runs on CPU and GPU without requiring a full PyTorch install, dramatically reducing the default dependency footprint. Users who need PyTorch-specific features can opt into the `[torch]` extra.

### 7. Intermediate data format: Keypoints2D

**Decision:** Introduce a lightweight `Keypoints2D` dataclass in `pose-estimation` as the interface between 2D estimation and 3D lifting.

```python
@dataclass
class Keypoints2D:
    points: np.ndarray      # shape (J, 2) — pixel coordinates
    confidence: np.ndarray   # shape (J,) — per-joint confidence [0,1]
    image_size: tuple[int, int]  # (height, width)
```

**Rationale:** 2D keypoints are in pixel space with variable joint counts depending on the model. `Pose3D` is in meters with a fixed joint set. The 3D lifter handles the conversion. `Keypoints2D` is internal to `pose-estimation` — not shared via `core` — since no other package needs raw 2D keypoints.

### 8. Pipeline does NOT include biomechanics

**Decision:** The `Pipeline` class chains video → 2D pose → 3D pose and returns `list[core.Pose3D]`. It does not invoke biomechanical analysis.

**Rationale:** The pipeline's job is to produce 3D poses from video. Biomechanical analysis is a separate concern — a consumer composes it: `poses = pipeline.run(video); metrics = biomechanics.compute(poses)`. This keeps the pipeline independent and reusable for non-biomechanics use cases (e.g., 3D viewer rendering, pose export).

## Risks / Trade-offs

**[Breaking biomechanics imports]** → Moving schemas from `biomechanics.schemas` to `core.schemas` changes import paths. **Mitigation:** `biomechanics` re-exports all moved types from its `__init__.py` for backward compatibility. Downstream code using `from biomechanics import Pose3D` continues to work.

**[Model accuracy on ski clothing]** → Pretrained ViTPose+ and MotionBERT have not been fine-tuned on winter sports. **Mitigation:** Confidence scores are propagated to allow low-confidence frames to be flagged. Fine-tuning is a later concern.

**[FFmpeg as system dependency]** → Requires FFmpeg installed on the host. **Mitigation:** Clear error message if not found; most ML environments already include FFmpeg.

**[Large model downloads]** → ViTPose+ (~350MB) and MotionBERT (~150MB). **Mitigation:** Lazy download with progress bar; cache in `~/.cache/snowclaw/`.

**[Single-person assumption]** → Pipeline assumes one skier/snowboarder per video. **Mitigation:** Document the constraint; add person detection + largest-bbox selection as a lightweight pre-filter.
