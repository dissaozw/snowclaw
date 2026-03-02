## Context

SnowCoach AI has a complete biomechanics engine (`packages/biomechanics/`) that computes skiing/snowboarding metrics from `Pose3D` objects, but no way to produce those objects from raw video. The ML pipeline defined in PLAN.md §3-4 has six stages; this change implements stages 1-3 (video preprocessing, 2D pose, 3D lifting) and wires them together.

The existing `biomechanics` package is pure NumPy with no ML dependencies. This boundary must be preserved — ML-heavy code lives in separate packages with isolated dependencies.

**Current state:** Video → _(gap)_ → `Pose3D` → Biomechanics metrics
**Target state:** Video → Frames → 2D Keypoints → 3D Pose (`Pose3D`) → Biomechanics metrics

## Goals / Non-Goals

**Goals:**
- Extract frames from video files at configurable FPS with resolution normalization
- Detect 2D human keypoints from frames using a pluggable model backend
- Lift 2D keypoint sequences to 3D with temporal consistency
- Output `biomechanics.Pose3D` objects directly consumable by the existing metrics engine
- Define abstract interfaces so model backends are swappable without changing downstream code
- Provide a single `Pipeline` entry point: `video path → SessionMetrics + Frame[]`

**Non-Goals:**
- SMPL-X mesh recovery (Phase 2)
- Scene reconstruction / depth estimation (Phase 2)
- Multi-person tracking or person re-identification
- Real-time / streaming inference (batch processing only for Phase 1)
- Model training or fine-tuning — we wrap pretrained checkpoints
- FastAPI backend or 3D viewer (separate changes)

## Decisions

### 1. Two separate packages, not one

**Decision:** `video-pipeline` and `pose-estimation` are distinct packages under `packages/`.

**Rationale:** `video-pipeline` has zero ML dependencies (just FFmpeg subprocess calls). Bundling it with `pose-estimation` would force PyTorch installation just to extract frames. This follows the existing pattern where `biomechanics` is pure NumPy.

**Alternative considered:** Single `ml-pipeline` package — rejected because it violates the minimal-deps-per-package convention and makes frame extraction unusable without a GPU stack.

### 2. FFmpeg via subprocess, not ffmpeg-python

**Decision:** Call FFmpeg directly via `subprocess.run` with typed wrapper functions rather than using the `ffmpeg-python` binding library.

**Rationale:** `ffmpeg-python` is poorly maintained (last release 2022) and adds an abstraction layer that obscures the actual FFmpeg commands being run. Direct subprocess calls are transparent, debuggable, and have zero additional dependencies. We provide our own thin typed wrappers.

**Alternative considered:** `ffmpeg-python` — rejected due to maintenance status. `PyAV` — rejected as overkill; we only need basic frame extraction, not per-packet manipulation.

### 3. Abstract base classes for model backends

**Decision:** Define `PoseEstimator2D` (ABC) and `PoseLifter3D` (ABC) interfaces. Ship with `ViTPoseBackend` and `MotionBERTBackend` as concrete implementations.

```
class PoseEstimator2D(ABC):
    @abstractmethod
    def predict(self, frames: list[np.ndarray]) -> list[Keypoints2D]: ...

class PoseLifter3D(ABC):
    @abstractmethod
    def lift(self, keypoints_2d: list[Keypoints2D]) -> list[Pose3D]: ...
```

**Rationale:** PLAN.md explicitly calls for "pluggable model backends" so the community can swap in newer models (RTMPose, Context-Aware PoseFormer, etc.) without modifying downstream code. ABCs enforce the contract at the type level.

**Alternative considered:** Duck typing with Protocols — workable but ABCs provide clearer error messages for implementors and are more conventional in this codebase.

### 4. Model weights via lazy download

**Decision:** Model weights are not bundled. On first use, each backend downloads its checkpoint to `~/.cache/snowclaw/models/` and caches it. A standalone `download_models.py` script pre-fetches all weights.

**Rationale:** Keeps the repo and package size small. The cache directory is user-local and survives reinstalls. Lazy download means users only fetch what they actually use.

### 5. ONNX Runtime as default inference engine

**Decision:** Ship ViTPose+ and MotionBERT as ONNX models for inference, with a PyTorch fallback.

**Rationale:** ONNX Runtime runs on CPU and GPU without requiring a full PyTorch install, dramatically reducing the default dependency footprint. Users who need PyTorch-specific features (custom layers, training) can opt into the `[torch]` extra.

**Alternative considered:** PyTorch-only — rejected because it's a 2GB+ install that blocks CPU-only users. TensorRT — too NVIDIA-specific for an open-source tool.

### 6. Intermediate data format: Keypoints2D

**Decision:** Introduce a lightweight `Keypoints2D` dataclass as the interface between 2D estimation and 3D lifting, separate from `Pose3D`.

```python
@dataclass
class Keypoints2D:
    points: np.ndarray      # shape (J, 2) — pixel coordinates
    confidence: np.ndarray   # shape (J,) — per-joint confidence [0,1]
    image_size: tuple[int, int]  # (height, width)
```

**Rationale:** 2D keypoints are in pixel space with variable joint counts depending on the model. `Pose3D` is in meters with a fixed joint set. Forcing 2D results into `Pose3D` would lose information and conflate coordinate systems. The 3D lifter handles the conversion.

## Risks / Trade-offs

**[Model accuracy on ski clothing]** → Pretrained ViTPose+ and MotionBERT have not been fine-tuned on winter sports. Bulky clothing and helmets may reduce joint detection accuracy. **Mitigation:** Confidence scores are propagated to downstream metrics, allowing low-confidence frames to be flagged or excluded. Fine-tuning is a Phase 3+ concern.

**[FFmpeg as system dependency]** → Requires FFmpeg installed on the host, which isn't pip-installable. **Mitigation:** Clear error message if `ffmpeg` is not found on PATH; document installation in README. Most ML environments and Docker images already include FFmpeg.

**[Large model downloads]** → ViTPose+ (~350MB) and MotionBERT (~150MB) must be downloaded. **Mitigation:** Lazy download with progress bar; `download_models.py` for CI/Docker pre-warming; models cached in `~/.cache/snowclaw/`.

**[ONNX conversion fidelity]** → Some model ops may not convert cleanly to ONNX. **Mitigation:** PyTorch fallback backend. Test numeric equivalence between ONNX and PyTorch outputs in CI.

**[Single-person assumption]** → Pipeline assumes one skier/snowboarder per video. Multi-person videos will produce incorrect results. **Mitigation:** Document the constraint. Add person detection + selection (largest bounding box) as a lightweight pre-filter in the video pipeline.
