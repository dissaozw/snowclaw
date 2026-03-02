# SnowCoach AI — Open-Source AI Skiing & Snowboarding Instructor

## Technical Project Plan v1.0

-----

## 1. Executive Summary

SnowCoach AI is an open-source platform that transforms amateur ski/snowboard video into actionable coaching feedback. Unlike existing products (SkiPro AI, Carv, PisteAI), SnowCoach AI differentiates on four axes:

1. **Video-to-3D reconstruction** — pause any frame and orbit the skier/snowboarder in full 3D, zoom into ski-snow contact, edge angles, etc.
1. **State-of-the-art motion tracking** — leveraging the latest transformer-based 3D human pose estimation (BioPose, Context-Aware PoseFormer, SMPL-X mesh recovery) instead of legacy OpenPose/MediaPipe skeletons.
1. **Systematic progression engine** — built on the CSIA (Canadian Ski Instructors' Alliance) 4-level framework and CASI (Canadian Association of Snowboard Instructors) equivalents, tracking improvement over weeks/months rather than one-shot reports.
1. **Fully open-source** — MIT-licensed core, community-extensible skill frameworks, and pluggable model backends.

-----

## 2. Competitive Landscape & Gap Analysis

|Feature             |SkiPro AI                         |Carv        |PisteAI     |SnowCoach AI (ours)               |
|--------------------|----------------------------------|------------|------------|----------------------------------|
|Input               |Video (2D overlay)                |Boot sensors|Video       |Video                             |
|3D Viewing          |✗                                 |✗           |✗           |✓ Full 3D orbit                   |
|Snowboard           |✗                                 |✗           |Partial     |✓ Native                          |
|Pose Model          |Legacy (likely OpenPose/AlphaPose)|N/A (sensor)|Unknown     |SOTA transformer HPE              |
|Progression Tracking|One-shot report                   |Ski IQ score|Basic drills|CSIA/CASI-aligned progression plan|
|Framework           |Proprietary                       |Proprietary |Proprietary |Open-source (MIT)                 |
|Offline             |✗                                 |✓           |✗           |✓ (local inference option)        |

### Key Gaps in Existing Products

**SkiPro AI** (from the screenshots you shared): Uses 2D skeleton overlay with metrics like inclination angle, core stability, COM height, and a "champion match" percentage. Their limitations include flat 2D overlays with no ability to rotate viewpoint, what appears to be a legacy pose model (visible jitter in joint tracking), and reports that are single-session snapshots without longitudinal progression.

**Carv**: Hardware-dependent (boot sensor insoles), subscription model ($119/season), skiing-only, and provides a "Ski IQ" score without structured teaching framework alignment.

**PisteAI**: Still in beta, video-based with drills, but no 3D reconstruction and unclear underlying pose estimation quality.

-----

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Web App      │  │  Mobile App  │  │  3D Viewer (Three.js)│  │
│  │  (React/Next) │  │  (React      │  │  - Orbit controls    │  │
│  │              │  │   Native)    │  │  - Frame scrubbing   │  │
│  │              │  │              │  │  - Zoom to ski/board  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         └─────────────────┼─────────────────────┘              │
└───────────────────────────┼─────────────────────────────────────┘
                            │ REST / WebSocket / gRPC
┌───────────────────────────┼─────────────────────────────────────┐
│                     API GATEWAY (FastAPI)                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐│
│  │ Video      │  │ Analysis   │  │ Progression│  │ Report    ││
│  │ Upload     │  │ Pipeline   │  │ Engine     │  │ Generator ││
│  │ Service    │  │ Orchestr.  │  │            │  │           ││
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘│
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                    ML PIPELINE LAYER                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Stage 1: Video Preprocessing                           │    │
│  │  - Scene stabilization (background subtraction)         │    │
│  │  - Skier/snowboarder detection & tracking (ByteTrack)   │    │
│  │  - Frame sampling & keyframe extraction                 │    │
│  └───────────────────────┬─────────────────────────────────┘    │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Stage 2: 2D Pose Estimation                            │    │
│  │  - Primary: ViTPose+ / RTMPose (SOTA 2D detectors)     │    │
│  │  - Sport-specific keypoints (ski tips, edges, poles)    │    │
│  │  - Confidence scoring per joint                         │    │
│  └───────────────────────┬─────────────────────────────────┘    │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Stage 3: 3D Pose Lifting + Mesh Recovery               │    │
│  │  - MotionBERT / Context-Aware PoseFormer (2D → 3D)     │    │
│  │  - SMPL-X mesh recovery for full body shape             │    │
│  │  - BioPose NeurIK for biomechanical joint constraints   │    │
│  │  - Temporal smoothing (Savitzky-Golay / Kalman filter)  │    │
│  └───────────────────────┬─────────────────────────────────┘    │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Stage 4: 3D Scene Reconstruction                       │    │
│  │  - Monocular depth estimation (Depth Anything V2)       │    │
│  │  - Snow surface plane fitting (RANSAC)                  │    │
│  │  - Equipment detection (ski/snowboard segmentation)     │    │
│  │  - Gaussian Splatting for background (optional)         │    │
│  └───────────────────────┬─────────────────────────────────┘    │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Stage 5: Biomechanical Analysis Engine                 │    │
│  │  - Edge angle computation                               │    │
│  │  - Inclination / angulation decomposition               │    │
│  │  - Center of mass trajectory                            │    │
│  │  - Rotary / pressure / balance skill metrics            │    │
│  │  - Turn phase segmentation (initiation → fall line →    │    │
│  │    completion)                                          │    │
│  └───────────────────────┬─────────────────────────────────┘    │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Stage 6: LLM Coaching Engine                           │    │
│  │  - CSIA/CASI framework knowledge base (RAG)             │    │
│  │  - Biomechanical metrics → natural language feedback     │    │
│  │  - Session comparison & progression reasoning           │    │
│  │  - Drill recommendation                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                      DATA LAYER                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐│
│  │ PostgreSQL │  │ Object     │  │ Vector DB  │  │ Redis     ││
│  │ (users,    │  │ Storage    │  │ (CSIA/CASI │  │ (job      ││
│  │  sessions, │  │ (S3/MinIO) │  │  knowledge │  │  queue,   ││
│  │  metrics)  │  │ (videos,   │  │  embeddings│  │  cache)   ││
│  │            │  │  3D assets)│  │  )         │  │           ││
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘│
└──────────────────────────────────────────────────────────────────┘
```

-----

## 4. Core Technical Components

### 4.1 Video-to-3D Pipeline (Key Differentiator)

The goal: a user uploads a single monocular ski video, and the system produces a navigable 3D scene where they can orbit around themselves at any frame.

**Approach — Hybrid Mesh + Scene Reconstruction:**

1. **Human mesh recovery**: Use SMPL-X (or SMPL+H for hands) to recover a parametric 3D body mesh per frame. The latest approach, BioPose (Jan 2025), adds biomechanical accuracy via Neural Inverse Kinematics — critical for analyzing joint angles in skiing.
1. **Equipment modeling**: Train or fine-tune a segmentation model (SAM 2) to detect skis, snowboards, poles, and boots. Map detected equipment onto the SMPL mesh as rigid attachments using known boot-binding geometry.
1. **Scene reconstruction**: Use Depth Anything V2 for monocular depth, then fit a snow surface plane via RANSAC. Optionally use 3D Gaussian Splatting (e.g., gsplat) for photorealistic background reconstruction from the video frames.
1. **3D Viewer**: Three.js or React Three Fiber in the browser. The user scrubs a timeline, and for each frame:
- The SMPL mesh (with equipment) is rendered
- The snow plane and approximate scene geometry provide context
- Orbit controls allow full 360° rotation
- Overlay panels show real-time metrics (edge angle, COM, inclination)

**Challenges & Mitigations:**

|Challenge                             |Mitigation                                                                                                            |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------|
|Depth ambiguity from monocular video  |Temporal consistency via MotionBERT; snow plane as a ground-truth reference surface                                   |
|Bulky ski clothing obscures body shape|Fine-tune SMPL-X on winter sports data; use silhouette + joint constraints                                            |
|Fast motion blur                      |Keyframe selection + optical flow interpolation                                                                       |
|Ski/board occlusion by snow spray     |Temporal interpolation of equipment pose from visible frames                                                          |
|Single viewpoint limitation           |Clearly communicate to users that the 3D is a reconstruction, not a true multi-view capture; show confidence heat maps|

### 4.2 Motion Tracking — Model Selection

We propose a **pluggable model backend** so the community can swap in newer models as they emerge:

**2D Pose Detection (Stage 2):**

- **Primary: ViTPose+** — Vision Transformer-based, SOTA on COCO/MPII, robust to occlusion
- **Alternative: RTMPose** — faster inference, suitable for mobile/edge deployment
- **Sport-specific fine-tuning**: Add custom keypoints for ski tips, tail, bindings, pole grips, and snowboard nose/tail/edges. This requires collecting and annotating a ski/snowboard-specific dataset (see Section 6).

**3D Pose Lifting (Stage 3):**

- **Primary: MotionBERT** — temporal transformer that lifts 2D sequences to 3D with strong temporal consistency
- **Alternative: Context-Aware PoseFormer** — single-frame but context-rich, good for keyframe analysis
- **Biomechanical refinement: BioPose NeurIK** — enforces anatomical joint constraints (critical for knee/hip angles in skiing)

**Mesh Recovery (Stage 3b):**

- **SMPL-X via HMR 2.0 or TokenHMR** — full body mesh from single image
- Fine-tune on sports action datasets (e.g., SportsPose, FIT3D) for better generalization to athletic poses

### 4.3 Biomechanical Analysis Engine

Derived metrics computed from 3D pose per frame:

**Skiing Metrics (CSIA Five Skills: Rotary, Edging, Balance, Pressure, Coordination):**

|Metric                     |Computation                                                       |CSIA Skill      |
|---------------------------|------------------------------------------------------------------|----------------|
|Edge Angle                 |Angle between ski plane normal and snow surface normal            |Edging          |
|Inclination Angle          |Angle of body's long axis relative to vertical                    |Balance         |
|Angulation                 |Lateral separation between upper and lower body planes            |Edging + Balance|
|Knee Flex Angle            |3D angle at knee joint (femur-tibia)                              |Pressure        |
|Hip Flex Angle             |3D angle at hip joint                                             |Pressure        |
|Fore-aft Balance           |COM projection relative to boot midpoint along ski axis           |Balance         |
|Lateral Balance            |COM projection relative to boot midpoint perpendicular to ski axis|Balance         |
|Upper/Lower Body Separation|Rotation difference between shoulder plane and hip plane          |Rotary          |
|Pole Plant Timing          |Temporal alignment of pole contact with turn phase                |Coordination    |
|Turn Shape Symmetry        |Radius consistency between left and right turns                   |Coordination    |
|COM Height %               |Vertical oscillation of center of mass through turn phases        |Pressure        |
|Speed & Turn Radius        |Estimated from camera motion + scene geometry                     |(Global)        |

**Snowboard Metrics (CASI Framework):**

|Metric              |Computation                                            |CASI Skill|
|--------------------|-------------------------------------------------------|----------|
|Board Tilt Angle    |Angle between board plane and snow surface             |Edging    |
|Flex/Extension      |Knee and ankle angles through turn                     |Pressure  |
|Upper Body Alignment|Shoulder-to-board angle (should remain relatively open)|Rotary    |
|Weight Distribution |Fore-aft COM relative to board center                  |Balance   |
|Counter-Rotation    |Upper vs lower body rotation differential              |Rotary    |

### 4.4 Progression Engine

This is the **second key differentiator** — instead of one-shot reports, SnowCoach AI builds a longitudinal skill profile.

**Data Model:**

```
User
├── SkillProfile
│   ├── discipline: ski | snowboard
│   ├── current_level: CSIA L1-L4 equivalent / CASI L1-L4 equivalent
│   ├── skill_scores: { rotary: 72, edging: 65, balance: 81, pressure: 58, coordination: 70 }
│   └── progression_history: [...]
├── Sessions[]
│   ├── video_id
│   ├── date, resort, conditions
│   ├── runs[]
│   │   ├── metrics_per_frame[]
│   │   ├── aggregate_metrics
│   │   └── turn_analysis[]
│   └── session_report
└── LearningPlan
    ├── current_focus: "Increase edge angle in toe-side turns"
    ├── drills: [{ name, description, video_ref, target_metric }]
    ├── milestones: [{ metric, target_value, deadline }]
    └── history: [{ plan_version, date, focus_area, outcome }]
```

**Progression Logic:**

1. After each session, compare aggregate metrics against the user's historical baseline and their target level's benchmarks.
1. Identify the **weakest CSIA/CASI skill** that is most impactful for the user's current level progression.
1. Generate a focused drill set (3-5 drills) targeting that skill, sourced from the CSIA/CASI knowledge base.
1. Track drill effectiveness across sessions — if a metric improves, advance the plan; if stagnant, suggest alternative approaches.
1. When all five skills reach the benchmark for the next level, recommend the user for level-up and shift focus.

**LLM Integration for Coaching Reports:**

Use RAG (Retrieval Augmented Generation) with a vector database containing:

- CSIA Technical Manual content (skills, progressions, common errors)
- CASI Technical Manual content
- Curated drill library with video references
- Anonymized aggregate data from expert skiers/snowboarders (as reference benchmarks)

The LLM generates natural-language reports like:

> *"Session Summary — March 1, 2026, Whistler Blackcomb"*
>
> *Your edge angles have improved 12% since last session (avg 48° → 54°), moving you closer to the Level 3 benchmark of 60°+. Your biggest opportunity is fore-aft balance: your COM shifts 8cm rearward during turn initiation, which limits your ability to engage the ski tip early. This is the #1 factor holding back your turn shape quality.*
>
> *This week's focus: Drill 1 — "Javelin Turns" (ski on outside ski only, inside ski lifted). This will force you to commit your weight forward…*

### 4.5 SnowIQ Score (Gamification)

Inspired by Carv's "Ski IQ" but grounded in the CSIA/CASI framework:

```
SnowIQ = weighted_average(
    rotary_score      × 0.20,
    edging_score      × 0.25,
    balance_score     × 0.25,
    pressure_score    × 0.15,
    coordination_score × 0.15
)
```

Mapped to a 0-200 scale:

- 0-50: Beginner (CSIA L1 student level)
- 51-100: Intermediate (CSIA L1 instructor standard)
- 101-140: Advanced (CSIA L2 standard)
- 141-170: Expert (CSIA L3 standard)
- 171-200: Elite (CSIA L4 / racing standard)

-----

## 5. Technology Stack

|Layer            |Technology                           |Rationale                               |
|-----------------|-------------------------------------|----------------------------------------|
|Frontend — Web   |Next.js 14 + React Three Fiber       |SSR, great DX, Three.js integration     |
|Frontend — Mobile|React Native + expo-three            |Cross-platform, shared 3D viewer code   |
|3D Viewer        |Three.js / React Three Fiber         |Industry standard, WebGL, orbit controls|
|Backend API      |FastAPI (Python)                     |Async, ML-ecosystem native, auto-docs   |
|ML Serving       |Triton Inference Server or TorchServe|Batched GPU inference, model versioning |
|Job Queue        |Celery + Redis                       |Async video processing pipeline         |
|Database         |PostgreSQL + TimescaleDB             |Relational + time-series metrics        |
|Vector DB        |Qdrant or Pgvector                   |RAG for CSIA/CASI knowledge base        |
|Object Storage   |MinIO (self-hosted) / S3             |Videos, 3D assets, SMPL meshes          |
|ML Framework     |PyTorch 2.x                          |SOTA model ecosystem                    |
|3D Formats       |glTF 2.0 (.glb)                      |Web-standard, compact, animatable       |
|CI/CD            |GitHub Actions                       |OSS-native                              |
|Containerization |Docker + Docker Compose              |Reproducible dev/deploy                 |
|GPU Infra        |NVIDIA CUDA, optional Apple MPS      |Training + inference                    |

-----

## 6. Data Strategy

### 6.1 Training Data Needed

|Dataset                   |Purpose                                        |Source Strategy                                                                         |
|--------------------------|-----------------------------------------------|----------------------------------------------------------------------------------------|
|Ski/Snowboard Pose Dataset|Fine-tune 2D + 3D pose models for winter sports|Community annotation campaign + synthetic data (BEDLAM-style rendering with ski avatars)|
|Equipment Keypoint Dataset|Detect ski tips/tails, snowboard edges, poles  |Manual annotation of ~5K frames from YouTube ski instruction videos                     |
|Expert Reference Runs     |Benchmark metrics for each CSIA/CASI level     |Partner with ski schools; collect consented video of L1-L4 certified instructors        |
|CSIA/CASI Knowledge Base  |RAG corpus for coaching engine                 |Digitize official manuals (with licensing); supplement with open educational content    |

### 6.2 Synthetic Data Pipeline

To bootstrap without massive real-world annotation:

1. Use AMASS motion capture database (contains skiing-like motions)
1. Render synthetic skiers in Blender/Unity with varied clothing, equipment, slopes, and lighting using BEDLAM-style pipeline
1. Generate perfect 3D ground truth + 2D projections for training
1. Domain adaptation via style transfer to bridge the sim-to-real gap

### 6.3 Community Data Contribution

Open-source annotation tool (Label Studio or CVAT instance) where community members can:

- Upload their ski/snowboard videos (with consent)
- Annotate equipment keypoints
- Self-label their skill level (validated by certified instructors in the community)

-----

## 7. Development Roadmap

> **Methodology**: Solo/small-team development with Claude Code as primary implementation partner. Claude Code handles boilerplate, model integration plumbing, frontend scaffolding, tests, and docs. Human focuses on ML architecture decisions, ski domain expertise, and UX design. Estimated ~60-70% of raw code output via Claude Code.

### Phase 1 — Week 1-2: "Video In, Skeleton Out"

**Goal:** Two outputs from a single ski/snowboard video: (1) an annotated video with skeleton overlay and metrics, and (2) a simple 3D skeleton viewer with orbit controls.

|Day  |Task                                                                              |Claude Code Role                                                 |
|-----|----------------------------------------------------------------------------------|-----------------------------------------------------------------|
|1-2  |Shared `core` package (Pose3D, Frame, enums), FFmpeg video-pipeline               |Scaffold packages, extract schemas from biomechanics             |
|3-4  |Integrate ViTPose+ (2D) + MotionBERT (2D→3D lifting) with pluggable backends     |Write model wrapper classes, download scripts, inference pipeline|
|5-7  |Video annotation renderer: draw skeleton, plumb line, metrics on each frame       |OpenCV/Pillow drawing, FFmpeg re-encode to output video          |
|8-10 |Simple Three.js viewer: skeleton render, orbit controls, frame scrubber           |React Three Fiber component — skeleton only, no mesh/slope       |
|11-14|FastAPI backend: video upload → async pipeline → annotated video + 3D data output |API routes, Celery tasks, WebSocket status updates               |

**Deliverables:**
1. **Annotated video** — skeleton overlay with joint dots, connecting lines, plumb line (COM), and basic metric numbers (knee angle, inclination, COM height). Like SkiPro AI but with transformer-based accuracy (ViTPose+) and temporal smoothing.
2. **Simple 3D viewer** — orbitable skeleton in the browser with frame scrubbing. Skeleton only — no body mesh, no snow surface, no ski geometry (those are Phase 2).

### Phase 2 — Week 3-4: "Full Mesh + Snow Scene"

**Goal:** SMPL-X body mesh, equipment detection, snow surface reconstruction, snowboard support. Upgrade the simple skeleton viewer into a full 3D scene.

|Day  |Task                                                                            |Claude Code Role                                   |
|-----|--------------------------------------------------------------------------------|---------------------------------------------------|
|1-3  |SMPL-X mesh recovery (HMR 2.0), glTF animated export                            |Model integration, mesh-to-glTF conversion pipeline|
|4-5  |Depth Anything V2 → snow surface plane fitting (RANSAC)                         |Depth pipeline + plane extraction code             |
|6-7  |SAM 2 equipment segmentation (ski/board/poles), mesh attachment                 |Fine-tune script scaffolding, attachment geometry  |
|8-10 |Snowboard mode: adapted skeleton mapping, board-specific metrics                |Extend biomechanics module, UI toggle              |
|11-14|Enhanced viewer: body mesh, snow slope, ski/board geometry, measurement tools    |Three.js enhancements, UI polish                   |

**Deliverable:** Full 3D scene — body mesh with detected ski/board geometry on a depth-estimated snow surface. Users can see how their ski edges interact with the snow.

### Phase 3 — Week 5-6: "AI Coach"

**Goal:** CSIA/CASI-grounded analysis, LLM coaching, reports.

|Day  |Task                                                                                     |Claude Code Role                                    |
|-----|-----------------------------------------------------------------------------------------|----------------------------------------------------|
|1-3  |Full biomechanical engine: all metrics from Section 4.3, turn phase segmentation         |Implement metric computations, signal processing    |
|4-5  |CSIA/CASI knowledge base: chunk + embed manuals into Qdrant/pgvector                     |RAG pipeline, embedding scripts, retrieval API      |
|6-8  |LLM coaching engine: metrics → structured prompt → Claude API → natural language report  |Prompt engineering, report template, API integration|
|9-10 |SnowIQ scoring + session report (web + PDF export)                                       |Score computation, PDF generation, dashboard UI     |
|11-14|Progression tracking: multi-session comparison, skill trend charts, drill recommendations|DB schema, comparison logic, Recharts dashboards    |

**Deliverable:** Upload video → full coaching report with SnowIQ, drill plan, session-over-session trends.

### Phase 4 — Week 7-8: "Ship It"

**Goal:** Polish, mobile, open-source launch.

|Day  |Task                                                                                                 |Claude Code Role                           |
|-----|-----------------------------------------------------------------------------------------------------|-------------------------------------------|
|1-3  |Adaptive learning plan engine (auto-updates focus per session)                                       |Progression logic, plan generation         |
|4-5  |Model optimization: ONNX export, INT8 quantization, batch inference tuning                           |Conversion scripts, benchmarking           |
|6-8  |React Native app: video capture + upload + report viewer                                             |Scaffold full mobile app                   |
|9-10 |Open-source prep: CONTRIBUTING.md, architecture docs, API docs, sample data                          |Generate all docs, README, developer guides|
|11-14|Community infra: Label Studio annotation setup, GitHub issue templates, first-time contributor issues|Config files, templates, onboarding scripts|

**Deliverable:** Public launch — GitHub repo, web app, mobile app, documentation.

### Total: ~8 weeks (2 months) to MVP launch

**Comparison:** Traditional estimate was 12 months. Claude Code compresses this ~6x by eliminating:

- Boilerplate/scaffolding time (~40% of traditional dev)
- Research-to-code translation for model integrations (~20%)
- Frontend component development (~15%)
- Documentation and test writing (~15%)

**What still takes human time:**

- ML architecture decisions and model evaluation
- Ski domain expertise for metric calibration
- UX design and user testing with real skiers
- Dataset curation and annotation quality control
- Partnership outreach (CSIA/CASI, ski schools)

### Post-Launch (Month 3+)

- Real-time AR feedback (Apple Vision Pro / Meta Orion)
- Multi-person group lesson analysis
- Race line optimization module
- Terrain park / freestyle module
- Federated learning for privacy-preserving model improvement
- Expert marketplace (connect users with certified instructors)

-----

## 8. Open Source Strategy

### 8.1 Repository Structure

```
snowcoach-ai/
├── README.md
├── LICENSE (MIT)
├── CONTRIBUTING.md
├── docs/
│   ├── architecture.md
│   ├── model-guide.md
│   └── skill-frameworks/
│       ├── csia-skiing.md
│       └── casi-snowboarding.md
├── packages/
│   ├── core/                  # Shared types, configs, utils
│   ├── video-pipeline/        # FFmpeg preprocessing, frame extraction
│   ├── pose-estimation/       # 2D + 3D pose model wrappers
│   ├── mesh-recovery/         # SMPL-X integration
│   ├── scene-reconstruction/  # Depth estimation, snow plane fitting
│   ├── biomechanics/          # Metric computation engine
│   ├── coaching-engine/       # LLM + RAG + progression logic
│   ├── viewer-3d/             # React Three Fiber component library
│   ├── api/                   # FastAPI backend
│   ├── web/                   # Next.js frontend
│   └── mobile/                # React Native app
├── models/                    # Model configs, weights download scripts
├── data/
│   ├── annotations/           # Community annotation schemas
│   └── benchmarks/            # Reference metric datasets
├── scripts/
│   ├── train/                 # Fine-tuning scripts
│   └── evaluate/              # Model evaluation
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.ml
│   └── docker-compose.yml
└── tests/
```

### 8.2 Contribution Areas

|Area                  |Skill Level    |Description                                      |
|----------------------|---------------|-------------------------------------------------|
|Pose model fine-tuning|Advanced ML    |Train on ski/snowboard data                      |
|Equipment detection   |Intermediate ML|Annotate + train segmentation                    |
|Biomechanics metrics  |Domain expert  |Implement new skiing/snowboarding metrics        |
|Skill frameworks      |Ski instructor |Add PSIA, BASI, ÖSV, or other national frameworks|
|3D viewer features    |Frontend       |New visualization modes, measurement tools       |
|Mobile app            |Mobile dev     |React Native development                         |
|Documentation         |Any level      |Tutorials, translations, guides                  |
|Dataset annotation    |Beginner       |Label ski videos with keypoints                  |

### 8.3 Community Governance

- **Core maintainers**: 3-5 people with merge rights, rotating monthly lead
- **Ski/Snowboard Advisory Board**: Certified instructors (CSIA L3+, CASI L3+, PSIA L3+) who review the coaching engine's accuracy
- **RFC process** for major architectural changes
- **Monthly community calls** for roadmap discussion

-----

## 9. Key Technical Risks

|Risk                                                              |Likelihood|Impact|Mitigation                                                                                                                     |
|------------------------------------------------------------------|----------|------|-------------------------------------------------------------------------------------------------------------------------------|
|3D reconstruction quality insufficient from single monocular video|High      |High  |Clearly communicate limitations; use confidence scores; recommend filming guidelines (side angle, steady camera, good lighting)|
|Model inference too slow for good UX                              |Medium    |High  |Async processing with progress bar; offer tiered quality (fast skeleton vs. full mesh); GPU batching                           |
|Ski/snowboard-specific pose estimation data scarcity              |High      |Medium|Synthetic data pipeline; active learning with community annotation; transfer learning from general sports pose datasets        |
|CSIA/CASI framework licensing for knowledge base                  |Medium    |Medium|Partner officially with CSIA/CASI; fallback to open educational content + original instructional writing                       |
|GPU cost for self-hosted inference                                |Medium    |Medium|ONNX + INT8 quantization for CPU inference; offer local inference mode; community-hosted GPU instances                         |
|Open-source sustainability                                        |Medium    |Medium|Sponsorship model (GitHub Sponsors, Open Collective); optional hosted tier for revenue                                         |

-----

## 10. Recommended Filming Guidelines (for Users)

To maximize analysis quality, the system should guide users to:

1. **Camera angle**: Side view (perpendicular to fall line), capturing full body from head to ski tips
1. **Camera stability**: Use a tripod or have a stationary filmer; panning is acceptable if smooth
1. **Distance**: 5-15 meters from skier — close enough to see joint detail, far enough for full body
1. **Lighting**: Avoid filming directly into sun; overcast days produce best results (even lighting)
1. **Duration**: 3-5 linked turns minimum per run analysis
1. **Resolution**: 1080p minimum, 4K preferred; 60fps strongly recommended for fast skiing
1. **Clothing**: Form-fitting layers improve pose estimation (baggy jackets reduce accuracy)

-----

## 11. Success Metrics

|Metric                         |Target (v1.0)                                      |Measurement                                 |
|-------------------------------|---------------------------------------------------|--------------------------------------------|
|3D Pose MPJPE                  |< 50mm on ski-specific test set                    |Mean Per-Joint Position Error               |
|Edge Angle Accuracy            |±5° vs. manual expert annotation                   |Compared against certified instructor labels|
|User Skill Level Classification|80%+ agreement with certified instructor assessment|Blind evaluation by CSIA/CASI instructors   |
|Progression Plan Relevance     |4.0+ / 5.0 user rating                             |Post-session survey                         |
|Video Processing Time          |< 5 min for 60s video (GPU)                        |End-to-end pipeline benchmark               |
|GitHub Stars                   |1,000+ in first 6 months                           |Community adoption signal                   |
|Active Contributors            |20+ per month                                      |GitHub Insights                             |

-----

## 12. Getting Started (for Contributors)

```bash
# Clone the repository
git clone https://github.com/snowcoach-ai/snowcoach.git
cd snowcoach

# Start development environment
docker compose up -d

# Download pretrained models
python scripts/download_models.py

# Run the pipeline on a sample video
python -m packages.api.cli analyze --video samples/ski_demo.mp4

# Start the web viewer
cd packages/web && npm install && npm run dev
```

-----

## Appendix A: Key Research Papers

|Paper                                                         |Year|Relevance                           |
|--------------------------------------------------------------|----|------------------------------------|
|BioPose: Biomechanically-accurate 3D Pose from Monocular Video|2025|Biomechanical accuracy via NeurIK   |
|MotionBERT: Unified Pretraining for Human Motion Analysis     |2023|Temporal 2D→3D lifting              |
|Context-Aware PoseFormer                                      |2024|Single-frame 3D with spatial context|
|ViTPose+: Vision Transformer for Generic Body Pose Estimation |2023|SOTA 2D pose detection              |
|SMPL-X: Expressive Body Capture                               |2019|Parametric body model               |
|Depth Anything V2                                             |2024|Monocular depth estimation          |
|SAM 2: Segment Anything in Images and Videos                  |2024|Equipment segmentation              |
|3D Gaussian Splatting                                         |2023|Scene reconstruction                |
|BEDLAM: Synthetic Training Data for HPE                       |2023|Synthetic data pipeline reference   |

## Appendix B: CSIA Five Skills Reference

The CSIA teaching framework identifies five fundamental skiing skills that form the basis of all ski technique:

1. **Rotary**: The turning or pivoting of the body and skis. Includes leg rotation, upper/lower body separation, and steering.
1. **Edging**: The tilting of the skis on their edges to grip the snow. Involves inclination (whole body lean) and angulation (lateral body bend).
1. **Balance**: The ability to maintain equilibrium. Encompasses fore-aft balance, lateral balance, and dynamic balance through movement.
1. **Pressure**: The management of forces through flexion/extension. Controls ski-snow contact pressure and weight distribution.
1. **Coordination**: The timing and blending of all skills. Determines turn quality, rhythm, and flow.

Each skill is developed progressively from Level 1 (basic) through Level 4 (expert), with increasing precision, range, and versatility required at each stage.
