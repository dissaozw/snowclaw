# AGENTS.md — Context for AI Agents

## Project

SnowCoach AI (`snowclaw`) — open-source AI skiing & snowboarding instructor. Transforms monocular ski/snowboard video into 3D reconstructions and coaching feedback aligned to the CSIA/CASI teaching frameworks.

## Key Files

- `PLAN.md` — Full technical plan: architecture, ML pipeline stages, roadmap, and domain details. **Read this first** for any non-trivial task.
- `pyproject.toml` — Python project config (hatchling build, numpy/pydantic/scipy deps, pytest config).
- `README.md` — User-facing project overview.

## Repo Structure

```
snowclaw/
├── packages/
│   ├── biomechanics/       # ✅ DONE — Metrics, SnowIQ, turn segmentation, schemas
│   ├── video-pipeline/     # 🔲 FFmpeg preprocessing, frame extraction, keyframes
│   ├── pose-estimation/    # 🔲 ViTPose+ (2D) + MotionBERT (3D lifting) wrappers
│   ├── mesh-recovery/      # 🔲 SMPL-X integration via HMR 2.0
│   ├── scene-reconstruction/ # 🔲 Depth Anything V2, snow plane RANSAC
│   ├── coaching-engine/    # 🔲 LLM + RAG + CSIA/CASI progression logic
│   ├── viewer-3d/          # 🔲 React Three Fiber skeleton/mesh viewer
│   ├── api/                # 🔲 FastAPI backend, Celery tasks
│   ├── web/                # 🔲 Next.js frontend
│   ├── core/               # 🔲 Shared types, configs, utilities
│   └── mobile/             # 🔲 React Native app
├── docker/                 # 🔲 Dockerfiles, docker-compose
├── models/                 # 🔲 Model configs, weight download scripts
├── data/                   # 🔲 Annotation schemas, benchmarks
├── scripts/                # 🔲 Training and evaluation scripts
└── tests/
```

## Conventions

- **Language**: Python 3.11+ for backend/ML, TypeScript for frontend.
- **Build**: Hatchling (Python), packages live under `packages/`.
- **Testing**: pytest, test files colocated at `packages/<name>/tests/test_*.py`.
- **Coordinate system**: Y-up, right-hand (X=right, Y=up, Z=toward camera). Snow surface normal ≈ `[0, 1, 0]`.
- **Angles**: Degrees unless otherwise noted.
- **Types**: Use Pydantic v2 models for schemas and data validation.
- **Dependencies**: Keep per-package deps minimal. `biomechanics` is pure numpy — no ML deps. ML-heavy packages (pose-estimation, mesh-recovery) isolate their torch/ONNX deps.
- **License**: MIT.

## Domain Context

- **CSIA Five Skills**: Rotary, Edging, Balance, Pressure, Coordination — the foundation for all ski metrics and progression logic.
- **CASI**: Snowboard equivalent framework with similar skill categories.
- **SnowIQ**: 0–200 composite score (weighted average of the five skills), mapped to CSIA levels (L1–L4).
- **ML Pipeline**: 6 stages — Video Preprocessing → 2D Pose → 3D Pose Lifting → Scene Reconstruction → Biomechanical Analysis → LLM Coaching. See PLAN.md §3–4 for details.

## Current Phase

**Phase 1 — "Video In, Skeleton Out"** (see PLAN.md §7):
- ✅ Biomechanics engine (metrics, SnowIQ, turn segmentation)
- 🔲 Video pipeline (FFmpeg, frame extraction)
- 🔲 Pose estimation wrappers (ViTPose+, MotionBERT)
- 🔲 3D viewer (React Three Fiber)
- 🔲 FastAPI backend with async pipeline

## OpenSpec Workflow

This project uses **OpenSpec** (`openspec`, globally installed) for spec-driven development.

- All change proposals live in `openspec/changes/<change-name>/`
- Before implementing new features, check if a proposal exists: `ls openspec/changes/`
- Key files per proposal: `proposal.md`, `design.md`, `tasks.md`, `specs/`
- Commands: `openspec propose`, `openspec apply`, `openspec archive`, `openspec explore`
- **Always check `tasks.md` for the current checklist** — implement only what's specified, in order.
- Mark tasks complete in `tasks.md` as you finish them.

## Guidelines for Agents

- Read `PLAN.md` before making architectural decisions — it contains detailed rationale for technology choices.
- Check `openspec/changes/` for any active proposals before starting implementation work.
- New packages should follow the `biomechanics` package as a structural template (module + schemas + tests + README).
- Keep ML inference code behind abstract interfaces so models are swappable (pluggable backend pattern).
- Tests are required — run with `pytest` from repo root.
- Don't add deps to the root `pyproject.toml` unless they're truly shared. Prefer per-package optional dependencies for heavy ML libs.
