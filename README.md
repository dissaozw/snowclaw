# SnowCoach AI

Open-source AI skiing & snowboarding instructor that transforms amateur video into actionable coaching feedback.

## What Makes It Different

- **Video-to-3D Reconstruction** — Upload a video, pause any frame, and orbit around the skier/snowboarder in full 3D. Zoom into ski-snow contact, edge angles, and more.
- **State-of-the-Art Motion Tracking** — Transformer-based 3D human pose estimation (BioPose, MotionBERT, SMPL-X) instead of legacy skeleton models.
- **Progression Engine** — Built on CSIA/CASI teaching frameworks. Tracks improvement over weeks and months, not just one-shot reports.
- **SnowIQ Score** — A 0–200 skill rating grounded in the CSIA/CASI framework, covering rotary, edging, balance, pressure, and coordination.
- **Fully Open Source** — MIT-licensed core with pluggable model backends and community-extensible skill frameworks.

## How It Works

1. Upload a ski or snowboard video
2. The ML pipeline extracts 2D pose, lifts to 3D, and recovers a full body mesh
3. A biomechanical engine computes skiing/snowboarding metrics (edge angles, balance, pressure, etc.)
4. An LLM coaching engine generates natural-language feedback aligned to CSIA/CASI skill levels
5. View results in an interactive 3D viewer with orbit controls, overlays, and frame scrubbing

## Architecture Overview

| Layer       | Technology                                |
|-------------|-------------------------------------------|
| Frontend    | Next.js + React Three Fiber               |
| Mobile      | React Native + expo-three                 |
| 3D Viewer   | Three.js / React Three Fiber              |
| Backend API | FastAPI (Python)                          |
| ML Serving  | Triton Inference Server or TorchServe     |
| Job Queue   | Celery + Redis                            |
| Database    | PostgreSQL + TimescaleDB                  |
| Vector DB   | Qdrant or Pgvector                        |
| Storage     | MinIO (self-hosted) / S3                  |
| ML Framework| PyTorch 2.x                               |
| CI/CD       | GitHub Actions + Docker                   |

## Getting Started

```bash
# Clone the repository
git clone https://github.com/snowcoach-ai/snowcoach.git
cd snowcoach

# Install dependencies (requires uv — https://docs.astral.sh/uv/)
uv sync

# Start development environment
docker compose up -d

# Download pretrained models
uv run python scripts/download_models.py

# Run the pipeline on a sample video
PYTHONPATH=packages uv run python -m snowclaw.cli process data/samples/ski_demo.mp4 --output-dir ./results/

# Run with mock backends (no model download, validates pipeline end-to-end)
PYTHONPATH=packages uv run python -m snowclaw.cli process data/samples/ski_demo.mp4 --output-dir ./results/ --mock

# Start the web viewer
cd packages/web && npm install && npm run dev
```

## Filming Tips

For best results when recording video for analysis:

- **Angle**: Side view, perpendicular to the fall line, capturing full body
- **Stability**: Use a tripod or stationary filmer
- **Distance**: 5–15 meters from the subject
- **Resolution**: 1080p minimum, 4K preferred, 60fps recommended
- **Duration**: At least 3–5 linked turns per run

## Development Tools

This project includes Claude Code skills for code quality and spec-driven development.

### Code Quality Skills

| Command | Description |
|---------|-------------|
| `/test` | Run pytest with failure analysis and suggested fixes |
| `/lint` | Run ruff linter and formatter with auto-fix |
| `/review` | Structured code review (correctness, security, testing, style, performance) |
| `/simplify` | Identify unnecessary complexity and suggest simplifications |

### OpenSpec (Spec-Driven Development)

OpenSpec requires Node.js 20.19.0+ and a one-time global install:

```bash
npm install -g @fission-ai/openspec@latest
openspec init
```

| Command | Description |
|---------|-------------|
| `/opsx:propose` | Create a change proposal with specs, design, and task checklist |
| `/opsx:apply` | Implement tasks from a spec |
| `/opsx:archive` | Archive a completed change |
| `/opsx:explore` | Explore ideas and clarify requirements before committing to a spec |

### Linting

Python linting uses [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`:

```bash
uv sync
uv run ruff check packages/
uv run ruff format packages/
```

## Contributing

Contributions are welcome across many areas:

| Area                    | Skill Level     |
|-------------------------|-----------------|
| Pose model fine-tuning  | Advanced ML     |
| Equipment detection     | Intermediate ML |
| Biomechanics metrics    | Domain expert   |
| Skill frameworks        | Ski instructor  |
| 3D viewer features      | Frontend        |
| Mobile app              | Mobile dev      |
| Documentation           | Any level       |
| Dataset annotation      | Beginner        |

See the full project plan in [PLAN.md](PLAN.md) for detailed architecture, roadmap, and technical decisions.

## License

MIT
