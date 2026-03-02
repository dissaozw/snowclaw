# GPU Cloud Deployment Guide

## Prerequisites

- Docker and Docker Compose v2
- NVIDIA Container Toolkit (for GPU support)
- GPU with CUDA 12.x support (recommended: NVIDIA T4, A10, or better)

## Quick Start

```bash
# Clone and start
git clone https://github.com/yourorg/snowclaw.git
cd snowclaw
docker compose up --build
```

Access at `http://localhost:3000`

## CPU Fallback

If no GPU is available, remove the `deploy.resources.reservations` section from `docker-compose.yml` and set:

```yaml
environment:
  - SNOWCLAW_DEVICE=cpu
```

**Warning:** CPU inference is significantly slower (~10x for 2D pose, ~5x for 3D lifting).

## Cloud Deployment

### AWS EC2 GPU

1. Launch a `g4dn.xlarge` instance (NVIDIA T4, ~$0.53/hr)
2. Use the Deep Learning AMI (includes NVIDIA drivers + Docker)
3. Clone repo and run `docker compose up --build -d`

### GCP Compute Engine

1. Create a VM with NVIDIA T4 or A10G GPU
2. Install Container-Optimized OS or use the Deep Learning VM
3. Install NVIDIA Container Toolkit: `sudo nvidia-ctk runtime configure`
4. Clone and run: `docker compose up --build -d`

### RunPod

1. Deploy a GPU Pod (select RTX 3090 or A100)
2. Use the PyTorch template as base
3. SSH in, clone repo, run `docker compose up --build -d`
4. Expose port 3000 in RunPod settings

### Lambda Labs

1. Launch an instance with any GPU tier
2. SSH in: `ssh ubuntu@<ip>`
3. Docker is pre-installed. Clone and run:
   ```bash
   git clone https://github.com/yourorg/snowclaw.git
   cd snowclaw
   docker compose up --build -d
   ```

## Architecture

```
Browser (port 3000)
    │
    ├── Static files (Nginx)
    └── /api/* ──► FastAPI (port 8000)
                      │
                      ├── Celery Worker (GPU)
                      │    ├── ViTPose+ (2D)
                      │    ├── MotionBERT (3D)
                      │    └── Video Annotation
                      │
                      └── Redis (port 6379)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CELERY_BROKER_URL` | `redis://redis:6379/0` | Redis URL for Celery |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/0` | Redis URL for results |
| `SNOWCLAW_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `RESULT_RETENTION_HOURS` | `24` | Hours before results are cleaned up |
