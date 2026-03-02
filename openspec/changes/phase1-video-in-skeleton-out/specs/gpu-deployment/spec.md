## ADDED Requirements

### Requirement: GPU-enabled Dockerfile
The system SHALL provide a Dockerfile based on an NVIDIA CUDA base image that includes all dependencies (FFmpeg, Python, ONNX Runtime GPU, model weights) for running the full pipeline on a GPU.

#### Scenario: Build and run with GPU
- **WHEN** a user runs `docker build` and `docker run --gpus all`
- **THEN** the container SHALL start the FastAPI server with GPU-accelerated inference

#### Scenario: Pre-warmed model weights
- **WHEN** the Docker image is built
- **THEN** ViTPose+ and MotionBERT weights SHALL be included in the image so no download is needed at runtime

### Requirement: Docker Compose for one-command startup
The system SHALL provide a `docker-compose.yml` that starts the full stack (FastAPI server, Celery worker, Redis, 3D viewer frontend) with a single `docker compose up` command.

#### Scenario: Start entire stack
- **WHEN** a user runs `docker compose up`
- **THEN** all services SHALL start and the application SHALL be accessible at `http://localhost:3000`

#### Scenario: GPU passthrough
- **WHEN** the host has an NVIDIA GPU with drivers installed
- **THEN** docker-compose SHALL pass the GPU to the ML worker container via NVIDIA Container Toolkit

### Requirement: Cloud GPU deployment guide
The system SHALL include documentation for deploying to common GPU cloud providers (AWS EC2 with GPU, GCP with T4/A10, RunPod, Lambda Labs).

#### Scenario: Deploy to a cloud GPU
- **WHEN** a user follows the deployment guide for their chosen provider
- **THEN** they SHALL have a running instance processing videos with GPU acceleration

### Requirement: CPU fallback mode
The system SHALL still function on CPU-only machines (slower but working), so development and testing do not require a GPU.

#### Scenario: Run without GPU
- **WHEN** a user runs the Docker image without `--gpus` flag
- **THEN** the system SHALL fall back to CPU inference with a warning about slower performance
