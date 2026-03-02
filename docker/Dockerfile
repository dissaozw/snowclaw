# SnowClaw API + ML Pipeline
# Based on NVIDIA CUDA for GPU inference support
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/packages

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    numpy>=1.26 \
    pydantic>=2.0 \
    scipy>=1.11 \
    opencv-python-headless \
    onnxruntime-gpu \
    fastapi \
    uvicorn[standard] \
    celery[redis] \
    python-multipart \
    httpx

# Pre-download model weights into the image
RUN mkdir -p /root/.cache/snowclaw/models
# NOTE: Replace with actual model URLs when available
# RUN python -c "from pose_estimation.model_cache import download_model; \
#     download_model('https://...', 'vitpose_plus_base.onnx'); \
#     download_model('https://...', 'motionbert_lite.onnx')"

# Copy application code
COPY packages/ packages/
COPY data/ data/

# Detect GPU availability at runtime
ENV SNOWCLAW_DEVICE=auto

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
