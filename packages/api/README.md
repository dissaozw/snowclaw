# SnowClaw API

FastAPI backend with async video processing pipeline for the SnowClaw AI skiing/snowboarding instructor.

## Architecture

```
Client                     API Server                    Celery Worker
  │                            │                              │
  ├── POST /api/upload ──────► │                              │
  │                            ├── create job (queued) ──►    │
  │   ◄── 202 {job_id} ──────┤├── process_video.delay() ───► │
  │                            │                              ├── extract_frames
  ├── WS /api/ws/status/{id} ► │                              ├── pose_2d (ViTPose+)
  │   ◄── {stage, progress} ──┤│ ◄── job_store.update() ─────┤├── pose_3d (MotionBERT)
  │   ◄── {completed} ────────┤│                              ├── annotate_video
  │                            │                              ├── compute_metrics
  ├── GET /results/{id}/video ► │                              └── save results
  │   ◄── annotated.mp4 ──────┤│
  ├── GET /results/{id}/poses ► │
  │   ◄── [{pose, metrics}] ──┤│
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload video file, returns `{job_id, status}` |
| WS | `/api/ws/status/{job_id}` | WebSocket progress updates (stage + %) |
| GET | `/api/results/{job_id}/video` | Download annotated video |
| GET | `/api/results/{job_id}/poses` | Per-frame Pose3D + metrics JSON |

## Pipeline Stages

1. **extracting_frames** — FFmpeg frame extraction via `video_pipeline.extract_frames`
2. **pose_2d** — 2D keypoint detection via `pose_estimation.ViTPoseBackend`
3. **pose_3d** — 3D pose lifting via `pose_estimation.MotionBERTBackend`
4. **annotating_video** — Skeleton overlay via `video_annotation.annotate_video`
5. **computing_metrics** — Per-frame biomechanical metrics
6. **saving_results** — Write `annotated.mp4` and `poses.json`

## Dependencies

- **fastapi** + **uvicorn** — HTTP/WebSocket server
- **celery** + **redis** — Async task queue
- **python-multipart** — File upload parsing

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `SNOWCLAW_UPLOAD_DIR` | `/tmp/snowclaw/uploads` | Upload storage path |
| `SNOWCLAW_RESULTS_DIR` | `/tmp/snowclaw/results` | Result file storage path |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/0` | Celery result backend |
| `SNOWCLAW_MAX_UPLOAD_MB` | `500` | Max upload file size (MB) |
| `SNOWCLAW_RESULT_RETENTION_SECONDS` | `86400` | Auto-cleanup after N seconds |

## Running

```bash
# Start the API server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Start the Celery worker
celery -A api.tasks worker --loglevel=info
```

## Testing

```bash
pytest packages/api/tests/ -v
```

Tests use httpx `AsyncClient` with mocked Celery tasks and in-memory job storage — no Redis required.
