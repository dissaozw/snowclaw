# SnowClaw Video Pipeline

FFmpeg-based video preprocessing for the SnowClaw pipeline. Extracts frames, normalizes resolution, and reads metadata.

## Features

- **Frame extraction** — Extract RGB frames at configurable FPS
- **Resolution normalization** — Downscale to max dimension preserving aspect ratio (no upscaling)
- **Metadata extraction** — Duration, FPS, resolution, codec, frame count via ffprobe
- **FFmpeg check** — Verify FFmpeg is available on PATH

## Requirements

- FFmpeg must be installed and on PATH
- No Python ML dependencies
