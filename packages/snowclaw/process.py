"""CLI command: process a video through the full pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _mock_env() -> bool:
    """Return True if SNOWCLAW_MOCK_MODELS env var requests mock backends."""
    return os.environ.get("SNOWCLAW_MOCK_MODELS", "").strip() in ("1", "true", "yes")


def main(argv: list[str] | None = None) -> int:
    """Run the SnowClaw CLI."""
    parser = argparse.ArgumentParser(
        prog="snowclaw",
        description="SnowClaw AI — process ski/snowboard video into annotated output + 3D poses",
    )
    subparsers = parser.add_subparsers(dest="command")

    # process subcommand
    process_parser = subparsers.add_parser(
        "process",
        help="Process a video through the full pipeline",
    )
    process_parser.add_argument(
        "video",
        type=str,
        help="Path to the input video file",
    )
    process_parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save output files (default: ./results)",
    )
    process_parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target FPS for frame extraction (default: native FPS)",
    )
    process_parser.add_argument(
        "--max-dimension",
        type=int,
        default=1920,
        help="Max pixel dimension for frame processing (default: 1920)",
    )
    process_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto)",
    )
    process_parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help=(
            "Use mock pose estimation backends (no model download required). "
            "Useful for CI, testing, and pipeline validation. "
            "Also enabled by SNOWCLAW_MOCK_MODELS=1 env var."
        ),
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "process":
        return _process_video(args)

    return 0


def _process_video(args: argparse.Namespace) -> int:
    """Run the full pipeline on a video file."""
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # Step 1: Check FFmpeg
    print("[1/5] Checking FFmpeg...")
    from video_pipeline import ffmpeg_check
    try:
        ffmpeg_check()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Step 2: Extract metadata and frames
    print("[2/5] Extracting frames...")
    from video_pipeline import extract_metadata, extract_frames

    meta = extract_metadata(video_path)
    print(f"  Video: {meta.width}x{meta.height} @ {meta.fps:.1f} fps, {meta.duration_s:.1f}s")

    frames = extract_frames(
        video_path,
        target_fps=args.fps,
        max_dimension=args.max_dimension,
    )
    print(f"  Extracted {len(frames)} frames")

    # Step 3: 2D pose estimation
    use_mock = getattr(args, "mock", False) or _mock_env()

    if use_mock:
        print("[3/5] Detecting 2D keypoints (mock — no model download)...")
        from pose_estimation.mock_backend import MockViTPoseBackend
        estimator = MockViTPoseBackend()
    else:
        print("[3/5] Detecting 2D keypoints (ViTPose+)...")
        from pose_estimation.vitpose_backend import ViTPoseBackend
        estimator = ViTPoseBackend(device=args.device)

    keypoints_2d = estimator.predict(frames)
    print(f"  Detected keypoints in {len(keypoints_2d)} frames")

    # Step 4: 3D pose lifting
    if use_mock:
        print("[4/5] Lifting to 3D (mock — no model download)...")
        from pose_estimation.mock_backend import MockMotionBERTBackend
        lifter = MockMotionBERTBackend()
    else:
        print("[4/5] Lifting to 3D (MotionBERT)...")
        from pose_estimation.motionbert_backend import MotionBERTBackend
        lifter = MotionBERTBackend(device=args.device)

    poses_3d = lifter.lift(keypoints_2d)
    print(f"  Lifted {len(poses_3d)} frames to 3D")

    # Step 5: Annotate video and save results
    print("[5/5] Annotating video...")
    from video_annotation.renderer import annotate_video
    from video_annotation.skeleton import format_metrics

    output_video = output_dir / "annotated.mp4"
    annotate_video(video_path, keypoints_2d, poses_3d, output_video, fps=args.fps or meta.fps)

    # Save poses as JSON
    effective_fps = args.fps or meta.fps
    poses_json = []
    for i, pose in enumerate(poses_3d):
        frame_data = {
            "frame_idx": i,
            "timestamp_s": round(i / effective_fps, 4),
            "pose": pose.model_dump(exclude_none=True),
            "metrics": format_metrics(pose),
        }
        poses_json.append(frame_data)

    output_poses = output_dir / "poses.json"
    with open(output_poses, "w") as f:
        json.dump(poses_json, f, indent=2)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s!")
    print(f"  Annotated video: {output_video}")
    print(f"  Pose data:       {output_poses}")
    print(f"  Frames:          {len(poses_3d)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
