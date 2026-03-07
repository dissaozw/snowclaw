"""CLI command: crop a video to follow a tracked person."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _smooth(arr: np.ndarray, window: int = 31) -> np.ndarray:
    """Apply moving-average smoothing, clamping window to arr length."""
    w = min(window, len(arr))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return arr.copy()
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def _track_persons(video_path: Path, track_id: int | None) -> dict[int, list[tuple]]:
    """
    Run YOLO ByteTrack on the video and return per-frame bboxes keyed by track ID.

    Returns:
        tracks: {track_id: [(frame_idx, x1, y1, x2, y2), ...]}
        chosen_id: the track ID selected (largest cumulative bbox area)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: uv add ultralytics", file=sys.stderr)
        sys.exit(1)

    model = YOLO("yolo11n.pt")

    print("  Running YOLO ByteTrack person tracking...")
    results = model.track(
        source=str(video_path),
        classes=[0],          # person only
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
        stream=True,
    )

    tracks: dict[int, list[tuple]] = {}
    frame_idx = 0
    for r in results:
        if r.boxes is not None and r.boxes.id is not None:
            ids  = r.boxes.id.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            for tid, box in zip(ids, xyxy):
                tracks.setdefault(int(tid), []).append(
                    (frame_idx, float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                )
        frame_idx += 1

    print(f"  Found {len(tracks)} unique tracks across {frame_idx} frames")
    return tracks, frame_idx


def _select_track(tracks: dict[int, list[tuple]], track_id: int | None) -> int:
    """Choose a track ID: explicit selection or the one with the largest total bbox area."""
    if track_id is not None:
        if track_id not in tracks:
            available = sorted(tracks.keys())
            print(
                f"Error: track ID {track_id} not found. Available IDs: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        return track_id

    # Auto-select: largest cumulative bounding-box area (= most prominent / closest person)
    def total_area(tid: int) -> float:
        return sum((x2 - x1) * (y2 - y1) for _, x1, y1, x2, y2 in tracks[tid])

    chosen = max(tracks.keys(), key=total_area)
    print(f"  Auto-selected track ID {chosen} (largest cumulative area)")
    return chosen


def _build_crop_trajectory(
    detections: list[tuple],
    total_frames: int,
    video_w: int,
    video_h: int,
    pad: float = 0.4,
    smooth_window: int = 45,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Interpolate + smooth the per-frame crop centre and compute a fixed crop size.

    Returns:
        cx_arr, cy_arr : per-frame crop centre (smoothed), shape (total_frames,)
        crop_w, crop_h : fixed crop window size (pixels)
    """
    det_frames = np.array([d[0] for d in detections])
    raw_cx = np.array([(d[1] + d[3]) / 2 for d in detections])
    raw_cy = np.array([(d[2] + d[4]) / 2 for d in detections])
    raw_bw = np.array([d[3] - d[1] for d in detections])
    raw_bh = np.array([d[4] - d[2] for d in detections])

    all_frames = np.arange(total_frames)

    # Interpolate to every frame
    cx_raw = np.interp(all_frames, det_frames, raw_cx)
    cy_raw = np.interp(all_frames, det_frames, raw_cy)
    bw_raw = np.interp(all_frames, det_frames, raw_bw)
    bh_raw = np.interp(all_frames, det_frames, raw_bh)

    # Smooth trajectories to reduce jitter
    cx_arr = _smooth(cx_raw, smooth_window)
    cy_arr = _smooth(cy_raw, smooth_window)
    bw_arr = _smooth(bw_raw, smooth_window)
    bh_arr = _smooth(bh_raw, smooth_window)

    # Fixed crop size: 90th-percentile bbox + padding (avoids size oscillation)
    med_bw = float(np.percentile(bw_arr, 90))
    med_bh = float(np.percentile(bh_arr, 90))
    crop_w = min(video_w, int(med_bw * (1 + pad * 2))) & ~1
    crop_h = min(video_h, int(med_bh * (1 + pad * 2))) & ~1
    crop_w = max(crop_w, 64)
    crop_h = max(crop_h, 64)

    return cx_arr, cy_arr, crop_w, crop_h


def _render_crop(
    video_path: Path,
    cx_arr: np.ndarray,
    cy_arr: np.ndarray,
    crop_w: int,
    crop_h: int,
    video_w: int,
    video_h: int,
    output_path: Path,
    out_w: int = 848,
    out_h: int = 476,
) -> None:
    """Use ffmpeg sendcmd to apply per-frame dynamic crop following the person."""

    # Build a sendcmd script that sets crop x/y every frame
    # sendcmd format: <time_s> crop x <val>; <time_s> crop y <val>
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    lines = []
    for i, (cx, cy) in enumerate(zip(cx_arr, cy_arr)):
        x = int(np.clip(cx - crop_w / 2, 0, video_w - crop_w))
        y = int(np.clip(cy - crop_h / 2, 0, video_h - crop_h))
        t = i / fps
        lines.append(f"{t:.6f} [enter] crop x {x};")
        lines.append(f"{t:.6f} [enter] crop y {y};")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(lines))
        cmd_file = f.name

    vf = (
        f"crop={crop_w}:{crop_h}:0:0,"
        f"sendcmd=filename={cmd_file},"
        f"crop=keep_aspect=1,"   # keeps the crop params from sendcmd
        f"scale={out_w}:{out_h}:flags=lanczos"
    )

    # sendcmd + crop chaining is tricky; use simpler approach: write frames via OpenCV
    Path(cmd_file).unlink(missing_ok=True)
    _render_crop_opencv(video_path, cx_arr, cy_arr, crop_w, crop_h,
                        video_w, video_h, output_path, out_w, out_h, fps)


def _render_crop_opencv(
    video_path: Path,
    cx_arr: np.ndarray,
    cy_arr: np.ndarray,
    crop_w: int,
    crop_h: int,
    video_w: int,
    video_h: int,
    output_path: Path,
    out_w: int,
    out_h: int,
    fps: float,
) -> None:
    """Frame-by-frame crop using OpenCV, then encode with ffmpeg for quality."""
    import cv2

    # Write raw frames to a pipe → ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        # Re-mux audio from original
        "-i", str(video_path),
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-crf", "17",
        "-preset", "fast",
        "-c:a", "copy",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i = min(frame_idx, len(cx_arr) - 1)
        cx = cx_arr[i]
        cy = cy_arr[i]
        x = int(np.clip(cx - crop_w / 2, 0, video_w - crop_w))
        y = int(np.clip(cy - crop_h / 2, 0, video_h - crop_h))
        cropped = frame[y : y + crop_h, x : x + crop_w]
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        proc.stdin.write(resized.tobytes())
        frame_idx += 1

    cap.release()
    proc.stdin.close()
    proc.wait()


def run_crop(args) -> int:
    """Entry point for the `snowclaw crop` subcommand."""
    import cv2

    video_path = Path(args.video)
    output_path = Path(args.output) if args.output else video_path.with_stem(video_path.stem + "_cropped")

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get video dimensions
    cap = cv2.VideoCapture(str(video_path))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"  Input: {video_w}x{video_h} @ {fps:.1f}fps, {total_frames} frames")

    # Step 1: Track all persons
    tracks, n_frames = _track_persons(video_path, args.track_id)
    if not tracks:
        print("Error: No persons detected in video.", file=sys.stderr)
        return 1

    # Step 2: Select which person to follow
    chosen_id = _select_track(tracks, args.track_id)
    detections = sorted(tracks[chosen_id], key=lambda d: d[0])
    print(f"  Tracking person #{chosen_id} across {len(detections)} frames")

    # Step 3: Build smooth crop trajectory
    cx_arr, cy_arr, crop_w, crop_h = _build_crop_trajectory(
        detections,
        total_frames=n_frames,
        video_w=video_w,
        video_h=video_h,
        pad=args.padding,
        smooth_window=args.smooth,
    )
    zoom = video_w / crop_w
    print(f"  Crop window: {crop_w}x{crop_h}px  (zoom ~{zoom:.1f}x)")

    # Step 4: Render
    print(f"  Rendering → {output_path} ...")
    _render_crop_opencv(
        video_path, cx_arr, cy_arr, crop_w, crop_h,
        video_w, video_h, output_path,
        out_w=args.out_width, out_h=args.out_height, fps=fps,
    )

    print(f"\nDone! → {output_path}")
    return 0
