"""CLI command: crop a video to follow a tracked person."""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np


def _smooth(arr: np.ndarray, window: int = 31) -> np.ndarray:
    """Apply moving-average smoothing with edge-replicate padding.

    Uses edge-replicate padding (not zero-padding) so the trajectory does not
    drift toward the origin in the first/last ``window // 2`` frames.
    """
    w = min(window, len(arr))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return arr.copy()
    kernel = np.ones(w) / w
    pad = w // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _track_persons(
    video_path: Path,
) -> tuple[dict[int, list[tuple]], int]:
    """
    Run YOLO ByteTrack on the video and return per-frame bboxes keyed by track ID.

    Returns:
        tracks:   {track_id: [(frame_idx, x1, y1, x2, y2), ...]}
        n_frames: total number of frames processed by the tracker
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
    """Choose a seed track ID: explicit selection or the one with largest total area."""
    if track_id is not None:
        if track_id not in tracks:
            available = sorted(tracks.keys())
            print(
                f"Error: track ID {track_id} not found. Available IDs: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        return track_id

    # Auto-select seed: largest cumulative bbox area (= most prominent track segment)
    def total_area(tid: int) -> float:
        return sum((x2 - x1) * (y2 - y1) for _, x1, y1, x2, y2 in tracks[tid])

    chosen = max(tracks.keys(), key=total_area)
    print(f"  Auto-selected seed track ID {chosen} (largest cumulative area)")
    return chosen


def _build_continuous_detections(
    tracks: dict[int, list[tuple]],
    total_frames: int,
    seed_track_id: int,
) -> list[tuple]:
    """Stitch fragmented track IDs into one continuous detection stream.

    For each frame, pick the detection closest to the previous chosen center.
    This handles ByteTrack ID fragmentation (same skier split across IDs).
    """
    frame_map: dict[int, list[tuple]] = {}
    for tid, dets in tracks.items():
        for f, x1, y1, x2, y2 in dets:
            frame_map.setdefault(int(f), []).append((tid, x1, y1, x2, y2))

    chosen: list[tuple] = []
    prev_cx = prev_cy = None

    for f in range(total_frames):
        cands = frame_map.get(f, [])
        if not cands:
            continue

        if prev_cx is None:
            # Prefer seed track at start if present on this frame.
            seed = [c for c in cands if c[0] == seed_track_id]
            if seed:
                _, x1, y1, x2, y2 = max(seed, key=lambda c: (c[3]-c[1]) * (c[4]-c[2]))
            else:
                _, x1, y1, x2, y2 = max(cands, key=lambda c: (c[3]-c[1]) * (c[4]-c[2]))
        else:
            def score(c: tuple) -> tuple:
                _, x1, y1, x2, y2 = c
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                dist2 = (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2
                area = (x2 - x1) * (y2 - y1)
                return (dist2, -area)

            _, x1, y1, x2, y2 = min(cands, key=score)

        prev_cx = (x1 + x2) / 2
        prev_cy = (y1 + y2) / 2
        chosen.append((f, x1, y1, x2, y2))

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
    # Keep crop aspect ratio equal to input video ratio to avoid distortion.
    med_bw = float(np.percentile(bw_arr, 90))
    med_bh = float(np.percentile(bh_arr, 90))
    base_crop_w = max(int(med_bw * (1 + pad * 2)), 64)
    base_crop_h = max(int(med_bh * (1 + pad * 2)), 64)

    target_aspect = video_w / max(video_h, 1)
    if base_crop_w / max(base_crop_h, 1) > target_aspect:
        # too wide -> increase height
        base_crop_h = int(round(base_crop_w / target_aspect))
    else:
        # too tall -> increase width
        base_crop_w = int(round(base_crop_h * target_aspect))

    max_even_w = (video_w & ~1) or video_w
    max_even_h = (video_h & ~1) or video_h
    crop_w = min(base_crop_w, max_even_w)
    crop_h = min(base_crop_h, max_even_h)

    # Re-apply aspect after clamping (fit inside frame).
    if crop_w / max(crop_h, 1) > target_aspect:
        crop_w = int(round(crop_h * target_aspect))
    else:
        crop_h = int(round(crop_w / target_aspect))

    crop_w = max(2, min(crop_w, video_w))
    crop_h = max(2, min(crop_h, video_h))
    if crop_w > 1 and crop_w % 2:
        crop_w -= 1
    if crop_h > 1 and crop_h % 2:
        crop_h -= 1

    return cx_arr, cy_arr, crop_w, crop_h


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
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    try:
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
    except BrokenPipeError:
        # Defer error reporting to ffmpeg stderr/return code after cleanup.
        pass
    finally:
        cap.release()
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
        return_code = proc.wait()
        if return_code != 0:
            stderr_raw = proc.stderr.read() if proc.stderr is not None else b""
            stderr = (
                stderr_raw.decode("utf-8", errors="replace")
                if isinstance(stderr_raw, bytes)
                else str(stderr_raw)
            ).strip()
            msg = f"ffmpeg crop encode failed with code {return_code}"
            if stderr:
                msg = f"{msg}: {stderr}"
            raise RuntimeError(msg)


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
    tracks, n_frames = _track_persons(video_path)
    if not tracks:
        print("Error: No persons detected in video.", file=sys.stderr)
        return 1

    # Step 2: Select which person to follow (seed track), then stitch fragmented IDs.
    chosen_id = _select_track(tracks, args.track_id)
    frame_count_for_trajectory = max(total_frames, n_frames)
    if abs(total_frames - n_frames) > 1:
        warnings.warn(
            (
                "Frame count mismatch between OpenCV metadata and tracker output "
                f"(opencv={total_frames}, tracker={n_frames}); "
                f"using {frame_count_for_trajectory} frames for crop trajectory."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    if args.track_id is None:
        detections = _build_continuous_detections(tracks, frame_count_for_trajectory, chosen_id)
        coverage = len(detections) / max(frame_count_for_trajectory, 1)
        print(
            f"  Continuous target built from fragmented tracks (seed #{chosen_id}); "
            f"coverage {len(detections)}/{frame_count_for_trajectory} ({coverage*100:.1f}%)"
        )
    else:
        detections = sorted(tracks[chosen_id], key=lambda d: d[0])
        print(f"  Tracking person #{chosen_id} across {len(detections)} frames")

    # Step 3: Build smooth crop trajectory
    cx_arr, cy_arr, crop_w, crop_h = _build_crop_trajectory(
        detections,
        total_frames=frame_count_for_trajectory,
        video_w=video_w,
        video_h=video_h,
        pad=args.padding,
        smooth_window=args.smooth,
    )
    zoom = video_w / crop_w

    # Enforce output aspect ratio = input aspect ratio (no stretching)
    out_w = int(args.out_width)
    out_h = max(2, int(round(out_w * video_h / max(video_w, 1))))
    if out_h % 2:
        out_h += 1
    if args.out_height != out_h:
        print(
            f"  Adjusted output size to preserve input aspect ratio: "
            f"{args.out_width}x{args.out_height} -> {out_w}x{out_h}"
        )

    print(f"  Crop window: {crop_w}x{crop_h}px  (zoom ~{zoom:.1f}x)")

    # Step 4: Render
    print(f"  Rendering → {output_path} ...")
    _render_crop_opencv(
        video_path, cx_arr, cy_arr, crop_w, crop_h,
        video_w, video_h, output_path,
        out_w=out_w, out_h=out_h, fps=fps,
    )

    print(f"\nDone! → {output_path}")
    return 0
