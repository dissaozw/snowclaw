import { useCallback, useEffect, useRef, useState } from "react";
import type { DataSourceConfig, Pose3DFrame } from "../types";

interface UsePoseDataReturn {
  /** All loaded pose frames. */
  frames: Pose3DFrame[];
  /** Current frame index. */
  currentFrame: number;
  /** Set the current frame index. */
  setCurrentFrame: (idx: number) => void;
  /** Whether playback is active. */
  isPlaying: boolean;
  /** Toggle play/pause. */
  togglePlay: () => void;
  /** Loading state. */
  loading: boolean;
  /** Error message if data loading failed. */
  error: string | null;
  /** Playback speed multiplier (1 = real-time). */
  playbackSpeed: number;
  /** Set playback speed. */
  setPlaybackSpeed: (speed: number) => void;
}

/**
 * Hook that loads pose data from either a local JSON file (standalone mode)
 * or from the API (api mode), and manages frame playback state.
 */
export function usePoseData(config: DataSourceConfig): UsePoseDataReturn {
  const [frames, setFrames] = useState<Pose3DFrame[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  const animFrameRef = useRef<number | null>(null);
  const lastTimestampRef = useRef<number | null>(null);
  const framesRef = useRef<Pose3DFrame[]>([]);
  const currentFrameRef = useRef(0);

  // Keep refs in sync with state.
  framesRef.current = frames;
  currentFrameRef.current = currentFrame;

  // Load data based on config.
  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      setLoading(true);
      setError(null);

      try {
        let url: string;

        if (config.mode === "standalone") {
          if (!config.jsonPath) {
            throw new Error("jsonPath is required in standalone mode");
          }
          url = config.jsonPath;
        } else {
          if (!config.jobId) {
            throw new Error("jobId is required in API mode");
          }
          const base = config.apiBaseUrl ?? "";
          url = `${base}/api/results/${config.jobId}/poses`;
        }

        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch pose data: ${response.status} ${response.statusText}`,
          );
        }

        const data: unknown = await response.json();

        // Accept both a raw array of frames or an object with a "frames" key.
        let frameArray: Pose3DFrame[];
        if (Array.isArray(data)) {
          frameArray = data as Pose3DFrame[];
        } else if (
          data !== null &&
          typeof data === "object" &&
          "frames" in data &&
          Array.isArray((data as { frames: unknown }).frames)
        ) {
          frameArray = (data as { frames: Pose3DFrame[] }).frames;
        } else {
          throw new Error("Unexpected data format: expected array of frames");
        }

        if (!cancelled) {
          setFrames(frameArray);
          setCurrentFrame(0);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : "Unknown error loading data",
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadData();

    return () => {
      cancelled = true;
    };
  }, [config.mode, config.jsonPath, config.jobId, config.apiBaseUrl]);

  // Playback animation loop.
  useEffect(() => {
    if (!isPlaying || framesRef.current.length < 2) {
      lastTimestampRef.current = null;
      return;
    }

    function tick(timestamp: number) {
      if (lastTimestampRef.current === null) {
        lastTimestampRef.current = timestamp;
        animFrameRef.current = requestAnimationFrame(tick);
        return;
      }

      const allFrames = framesRef.current;
      const idx = currentFrameRef.current;
      const currentPose = allFrames[idx];
      const nextIdx = idx + 1;

      if (nextIdx >= allFrames.length) {
        // Loop back to start.
        setCurrentFrame(0);
        lastTimestampRef.current = timestamp;
        animFrameRef.current = requestAnimationFrame(tick);
        return;
      }

      const nextPose = allFrames[nextIdx];
      if (!currentPose || !nextPose) {
        animFrameRef.current = requestAnimationFrame(tick);
        return;
      }

      // Calculate how long to hold the current frame based on timestamps.
      const frameDurationMs =
        (nextPose.timestamp_s - currentPose.timestamp_s) * 1000;
      const elapsedMs =
        (timestamp - lastTimestampRef.current) * playbackSpeed;

      if (elapsedMs >= frameDurationMs && frameDurationMs > 0) {
        setCurrentFrame(nextIdx);
        lastTimestampRef.current = timestamp;
      } else if (frameDurationMs <= 0) {
        // If timestamps are identical or invalid, advance at ~30fps.
        if (elapsedMs >= 33) {
          setCurrentFrame(nextIdx);
          lastTimestampRef.current = timestamp;
        }
      }

      animFrameRef.current = requestAnimationFrame(tick);
    }

    animFrameRef.current = requestAnimationFrame(tick);

    return () => {
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
    };
  }, [isPlaying, playbackSpeed]);

  const togglePlay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  return {
    frames,
    currentFrame,
    setCurrentFrame,
    isPlaying,
    togglePlay,
    loading,
    error,
    playbackSpeed,
    setPlaybackSpeed,
  };
}
