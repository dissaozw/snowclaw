import type { CSSProperties } from "react";

interface TimelineProps {
  /** Total number of frames. */
  totalFrames: number;
  /** Current frame index (0-based). */
  currentFrame: number;
  /** Callback when the user scrubs to a frame. */
  onFrameChange: (frame: number) => void;
  /** Whether playback is active. */
  isPlaying: boolean;
  /** Toggle play/pause. */
  onTogglePlay: () => void;
  /** Current timestamp in seconds. */
  timestamp: number;
  /** Current playback speed multiplier. */
  playbackSpeed: number;
  /** Callback to change playback speed. */
  onPlaybackSpeedChange: (speed: number) => void;
}

const SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4];

const containerStyle: CSSProperties = {
  position: "absolute",
  bottom: 0,
  left: 0,
  right: 0,
  padding: "12px 20px",
  background: "linear-gradient(transparent, rgba(0, 0, 0, 0.85))",
  display: "flex",
  alignItems: "center",
  gap: "12px",
  zIndex: 10,
};

const buttonStyle: CSSProperties = {
  background: "none",
  border: "1px solid rgba(255, 255, 255, 0.3)",
  borderRadius: "4px",
  color: "#e0e0e0",
  padding: "6px 12px",
  cursor: "pointer",
  fontSize: "14px",
  minWidth: "40px",
};

const sliderStyle: CSSProperties = {
  flex: 1,
  height: "4px",
  cursor: "pointer",
  accentColor: "#339af0",
};

const labelStyle: CSSProperties = {
  fontSize: "12px",
  color: "#a0aec0",
  fontFamily: "monospace",
  whiteSpace: "nowrap",
};

const speedButtonStyle = (active: boolean): CSSProperties => ({
  ...buttonStyle,
  padding: "2px 6px",
  fontSize: "11px",
  minWidth: "32px",
  background: active ? "rgba(51, 154, 240, 0.3)" : "none",
  borderColor: active ? "#339af0" : "rgba(255, 255, 255, 0.2)",
});

/**
 * Frame scrubber / timeline control bar.
 *
 * Displays a slider for scrubbing through frames, a play/pause button,
 * speed controls, and the current frame/timestamp readout.
 */
export function Timeline({
  totalFrames,
  currentFrame,
  onFrameChange,
  isPlaying,
  onTogglePlay,
  timestamp,
  playbackSpeed,
  onPlaybackSpeedChange,
}: TimelineProps) {
  const maxFrame = Math.max(0, totalFrames - 1);

  return (
    <div style={containerStyle}>
      {/* Play/Pause */}
      <button
        style={buttonStyle}
        onClick={onTogglePlay}
        title={isPlaying ? "Pause" : "Play"}
      >
        {isPlaying ? "||" : ">"}
      </button>

      {/* Frame slider */}
      <input
        type="range"
        min={0}
        max={maxFrame}
        value={currentFrame}
        onChange={(e) => onFrameChange(Number(e.target.value))}
        style={sliderStyle}
      />

      {/* Frame / time readout */}
      <span style={labelStyle}>
        {currentFrame + 1}/{totalFrames} &middot; {timestamp.toFixed(2)}s
      </span>

      {/* Speed controls */}
      <div style={{ display: "flex", gap: "2px", alignItems: "center" }}>
        <span style={{ ...labelStyle, marginRight: "4px" }}>Speed:</span>
        {SPEED_OPTIONS.map((speed) => (
          <button
            key={speed}
            style={speedButtonStyle(playbackSpeed === speed)}
            onClick={() => onPlaybackSpeedChange(speed)}
          >
            {speed}x
          </button>
        ))}
      </div>
    </div>
  );
}
