import { useCallback, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { MetricsPanel } from "./components/MetricsPanel";
import { Timeline } from "./components/Timeline";
import { Viewer } from "./components/Viewer";
import { usePoseData } from "./hooks/usePoseData";
import type { DataSourceConfig, ViewerMode } from "./types";

/** Parse URL search params to detect mode and config. */
function getInitialConfig(): DataSourceConfig {
  const params = new URLSearchParams(window.location.search);
  const mode = (params.get("mode") as ViewerMode) ?? "standalone";
  return {
    mode,
    jsonPath: params.get("json") ?? undefined,
    jobId: params.get("job_id") ?? undefined,
    apiBaseUrl: params.get("api_base") ?? undefined,
  };
}

const appContainerStyle: CSSProperties = {
  width: "100%",
  height: "100%",
  position: "relative",
  display: "flex",
  flexDirection: "column",
};

const viewerContainerStyle: CSSProperties = {
  flex: 1,
  position: "relative",
};

const setupPanelStyle: CSSProperties = {
  position: "absolute",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  background: "rgba(0, 0, 0, 0.85)",
  borderRadius: "12px",
  padding: "32px",
  minWidth: "380px",
  border: "1px solid rgba(255, 255, 255, 0.1)",
  zIndex: 20,
};

const setupTitleStyle: CSSProperties = {
  fontSize: "18px",
  fontWeight: 600,
  marginBottom: "20px",
  color: "#e0e0e0",
};

const inputGroupStyle: CSSProperties = {
  marginBottom: "16px",
};

const labelStyle: CSSProperties = {
  display: "block",
  fontSize: "13px",
  color: "#a0aec0",
  marginBottom: "4px",
};

const inputStyle: CSSProperties = {
  width: "100%",
  padding: "8px 12px",
  background: "rgba(255, 255, 255, 0.08)",
  border: "1px solid rgba(255, 255, 255, 0.2)",
  borderRadius: "6px",
  color: "#e0e0e0",
  fontSize: "14px",
  outline: "none",
};

const buttonStyle: CSSProperties = {
  padding: "8px 20px",
  background: "#339af0",
  border: "none",
  borderRadius: "6px",
  color: "#fff",
  fontSize: "14px",
  fontWeight: 600,
  cursor: "pointer",
  width: "100%",
};

const tabContainerStyle: CSSProperties = {
  display: "flex",
  gap: "4px",
  marginBottom: "16px",
};

const tabStyle = (active: boolean): CSSProperties => ({
  flex: 1,
  padding: "8px",
  background: active ? "rgba(51, 154, 240, 0.2)" : "rgba(255, 255, 255, 0.05)",
  border: `1px solid ${active ? "#339af0" : "rgba(255, 255, 255, 0.1)"}`,
  borderRadius: "6px",
  color: active ? "#339af0" : "#a0aec0",
  cursor: "pointer",
  fontSize: "13px",
  fontWeight: active ? 600 : 400,
  textAlign: "center" as const,
});

const errorBannerStyle: CSSProperties = {
  position: "absolute",
  top: 16,
  left: "50%",
  transform: "translateX(-50%)",
  background: "rgba(255, 59, 48, 0.9)",
  padding: "10px 20px",
  borderRadius: "8px",
  fontSize: "13px",
  color: "#fff",
  zIndex: 20,
  maxWidth: "500px",
  textAlign: "center",
};

const loadingStyle: CSSProperties = {
  position: "absolute",
  top: 16,
  left: "50%",
  transform: "translateX(-50%)",
  background: "rgba(0, 0, 0, 0.75)",
  padding: "10px 20px",
  borderRadius: "8px",
  fontSize: "13px",
  color: "#a0aec0",
  zIndex: 20,
};

/**
 * Main application component.
 *
 * Provides a setup panel to choose data source (standalone JSON file or API),
 * then renders the 3D viewer with timeline scrubber and metrics panel.
 */
export function App() {
  const initialConfig = useMemo(() => getInitialConfig(), []);

  const [config, setConfig] = useState<DataSourceConfig | null>(() => {
    // Auto-start if URL params provide enough info.
    if (initialConfig.mode === "standalone" && initialConfig.jsonPath) {
      return initialConfig;
    }
    if (initialConfig.mode === "api" && initialConfig.jobId) {
      return initialConfig;
    }
    return null;
  });

  const [setupMode, setSetupMode] = useState<ViewerMode>(
    initialConfig.mode ?? "standalone",
  );
  const [jsonPath, setJsonPath] = useState(initialConfig.jsonPath ?? "");
  const [jobId, setJobId] = useState(initialConfig.jobId ?? "");
  const [apiBase, setApiBase] = useState(initialConfig.apiBaseUrl ?? "");

  // Only call the hook when we have config. Pass a dummy config when null
  // to satisfy the hook's requirement, but loading won't actually trigger
  // because we conditionally show the setup panel instead.
  const hookConfig = useMemo<DataSourceConfig>(
    () =>
      config ?? {
        mode: "standalone",
        jsonPath: "__none__",
      },
    [config],
  );

  const {
    frames,
    currentFrame,
    setCurrentFrame,
    isPlaying,
    togglePlay,
    loading,
    error,
    playbackSpeed,
    setPlaybackSpeed,
  } = usePoseData(hookConfig);

  const currentPose = frames[currentFrame]?.pose ?? null;
  const currentMetrics = frames[currentFrame]?.metrics ?? null;
  const currentTimestamp = frames[currentFrame]?.timestamp_s ?? 0;

  const handleSubmit = useCallback(() => {
    if (setupMode === "standalone" && jsonPath.trim()) {
      setConfig({ mode: "standalone", jsonPath: jsonPath.trim() });
    } else if (setupMode === "api" && jobId.trim()) {
      setConfig({
        mode: "api",
        jobId: jobId.trim(),
        apiBaseUrl: apiBase.trim() || undefined,
      });
    }
  }, [setupMode, jsonPath, jobId, apiBase]);

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        // Create a blob URL for the selected file.
        const url = URL.createObjectURL(file);
        setJsonPath(url);
      }
    },
    [],
  );

  // Show setup panel if no config is set.
  if (!config) {
    return (
      <div style={appContainerStyle}>
        <div style={viewerContainerStyle}>
          <Viewer pose={null} />
          <div style={setupPanelStyle}>
            <div style={setupTitleStyle}>SnowClaw 3D Viewer</div>

            <div style={tabContainerStyle}>
              <button
                style={tabStyle(setupMode === "standalone")}
                onClick={() => setSetupMode("standalone")}
              >
                Standalone (JSON)
              </button>
              <button
                style={tabStyle(setupMode === "api")}
                onClick={() => setSetupMode("api")}
              >
                API Mode
              </button>
            </div>

            {setupMode === "standalone" && (
              <>
                <div style={inputGroupStyle}>
                  <label style={labelStyle}>JSON File URL or Path</label>
                  <input
                    type="text"
                    style={inputStyle}
                    placeholder="/data/poses.json"
                    value={jsonPath}
                    onChange={(e) => setJsonPath(e.target.value)}
                  />
                </div>
                <div style={inputGroupStyle}>
                  <label style={labelStyle}>Or select a local file</label>
                  <input
                    type="file"
                    accept=".json"
                    onChange={handleFileSelect}
                    style={{ ...inputStyle, padding: "6px" }}
                  />
                </div>
              </>
            )}

            {setupMode === "api" && (
              <>
                <div style={inputGroupStyle}>
                  <label style={labelStyle}>Job ID</label>
                  <input
                    type="text"
                    style={inputStyle}
                    placeholder="abc123"
                    value={jobId}
                    onChange={(e) => setJobId(e.target.value)}
                  />
                </div>
                <div style={inputGroupStyle}>
                  <label style={labelStyle}>API Base URL (optional)</label>
                  <input
                    type="text"
                    style={inputStyle}
                    placeholder="http://localhost:8000"
                    value={apiBase}
                    onChange={(e) => setApiBase(e.target.value)}
                  />
                </div>
              </>
            )}

            <button style={buttonStyle} onClick={handleSubmit}>
              Load Pose Data
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Main viewer with data loaded.
  return (
    <div style={appContainerStyle}>
      <div style={viewerContainerStyle}>
        <Viewer pose={currentPose} />

        {/* Metrics overlay */}
        <MetricsPanel metrics={currentMetrics} />

        {/* Error banner */}
        {error && <div style={errorBannerStyle}>{error}</div>}

        {/* Loading indicator */}
        {loading && <div style={loadingStyle}>Loading pose data...</div>}
      </div>

      {/* Timeline scrubber */}
      {frames.length > 0 && (
        <Timeline
          totalFrames={frames.length}
          currentFrame={currentFrame}
          onFrameChange={setCurrentFrame}
          isPlaying={isPlaying}
          onTogglePlay={togglePlay}
          timestamp={currentTimestamp}
          playbackSpeed={playbackSpeed}
          onPlaybackSpeedChange={setPlaybackSpeed}
        />
      )}
    </div>
  );
}
