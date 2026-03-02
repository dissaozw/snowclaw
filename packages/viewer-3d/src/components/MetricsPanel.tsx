import type { CSSProperties } from "react";
import type { FrameMetrics } from "../types";

interface MetricsPanelProps {
  /** Current frame metrics; null when no data is loaded. */
  metrics: FrameMetrics | null;
}

const panelStyle: CSSProperties = {
  position: "absolute",
  top: 16,
  right: 16,
  background: "rgba(0, 0, 0, 0.75)",
  borderRadius: "8px",
  padding: "16px 20px",
  minWidth: "220px",
  backdropFilter: "blur(8px)",
  border: "1px solid rgba(255, 255, 255, 0.1)",
  zIndex: 10,
};

const titleStyle: CSSProperties = {
  fontSize: "13px",
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  color: "#a0aec0",
  marginBottom: "12px",
  borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
  paddingBottom: "8px",
};

const rowStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  padding: "4px 0",
  fontSize: "13px",
};

const labelTextStyle: CSSProperties = {
  color: "#a0aec0",
};

const valueStyle: CSSProperties = {
  fontFamily: "monospace",
  fontWeight: 600,
  color: "#e0e0e0",
};

function MetricRow({
  label,
  value,
  unit,
  color,
}: {
  label: string;
  value: string;
  unit: string;
  color?: string;
}) {
  return (
    <div style={rowStyle}>
      <span style={labelTextStyle}>{label}</span>
      <span style={{ ...valueStyle, color: color ?? valueStyle.color }}>
        {value}
        <span style={{ fontSize: "11px", color: "#718096", marginLeft: "2px" }}>
          {unit}
        </span>
      </span>
    </div>
  );
}

/**
 * Metrics panel overlay showing biomechanical data for the current frame.
 *
 * Displays knee flexion angles (left/right), body inclination, and
 * center of mass height percentage.
 */
export function MetricsPanel({ metrics }: MetricsPanelProps) {
  if (!metrics) {
    return (
      <div style={panelStyle}>
        <div style={titleStyle}>Metrics</div>
        <div style={{ color: "#718096", fontSize: "13px" }}>
          No data loaded
        </div>
      </div>
    );
  }

  /** Color-code knee angles: green when well-flexed, yellow when moderate, red when too straight. */
  function kneeAngleColor(deg: number): string {
    if (deg < 120) return "#51cf66";
    if (deg < 150) return "#fcc419";
    return "#ff6b6b";
  }

  return (
    <div style={panelStyle}>
      <div style={titleStyle}>Metrics</div>
      <MetricRow
        label="L Knee"
        value={metrics.left_knee_deg.toFixed(1)}
        unit="deg"
        color={kneeAngleColor(metrics.left_knee_deg)}
      />
      <MetricRow
        label="R Knee"
        value={metrics.right_knee_deg.toFixed(1)}
        unit="deg"
        color={kneeAngleColor(metrics.right_knee_deg)}
      />
      <MetricRow
        label="Inclination"
        value={metrics.inclination_deg.toFixed(1)}
        unit="deg"
      />
      <MetricRow
        label="COM Height"
        value={(metrics.com_height_pct * 100).toFixed(1)}
        unit="%"
      />
    </div>
  );
}
