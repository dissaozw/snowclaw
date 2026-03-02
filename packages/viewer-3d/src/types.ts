/** Names of all 14 tracked joints. */
export const JOINT_NAMES = [
  "head",
  "neck",
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
] as const;

export type JointName = (typeof JOINT_NAMES)[number];

/** Bone connections as pairs of joint names. */
export const BONE_CONNECTIONS: ReadonlyArray<[JointName, JointName]> = [
  ["head", "neck"],
  ["neck", "left_shoulder"],
  ["neck", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["right_shoulder", "right_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_elbow", "right_wrist"],
  ["neck", "left_hip"],
  ["neck", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["right_hip", "right_knee"],
  ["left_knee", "left_ankle"],
  ["right_knee", "right_ankle"],
];

/** 3D coordinate as [x, y, z]. Y-up, right-hand coordinate system. */
export type Vec3 = [number, number, number];

/** Per-joint 3D positions keyed by joint name. */
export type JointPositions = Record<JointName, Vec3>;

/** Biomechanical metrics for a single frame. */
export interface FrameMetrics {
  left_knee_deg: number;
  right_knee_deg: number;
  inclination_deg: number;
  /** Center of mass height as a percentage (0-1). */
  com_height_pct: number;
}

/** A single frame of 3D pose data. */
export interface Pose3DFrame {
  frame_idx: number;
  timestamp_s: number;
  pose: JointPositions;
  metrics: FrameMetrics;
}

/** The data source mode for the viewer. */
export type ViewerMode = "standalone" | "api";

/** Configuration for data loading. */
export interface DataSourceConfig {
  mode: ViewerMode;
  /** Path to local JSON file (standalone mode). */
  jsonPath?: string;
  /** Job ID for API mode. */
  jobId?: string;
  /** Base URL for the API (defaults to current origin). */
  apiBaseUrl?: string;
}
