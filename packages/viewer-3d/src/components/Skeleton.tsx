import { useMemo } from "react";
import * as THREE from "three";
import type { JointName, JointPositions } from "../types";
import { BONE_CONNECTIONS, JOINT_NAMES } from "../types";

/** Radius of each joint sphere. */
const JOINT_RADIUS = 0.02;

/** Radius of each bone cylinder. */
const BONE_RADIUS = 0.008;

/** Color palette for different body regions. */
const JOINT_COLORS: Record<string, string> = {
  head: "#ff6b6b",
  neck: "#ffa94d",
  left_shoulder: "#51cf66",
  right_shoulder: "#51cf66",
  left_elbow: "#339af0",
  right_elbow: "#339af0",
  left_wrist: "#845ef7",
  right_wrist: "#845ef7",
  left_hip: "#fcc419",
  right_hip: "#fcc419",
  left_knee: "#20c997",
  right_knee: "#20c997",
  left_ankle: "#f06595",
  right_ankle: "#f06595",
};

const BONE_COLOR = "#94d0ff";

interface SkeletonProps {
  /** Joint positions for the current frame. */
  pose: JointPositions;
}

/** Renders a single bone as a cylinder between two joints. */
function Bone({ start, end }: { start: THREE.Vector3; end: THREE.Vector3 }) {
  const { position, quaternion, length } = useMemo(() => {
    const mid = new THREE.Vector3()
      .addVectors(start, end)
      .multiplyScalar(0.5);
    const direction = new THREE.Vector3().subVectors(end, start);
    const len = direction.length();

    // Cylinder is aligned along Y axis by default; rotate to point from start to end.
    const quat = new THREE.Quaternion();
    if (len > 1e-6) {
      const up = new THREE.Vector3(0, 1, 0);
      quat.setFromUnitVectors(up, direction.clone().normalize());
    }

    return { position: mid, quaternion: quat, length: len };
  }, [start, end]);

  if (length < 1e-6) return null;

  return (
    <mesh position={position} quaternion={quaternion}>
      <cylinderGeometry args={[BONE_RADIUS, BONE_RADIUS, length, 8]} />
      <meshStandardMaterial color={BONE_COLOR} />
    </mesh>
  );
}

/**
 * 3D skeleton visualization component.
 *
 * Renders spheres at each of the 14 joints and cylinders connecting
 * bones according to the defined bone connections.
 */
export function Skeleton({ pose }: SkeletonProps) {
  // Pre-convert joint positions to THREE.Vector3 for efficient reuse.
  const jointVectors = useMemo(() => {
    const map = new Map<JointName, THREE.Vector3>();
    for (const name of JOINT_NAMES) {
      const [x, y, z] = pose[name];
      map.set(name, new THREE.Vector3(x, y, z));
    }
    return map;
  }, [pose]);

  return (
    <group>
      {/* Joint spheres */}
      {JOINT_NAMES.map((name) => {
        const vec = jointVectors.get(name);
        if (!vec) return null;
        return (
          <mesh key={name} position={vec}>
            <sphereGeometry args={[JOINT_RADIUS, 16, 16]} />
            <meshStandardMaterial
              color={JOINT_COLORS[name] ?? "#ffffff"}
              emissive={JOINT_COLORS[name] ?? "#ffffff"}
              emissiveIntensity={0.3}
            />
          </mesh>
        );
      })}

      {/* Bone cylinders */}
      {BONE_CONNECTIONS.map(([a, b]) => {
        const startVec = jointVectors.get(a);
        const endVec = jointVectors.get(b);
        if (!startVec || !endVec) return null;
        return <Bone key={`${a}-${b}`} start={startVec} end={endVec} />;
      })}
    </group>
  );
}
