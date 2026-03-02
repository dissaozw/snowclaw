import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";
import type { JointPositions } from "../types";
import { JOINT_NAMES } from "../types";
import { GroundPlane } from "./GroundPlane";
import { Skeleton } from "./Skeleton";

interface ViewerProps {
  /** Joint positions for the current frame; null when no data. */
  pose: JointPositions | null;
}

/**
 * Compute the center of mass (average of all joint positions).
 */
function computeCenterOfMass(pose: JointPositions): THREE.Vector3 {
  const center = new THREE.Vector3(0, 0, 0);
  for (const name of JOINT_NAMES) {
    const [x, y, z] = pose[name];
    center.add(new THREE.Vector3(x, y, z));
  }
  center.divideScalar(JOINT_NAMES.length);
  return center;
}

/**
 * 3D scene setup with camera, lighting, orbit controls, ground plane,
 * and the skeleton visualization.
 *
 * OrbitControls target the skeleton's center of mass for intuitive
 * rotation, zoom, and pan around the skeleton.
 */
export function Viewer({ pose }: ViewerProps) {
  const centerOfMass = useMemo(() => {
    if (!pose) return new THREE.Vector3(0, 1, 0);
    return computeCenterOfMass(pose);
  }, [pose]);

  return (
    <Canvas
      camera={{
        position: [2, 1.5, 3],
        fov: 50,
        near: 0.01,
        far: 100,
      }}
      style={{ width: "100%", height: "100%" }}
      gl={{ antialias: true }}
    >
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 10, 5]} intensity={0.8} castShadow />
      <directionalLight position={[-3, 5, -5]} intensity={0.3} />
      <hemisphereLight
        args={["#b1e1ff", "#443322", 0.5]}
      />

      {/* Background color */}
      <color attach="background" args={["#1a1a2e"]} />

      {/* Orbit controls centered on skeleton */}
      <OrbitControls
        target={centerOfMass}
        enableDamping
        dampingFactor={0.1}
        minDistance={0.5}
        maxDistance={20}
      />

      {/* Ground plane */}
      <GroundPlane />

      {/* Skeleton */}
      {pose && <Skeleton pose={pose} />}
    </Canvas>
  );
}
