import { Grid } from "@react-three/drei";

/**
 * Flat grid ground plane rendered at y=0.
 *
 * Uses drei's Grid helper for an infinite-looking grid with fade-out.
 */
export function GroundPlane() {
  return (
    <Grid
      position={[0, 0, 0]}
      args={[10, 10]}
      cellSize={0.5}
      cellThickness={0.5}
      cellColor="#4a5568"
      sectionSize={2}
      sectionThickness={1}
      sectionColor="#718096"
      fadeDistance={15}
      fadeStrength={1.5}
      infiniteGrid
    />
  );
}
