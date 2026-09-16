"use client";

import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

interface Props {
  points: number[][];
}

/** Height ramp: deep blue (low) -> cyan -> warm yellow (high). */
function heightColor(t: number): [number, number, number] {
  const lo: [number, number, number] = [0.15, 0.3, 1.0];
  const mid: [number, number, number] = [0.0, 1.0, 1.0];
  const hi: [number, number, number] = [1.0, 0.85, 0.2];
  const a = t < 0.5 ? lo : mid;
  const b = t < 0.5 ? mid : hi;
  const f = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function Cloud({ points }: Props) {
  const { geometry, pointSize } = useMemo(() => {
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    let loY = Infinity;
    let hiY = -Infinity;
    for (const p of points) {
      if (p[1] < loY) loY = p[1];
      if (p[1] > hiY) hiY = p[1];
    }
    points.forEach((p, i) => {
      v.set(p[0], p[1], p[2]);
      box.expandByPoint(v);
      positions[i * 3] = p[0];
      positions[i * 3 + 1] = p[1];
      positions[i * 3 + 2] = p[2];
      const [r, g, b] = heightColor(hiY > loY ? (p[1] - loY) / (hiY - loY) : 0.5);
      colors[i * 3] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const size = box.getSize(new THREE.Vector3());
    return { geometry, pointSize: Math.max(size.length() / 400, 0.02) };
  }, [points]);

  return (
    <points geometry={geometry}>
      <pointsMaterial size={pointSize} vertexColors sizeAttenuation />
    </points>
  );
}

export default function PointCloudViewer({ points }: Props) {
  const framing = useMemo(() => {
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (const p of points) box.expandByPoint(v.set(p[0], p[1], p[2]));
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-6);
    return {
      center,
      radius,
      position: new THREE.Vector3(
        center.x + radius * 0.9,
        center.y + radius * 0.6,
        center.z + radius * 1.2,
      ),
    };
  }, [points]);

  if (points.length === 0) {
    return <div className="viewer-empty">No points to display.</div>;
  }
  const gridSize = framing.radius * 4;
  return (
    <div className="viewer">
      {/* Remount per dataset so the camera refits (camera props apply on mount). */}
      <Canvas
        key={points.length}
        camera={{
          position: framing.position.toArray(),
          fov: 55,
          near: Math.max(framing.radius / 1000, 0.01),
          far: framing.radius * 20,
        }}
        dpr={[1, 2]}
      >
        <color attach="background" args={["#0d1117"]} />
        <Cloud points={points} />
        <gridHelper
          args={[gridSize, 20, 0x333333, 0x1f2937]}
          position={[framing.center.x, framing.center.y - framing.radius, framing.center.z]}
        />
        <OrbitControls target={framing.center} makeDefault />
      </Canvas>
      <p className="muted viewer-caption">
        {points.length.toLocaleString()} points · drag to orbit · wheel to zoom · right-drag to pan
      </p>
    </div>
  );
}
