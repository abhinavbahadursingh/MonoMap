"use client";

import { useEffect, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

interface Props {
  trajectory: number[][];
}

export default function TrajectoryViewer({ trajectory }: Props) {
  const scene = useMemo(() => {
    const pts = trajectory.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    const box = new THREE.Box3().setFromPoints(pts);
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-6);
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(pts);
    const markerGeometry = new THREE.BufferGeometry().setFromPoints(pts);
    const lineMaterial = new THREE.LineBasicMaterial({ color: "#00ff88" });
    const line = new THREE.Line(lineGeometry, lineMaterial);
    return {
      center,
      radius,
      position: new THREE.Vector3(
        center.x + radius * 0.9,
        center.y + radius * 0.7,
        center.z + radius * 1.3,
      ),
      line,
      lineGeometry,
      markerGeometry,
      lineMaterial,
      start: pts[0],
      end: pts[pts.length - 1],
      markerSize: Math.max(radius / 60, 0.02),
    };
  }, [trajectory]);

  useEffect(
    () => () => {
      scene.lineGeometry.dispose();
      scene.markerGeometry.dispose();
      scene.lineMaterial.dispose();
    },
    [scene],
  );

  if (trajectory.length === 0) {
    return <div className="viewer-empty">No trajectory to display.</div>;
  }
  const gridSize = scene.radius * 4;
  return (
    <div className="viewer">
      {/* Remount per dataset so the camera refits (camera props apply on mount). */}
      <Canvas
        key={trajectory.length}
        camera={{
          position: scene.position.toArray(),
          fov: 55,
          near: Math.max(scene.radius / 1000, 0.01),
          far: scene.radius * 20,
        }}
        dpr={[1, 2]}
      >
        <color attach="background" args={["#0d1117"]} />
        <primitive object={scene.line} />
        <points geometry={scene.markerGeometry}>
          <pointsMaterial size={scene.markerSize / 2} color="#9ca3af" sizeAttenuation />
        </points>
        <mesh position={scene.start}>
          <sphereGeometry args={[scene.markerSize, 16, 16]} />
          <meshBasicMaterial color="#22ff66" />
        </mesh>
        <mesh position={scene.end}>
          <sphereGeometry args={[scene.markerSize, 16, 16]} />
          <meshBasicMaterial color="#ff4444" />
        </mesh>
        <gridHelper
          args={[gridSize, 20, 0x333333, 0x1f2937]}
          position={[scene.center.x, scene.center.y - scene.radius, scene.center.z]}
        />
        <OrbitControls target={scene.center} makeDefault />
      </Canvas>
      <p className="muted viewer-caption">
        {trajectory.length} poses · <span className="dot-start">●</span> start{" "}
        <span className="dot-end">●</span> end · drag to orbit · wheel to zoom · right-drag to pan
      </p>
    </div>
  );
}
