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

    // One slim, bright tube so the path reads clearly at any zoom.
    // (Plain THREE.Line is always 1px and washes out.)
    let tubeGeo: THREE.BufferGeometry | null = null;
    if (pts.length >= 2) {
      const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal");
      const segs = Math.min(300, Math.max(64, pts.length * 3));
      tubeGeo = new THREE.TubeGeometry(
        curve,
        segs,
        Math.max(radius * 0.008, 0.006),
        8,
        false,
      );
    }

    return {
      center,
      radius,
      tubeGeo,
      start: pts[0],
      end: pts[pts.length - 1],
      markerSize: Math.max(radius / 45, 0.025),
      position: new THREE.Vector3(
        center.x + radius * 0.9,
        center.y + radius * 0.7,
        center.z + radius * 1.3,
      ),
    };
  }, [trajectory]);

  useEffect(() => () => scene.tubeGeo?.dispose(), [scene]);

  if (trajectory.length === 0) {
    return <div className="viewer-empty">No trajectory to display.</div>;
  }
  const n = trajectory.length;
  const gridSize = scene.radius * 4;

  return (
    <div className="viewer">
      {/* Minimal axis legend */}
      <div className="viewer-axes" aria-label="Coordinate axes">
        <span className="axis-item">
          <i className="axis-dot axis-x" />X
        </span>
        <span className="axis-item">
          <i className="axis-dot axis-y" />Y
        </span>
        <span className="axis-item">
          <i className="axis-dot axis-z" />Z
        </span>
      </div>

      {/* Remount per dataset so the camera refits (camera props apply on mount). */}
      <Canvas
        key={n}
        camera={{
          position: scene.position.toArray(),
          fov: 55,
          near: Math.max(scene.radius / 1000, 0.01),
          far: scene.radius * 20,
        }}
        dpr={[1, 2]}
      >
        <color attach="background" args={["#0b0f16"]} />
        <ambientLight intensity={0.6} />

        {/* The camera path — single bright line */}
        {scene.tubeGeo && (
          <mesh geometry={scene.tubeGeo}>
            <meshBasicMaterial color="#22d3ee" toneMapped={false} />
          </mesh>
        )}

        {/* Start / end only */}
        <mesh position={scene.start}>
          <sphereGeometry args={[scene.markerSize, 16, 16]} />
          <meshBasicMaterial color="#22ff66" toneMapped={false} />
        </mesh>
        <mesh position={scene.end}>
          <sphereGeometry args={[scene.markerSize, 16, 16]} />
          <meshBasicMaterial color="#ff4444" toneMapped={false} />
        </mesh>

        <axesHelper args={[scene.radius * 0.6]} position={scene.center} />
        <gridHelper
          args={[gridSize, 20, 0x2a3348, 0x161c28]}
          position={[scene.center.x, scene.center.y - scene.radius, scene.center.z]}
        />
        <OrbitControls target={scene.center} makeDefault />
      </Canvas>
      <p className="muted viewer-caption">
        {n} poses · <span className="dot-start">●</span> start{" "}
        <span className="dot-end">●</span> end · drag to orbit · wheel to zoom · right-drag to pan
      </p>
    </div>
  );
}
