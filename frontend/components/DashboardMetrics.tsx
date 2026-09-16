"use client";

import { useEffect, useState } from "react";
import type { Phase, SlamResult } from "../lib/api";

interface Props {
  result: SlamResult | null;
  phase: Phase;
  liveElapsed: number;
}

function useCountUp(target: number, duration = 900): number {
  const [value, setValue] = useState(0);
  useEffect(() => {
    let raf = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const f = Math.min(1, (t - t0) / duration);
      setValue(target * (1 - Math.pow(1 - f, 3)));
      if (f < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return value;
}

function Metric({
  label,
  display,
  raw,
  decimals = 0,
  suffix = "",
  live = false,
}: {
  label: string;
  display: string;
  raw: number;
  decimals?: number;
  suffix?: string;
  live?: boolean;
}) {
  const animated = useCountUp(raw);
  const text =
    display !== ""
      ? display
      : decimals === 0
        ? Math.round(animated).toLocaleString() + suffix
        : animated.toFixed(decimals) + suffix;
  return (
    <div className="metric">
      <div className="metric-label">
        {label}
        {live && <span className="metric-live" />}
      </div>
      <div className="metric-value">{text}</div>
    </div>
  );
}

function fmtInt(n: number): string {
  return Math.round(n).toLocaleString();
}

/** Compact professional metrics dashboard. All values come from the backend. */
export default function DashboardMetrics({ result, phase, liveElapsed }: Props) {
  const busy = phase === "uploading" || phase === "processing";
  const has = result !== null;

  const processingTime = has ? result.processing_time_sec : busy ? liveElapsed : 0;
  const avgFps = has ? (result.avg_fps || 0) : 0;
  const mapPoints = has ? (result.map_points || result.point_count) : 0;
  const keyframes = has ? result.keyframe_count : 0;
  const frames = has ? (result.total_frames || result.frame_count) : 0;

  return (
    <section className="card metrics-card" aria-label="SLAM dashboard metrics">
      <div className="metrics-head">
        <h2>SLAM Dashboard</h2>
        <span
          className={
            "pill " +
            (phase === "done"
              ? "pill-done"
              : busy
                ? "pill-busy"
                : phase === "error"
                  ? "pill-error"
                  : "pill-idle")
          }
        >
          <span className="pill-dot" />
          {phase === "done"
            ? "DONE"
            : busy
              ? "PROCESSING"
              : phase === "error"
                ? "ERROR"
                : "IDLE"}
        </span>
      </div>
      <div className="metrics-grid">
        <Metric
          label="Process time"
          display={
            has
              ? `${result.processing_time_sec.toFixed(1)} s`
              : busy
                ? `${liveElapsed.toFixed(1)} s`
                : "—"
          }
          raw={processingTime}
          live={busy && !has}
        />
        <Metric
          label="Avg FPS"
          display={has ? `${avgFps.toFixed(1)} FPS` : "—"}
          raw={avgFps}
          suffix=" FPS"
        />
        <Metric
          label="Map points"
          display={has ? fmtInt(mapPoints) : "—"}
          raw={mapPoints}
        />
        <Metric
          label="Keyframes"
          display={has ? fmtInt(keyframes) : "—"}
          raw={keyframes}
        />
        <Metric
          label="Frames processed"
          display={has ? fmtInt(frames) : "—"}
          raw={frames}
        />
      </div>
      {!has && !busy && (
        <p className="muted metrics-note">
          Upload a video and run SLAM — live backend metrics appear here.
        </p>
      )}
    </section>
  );
}
