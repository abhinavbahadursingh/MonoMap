"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";
import DashboardMetrics from "../components/DashboardMetrics";
import FeatureTrackingPanel from "../components/FeatureTrackingPanel";
import PhaseParameters from "../components/PhaseParameters";
import VideoUpload from "../components/VideoUpload";
import { uploadVideo, type Phase, type SlamResult } from "../lib/api";

// three.js touches WebGL: client-only, never SSR.
const PointCloudViewerDynamic = dynamic(() => import("../components/PointCloudViewer"), {
  ssr: false,
  loading: () => <div className="viewer-empty">Loading 3D view…</div>,
});
const TrajectoryViewerDynamic = dynamic(() => import("../components/TrajectoryViewer"), {
  ssr: false,
  loading: () => <div className="viewer-empty">Loading 3D view…</div>,
});

function Stat({
  value,
  decimals = 0,
  prefix = "",
  suffix = "",
  label,
  sub,
}: {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  label: string;
  sub?: string;
}) {
  const [animated, setAnimated] = useState(0);
  useEffect(() => {
    let raf = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const f = Math.min(1, (t - t0) / 900);
      setAnimated(value * (1 - Math.pow(1 - f, 3)));
      if (f < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value]);
  const text =
    prefix +
    (decimals === 0
      ? Math.round(animated).toLocaleString()
      : animated.toFixed(decimals)) +
    suffix;
  return (
    <div className="stat">
      <div className="value">{text}</div>
      <div className="label">
        {label}
        {sub ? ` (${sub})` : ""}
      </div>
    </div>
  );
}

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [message, setMessage] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<SlamResult | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startRef = useRef(0);
  const viewsRef = useRef<HTMLDivElement | null>(null);
  const busy = phase === "uploading" || phase === "processing";
  const [viewsVisible, setViewsVisible] = useState(false);

  // Hide the floating button once the 3D results are on screen.
  useEffect(() => {
    const el = viewsRef.current;
    if (!el || phase !== "done" || !result) {
      setViewsVisible(false);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => setViewsVisible(entry.isIntersecting),
      { threshold: 0.15 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [phase, result]);

  const scrollToResults = useCallback(() => {
    viewsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  const stopTimer = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
  };
  useEffect(() => stopTimer, []);

  const startTimer = () => {
    stopTimer();
    startRef.current = performance.now();
    setElapsed(0);
    timerRef.current = setInterval(() => {
      setElapsed((performance.now() - startRef.current) / 1000);
    }, 250);
  };

  const handleSelect = useCallback(
    (picked: File | null) => {
      setFile(picked);
      setPreviewUrl((old) => {
        if (old) URL.revokeObjectURL(old);
        return picked ? URL.createObjectURL(picked) : null;
      });
      setResult(null);
      setMessage("");
      if (phase !== "uploading" && phase !== "processing") setPhase("idle");
    },
    [phase],
  );

  const handleRun = useCallback(async () => {
    if (!file || busy) return;
    setResult(null);
    setMessage("");
    setPhase("uploading");
    setProgress(0);
    startTimer();
    try {
      const res = await uploadVideo(file, (pct) => {
        setProgress(pct);
        if (pct >= 100) setPhase("processing");
      });
      setResult(res);
      setPhase("done");
      setMessage("");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Upload failed.");
      setPhase("error");
    } finally {
      stopTimer();
      setElapsed((performance.now() - startRef.current) / 1000);
    }
  }, [file, busy]);

  return (
    <main className="page">
      <header className="hero">
        <span className="hero-badge">Monocular Visual Odometry</span>
        <h1>videoSparse SLAM</h1>
        <p>Upload an .mp4 — probe → features → matching → motion → map → optimize, then inspect the 3D result.</p>
      </header>

      {/* TOP — live backend metrics dashboard */}
      <DashboardMetrics result={result} phase={phase} liveElapsed={elapsed} />

      {/* WORKFLOW — input video + tracking summary + 2D tracking, side by side */}
      <div className="aside-grid">
        <VideoUpload
          file={file}
          previewUrl={previewUrl}
          busy={busy}
          result={result}
          phase={phase}
          progress={progress}
          elapsedSec={elapsed}
          message={message}
          onSelect={handleSelect}
          onRun={handleRun}
        />
        <section className="card" aria-label="Tracking and map summary">
          <div className="metrics-head">
            <h2>Tracking &amp; Map Summary</h2>
            <span
              className={
                "pill " +
                (result?.tracking_status === "ACTIVE"
                  ? "pill-done"
                  : result
                    ? "pill-warn"
                    : busy
                      ? "pill-busy"
                      : "pill-idle")
              }
            >
              <span className="pill-dot" />
              {result ? result.tracking_status : busy ? "WORKING" : "STANDBY"}
            </span>
          </div>
          {result ? (
            <div className="summary-list">
              <div className="summary-row">
                <span className="muted">Mean features / frame</span>
                <strong>{result.tracked_features.toLocaleString()}</strong>
              </div>
              <div className="summary-row">
                <span className="muted">ORB range</span>
                <span>
                  {result.orb_stats.min} / {result.orb_stats.mean} / {result.orb_stats.max}
                </span>
              </div>
              <div className="summary-row">
                <span className="muted">Reliable frames</span>
                <span>
                  {result.orb_stats.reliable_frames}/{result.orb_stats.total_frames}
                </span>
              </div>
              <div className="summary-row">
                <span className="muted">Mean good matches / pair</span>
                <span>{Number(result.match_stats.mean).toFixed(1)}</span>
              </div>
              <div className="summary-row">
                <span className="muted">Realtime factor</span>
                <span>×{result.realtime_factor.toFixed(2)}</span>
              </div>
              <div className="summary-row">
                <span className="muted">Video duration</span>
                <span>{result.video_duration_sec.toFixed(2)} s</span>
              </div>
            </div>
          ) : (
            <p className="muted">
              {busy
                ? "Computing ORB features, matches and poses — real numbers land here when SLAM finishes."
                : "Run SLAM to populate tracking statistics from the backend."}
            </p>
          )}
        </section>
        <FeatureTrackingPanel previewUrl={previewUrl} result={result} phase={phase} />
      </div>

      {/* ALL PHASES — live parameters + terminal output from the pipeline */}
      <div className="phase-section">
        <PhaseParameters result={result} phase={phase} />
      </div>

      {result && (
        <div className="result-enter" key={`${result.processing_time_sec}-${result.point_count}`}>
          <section className="card">
            <h2>SLAM result</h2>
            <div className="stats-grid">
              <Stat value={result.processing_time_sec} decimals={1} suffix=" s" label="processing time" />
              <Stat value={result.frame_count} label="frames" sub={`of ${result.input_frames} uploaded`} />
              <Stat value={result.point_count} label="3D points" sub={`${result.keyframe_count} keyframes`} />
              <Stat value={result.realtime_factor} decimals={2} prefix="×" label="realtime factor" />
            </div>
          </section>

          <div className="views-grid" ref={viewsRef} id="slam-views" style={{ scrollMarginTop: "16px" }}>
            <section className="card">
              <h2>3D Sparse Point Cloud</h2>
              <PointCloudViewerDynamic points={result.points} />
              <p className="muted viewer-meta">
                {(result.map_points || result.point_count).toLocaleString()} 3D points ·{" "}
                {result.keyframe_count} keyframes · {result.frame_count} frames processed
              </p>
            </section>
            <section className="card">
              <h2>Camera Trajectory</h2>
              <TrajectoryViewerDynamic trajectory={result.trajectory} />
              <p className="muted viewer-meta">
                {result.trajectory.length} poses · {result.keyframe_count} keyframes ·{" "}
                {(result.avg_fps || 0).toFixed(1)} FPS avg
              </p>
            </section>
          </div>
        </div>
      )}

      {phase === "done" && result && !viewsVisible && (
        <div className="scroll-fab-wrap">
          <button className="btn scroll-fab" onClick={scrollToResults}>
            View 3D Results ↓
          </button>
        </div>
      )}
    </main>
  );
}
