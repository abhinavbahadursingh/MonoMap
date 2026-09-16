"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE, type Phase, type SlamResult } from "../lib/api";

interface Props {
  previewUrl: string | null;
  result: SlamResult | null;
  phase: Phase;
}

const NORM_W = 640;
const NORM_H = 360;
const MAX_DRAWN = 120;

function Sparkline({ counts, activeFrame }: { counts: number[]; activeFrame: number | null }) {
  if (!counts || counts.length === 0) return null;
  const w = 220;
  const h = 44;
  const max = Math.max(...counts, 1);
  const step = counts.length > 1 ? w / (counts.length - 1) : w;
  const pts = counts
    .map((c, i) => `${(i * step).toFixed(1)},${(h - (c / max) * (h - 6) - 3).toFixed(1)}`)
    .join(" ");
  const ax = activeFrame !== null && counts.length > 1 ? activeFrame * step : null;
  return (
    <svg
      className="spark"
      width={w}
      height={h}
      viewBox={`0 0 ${w} ${h}`}
      role="img"
      aria-label="ORB keypoints per frame"
    >
      <polyline points={pts} fill="none" stroke="#22d3ee" strokeWidth="1.6" />
      {ax !== null && (
        <line x1={ax} y1={0} x2={ax} y2={h} stroke="#34d399" strokeWidth="1.4" opacity="0.9" />
      )}
    </svg>
  );
}

/**
 * 2D Feature Tracking panel — DYNAMIC.
 * Plays back REAL backend ORB keypoints (tracking_track, 640x360 space)
 * in sync with the video clock: as the video plays, the overlay switches
 * to the nearest sampled backend frame, with pulse + motion-trail ghosts
 * so movement is visible. Nothing is invented — with no backend track the
 * panel shows a placeholder, never synthetic points.
 */
export default function FeatureTrackingPanel({ previewUrl, result, phase }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [hud, setHud] = useState({ frame: 0, count: 0, playing: false });
  const hudRef = useRef(hud);
  hudRef.current = hud;
  const busy = phase === "uploading" || phase === "processing";

  const status = result?.tracking_status ?? (busy ? "PROCESSING" : "STANDBY");
  const statusClass =
    status === "ACTIVE"
      ? "pill-done"
      : status === "DEGRADED"
        ? "pill-warn"
        : status === "LOST"
          ? "pill-error"
          : busy
            ? "pill-busy"
            : "pill-idle";
  const trackedCount =
    result?.tracked_features ??
    (result?.orb_counts?.length
      ? Math.round(
          result.orb_counts.reduce((a, b) => a + b, 0) / result.orb_counts.length,
        )
      : 0);
  const orbCounts = result?.orb_counts ?? [];

  // Better video: the backend-annotated ORB tracking video (keypoints +
  // counts drawn on every frame by the pipeline itself). Falls back to the
  // raw upload preview + canvas overlay when unavailable.
  const trackingUrl = result?.tracking_video_url
    ? `${API_BASE}${result.tracking_video_url}`
    : null;
  const useAnnotated = trackingUrl !== null;
  const videoSrc = trackingUrl ?? previewUrl;

  // Time-ordered backend track (fallback: single middle-frame sample).
  const track = useMemo(() => {
    if (result?.tracking_track?.length) {
      return [...result.tracking_track].sort((a, b) => a.frame - b.frame);
    }
    if (result?.tracking_keypoints?.length) {
      return [{ frame: result.tracking_frame_index ?? 0, points: result.tracking_keypoints }];
    }
    return [];
  }, [result]);

  // Dynamic render loop: video clock -> nearest backend frame -> canvas.
  // With the annotated backend video the keypoints are already drawn, so
  // the loop only keeps the HUD (frame / count / playing) in sync.
  useEffect(() => {
    let raf = 0;
    const loop = (t: number) => {
      raf = requestAnimationFrame(loop);
      const v = videoRef.current;
      if (!v || !result) return;
      // Map video clock -> processed frame index.
      const dur = v.duration || result.video_duration_sec || 0;
      const n = result.frame_count || 1;
      const fIdx =
        dur > 0 ? Math.min(n - 1, Math.max(0, Math.floor((v.currentTime / dur) * n))) : 0;
      if (useAnnotated) {
        const count = orbCounts[fIdx] ?? 0;
        const prevHud = hudRef.current;
        if (
          prevHud.frame !== fIdx ||
          prevHud.count !== count ||
          prevHud.playing !== !v.paused
        ) {
          setHud({ frame: fIdx, count, playing: !v.paused });
        }
        return;
      }
      const c = canvasRef.current;
      if (!c) return;
      // Fit overlay to displayed video box.
      const rect = v.getBoundingClientRect();
      const w = Math.max(1, Math.round(rect.width));
      const h = Math.max(1, Math.round(rect.height));
      if (c.width !== w || c.height !== h) {
        c.width = w;
        c.height = h;
      }
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, c.width, c.height);
      if (!track.length || !result) {
        // Honest empty state: crosshair sweep while busy, nothing otherwise.
        if (busy) {
          const x = ((t / 12) % (c.width + 80)) - 40;
          const g = ctx.createLinearGradient(x - 30, 0, x + 30, 0);
          g.addColorStop(0, "rgba(34,211,238,0)");
          g.addColorStop(0.5, "rgba(34,211,238,0.35)");
          g.addColorStop(1, "rgba(34,211,238,0)");
          ctx.fillStyle = g;
          ctx.fillRect(0, 0, c.width, c.height);
        }
        return;
      }
      // Nearest sampled backend entry at or before fIdx.
      let ti = 0;
      for (let i = 0; i < track.length; i++) {
        if (track[i].frame <= fIdx) ti = i;
        else break;
      }
      const cur = track[ti];
      const prev = track[Math.max(0, ti - 1)];
      const sx = c.width / NORM_W;
      const sy = c.height / NORM_H;

      // Motion-trail ghosts from the previous sampled frame (dim).
      if (prev !== cur && prev.points?.length) {
        ctx.fillStyle = "rgba(34,211,238,0.28)";
        const pn = Math.min(prev.points.length, MAX_DRAWN);
        for (let i = 0; i < pn; i++) {
          const p = prev.points[i];
          if (!Array.isArray(p) || p.length < 2) continue;
          ctx.beginPath();
          ctx.arc(p[0] * sx, p[1] * sy, 1.6, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      // Current frame points: bright, gently pulsing so live motion reads
      // even when the video is paused on one backend sample.
      const pts = cur.points ?? [];
      const cn = Math.min(pts.length, MAX_DRAWN);
      for (let i = 0; i < cn; i++) {
        const p = pts[i];
        if (!Array.isArray(p) || p.length < 2) continue;
        const r = 2.2 + Math.sin(t / 280 + i * 1.7) * 0.7;
        ctx.beginPath();
        ctx.arc(p[0] * sx, p[1] * sy, r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0,255,136,0.92)";
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(0,0,0,0.55)";
        ctx.stroke();
      }
      // Throttled HUD update (avoid re-render every rAF tick).
      const prevHud = hudRef.current;
      if (
        prevHud.frame !== fIdx ||
        prevHud.count !== cn ||
        prevHud.playing !== !v.paused
      ) {
        setHud({ frame: fIdx, count: cn, playing: !v.paused });
      }
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [track, result, busy, useAnnotated, orbCounts]);

  const replay = () => {
    const v = videoRef.current;
    if (!v) return;
    try {
      v.currentTime = 0;
      void v.play();
    } catch {
      /* play is best-effort (autoplay policies) */
    }
  };

  // Autoplay in a loop: the `autoPlay` attribute only fires on mount, so
  // explicitly (re)start playback whenever the video source arrives.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !videoSrc) return;
    v.loop = true;
    v.muted = true;
    const start = () => {
      try {
        const p = v.play();
        if (p && typeof p.catch === "function") p.catch(() => {});
      } catch {
        /* play is best-effort (autoplay policies) */
      }
    };
    if (v.readyState >= 1) {
      start();
    } else {
      v.addEventListener("loadedmetadata", start, { once: true });
      return () => v.removeEventListener("loadedmetadata", start);
    }
  }, [videoSrc]);

  return (
    <section className="card" aria-label="2D feature tracking">
      <div className="metrics-head">
        <h2>2D Feature Tracking</h2>
        <span className={"pill " + statusClass}>
          <span className="pill-dot" />
          TRACKING: {status}
        </span>
      </div>

      {videoSrc ? (
        <div className="track-stage" ref={wrapRef}>
          <video
            key={videoSrc}
            ref={videoRef}
            className="preview track-video"
            src={videoSrc}
            controls
            muted
            playsInline
            preload="metadata"
            loop
            autoPlay={!!result}
          />
          {!useAnnotated && (
            <canvas ref={canvasRef} className="track-overlay" aria-hidden="true" />
          )}
          {result && (useAnnotated || track.length > 0) && (
            <div className="track-hud">
              <span className={"hud-dot" + (hud.playing ? " hud-live" : "")} />
              FRAME {hud.frame}/{Math.max((result.frame_count || 1) - 1, 0)} · {hud.count} pts
              {hud.playing ? " · PLAYING" : " · PAUSED"}
            </div>
          )}
          {useAnnotated && (
            <div className="track-badge">
              BACKEND-ANNOTATED · KEYPOINTS DRAWN PER FRAME
            </div>
          )}
        </div>
      ) : (
        <div className="viewer-empty track-empty">
          No video selected — upload an .mp4 to preview tracking here.
        </div>
      )}

      <div className="track-meta">
        <div className="track-row">
          <span className="muted">Tracked features (mean/frame)</span>
          <strong>{result ? trackedCount.toLocaleString() : "—"}</strong>
        </div>
        <div className="track-row">
          <span className="muted">Overlay</span>
          <span className="muted">
            {useAnnotated
              ? "backend-annotated video · keypoints + counts drawn per frame"
              : result && track.length
                ? `live track · ${track.length} sampled frames · synced to video clock`
                : busy
                  ? "computing ORB features..."
                  : "runs after SLAM"}
          </span>
        </div>
        {result && result.orb_stats && (
          <div className="track-row">
            <span className="muted">
              ORB min/mean/max · {result.orb_stats.min}/{result.orb_stats.mean}/
              {result.orb_stats.max} · reliable {result.orb_stats.reliable_frames}/
              {result.orb_stats.total_frames}
            </span>
          </div>
        )}
        {orbCounts.length > 0 && (
          <div className="track-row track-spark-row">
            <span className="muted">Keypoints / frame</span>
            <Sparkline counts={orbCounts} activeFrame={result ? hud.frame : null} />
          </div>
        )}
        {result && (useAnnotated || track.length > 0) && (
          <div className="track-row">
            <span className="muted">Press play — points follow the video.</span>
            <button className="btn btn-replay" onClick={replay}>
              Replay tracking
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
