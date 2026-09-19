"use client";

import { useEffect, useRef, useState } from "react";
import {
  SAMPLE_VIDEO_FILENAME,
  STAGE_ORDER,
  fetchProgress,
  stageLabel,
  type Phase,
  type SlamProgress,
  type SlamResult,
} from "../lib/api";

interface Props {
  file: File | null;
  previewUrl: string | null;
  busy: boolean;
  result: SlamResult | null;
  phase: Phase;
  progress: number | null; // 0-100 during upload
  elapsedSec: number;
  message: string;
  onSelect: (file: File | null) => void;
  onRun: () => void;
  onUseSample: () => void;
  sampleLoading: boolean;
  // Fired once when the backend initiates its very first stage after the
  // upload has completed. The parent uses it to start the elapsed timer.
  onFirstStage?: () => void;
}

interface VideoMeta {
  duration: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  frames: number | null;
}

function fmtDuration(s: number | null): string {
  if (s === null || !Number.isFinite(s)) return "—";
  return `${s.toFixed(2)} s`;
}

/** Input video panel: upload + preview + real file/stream metadata + live run strip. */
export default function VideoUpload({
  file,
  previewUrl,
  busy,
  result,
  phase,
  progress,
  elapsedSec,
  message,
  onSelect,
  onRun,
  onUseSample,
  sampleLoading,
  onFirstStage,
}: Props) {
  const pick = (picked: File | null) => {
    if (picked && !picked.name.toLowerCase().endsWith(".mp4")) {
      alert("Only .mp4 videos are accepted.");
      return;
    }
    onSelect(picked);
  };

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [meta, setMeta] = useState<VideoMeta>({
    duration: null,
    width: null,
    height: null,
    fps: null,
    frames: null,
  });
  const [live, setLive] = useState<SlamProgress | null>(null);

  // Local stream metadata from the selected file (real, via the browser).
  useEffect(() => {
    setMeta({ duration: null, width: null, height: null, fps: null, frames: null });
  }, [previewUrl]);

  const firstStageNotifiedRef = useRef(false);
  const onFirstStageRef = useRef(onFirstStage);
  onFirstStageRef.current = onFirstStage;

  // Poll the real backend stage while SLAM crunches. The first time the
  // backend reports an active stage, the upload is fully received and the
  // pipeline has initiated — notify the parent once so it can start the
  // elapsed timer from that moment.
  useEffect(() => {
    if (phase !== "processing") {
      if (phase !== "uploading") setLive(null);
      if (phase === "idle" || phase === "uploading") {
        firstStageNotifiedRef.current = false;
      }
      return;
    }
    let stop = false;
    let fallbackIdx = 0;
    const notifyFirstStageOnce = () => {
      if (!firstStageNotifiedRef.current) {
        firstStageNotifiedRef.current = true;
        onFirstStageRef.current?.();
      }
    };
    const tick = async () => {
      const p = await fetchProgress();
      if (stop) return;
      if (p === null) {
        // Backend endpoint unreachable — cycle the real phase names as an
        // indeterminate indicator (no fake percent attached). The upload
        // itself is complete, so this is the best available start signal.
        fallbackIdx = (fallbackIdx + 1) % STAGE_ORDER.length;
        setLive({
          active: true,
          stage: STAGE_ORDER[fallbackIdx],
          stage_index: 0,
          stage_total: STAGE_ORDER.length,
          percent: 0,
        });
        notifyFirstStageOnce();
        return;
      }
      if (p.active) {
        // Genuine first-stage signal: the backend has received the upload
        // and initiated the pipeline (probe → …). Start the timer here.
        setLive(p);
        notifyFirstStageOnce();
        return;
      }
      // Reachable but idle/done/error: either the new run has not initiated
      // yet (stale previous result) or it already finished. Show a waiting
      // state without starting the timer — the timer must not include
      // upload time or stale runs.
      setLive({
        active: true,
        stage: "loading video",
        stage_index: 0,
        stage_total: STAGE_ORDER.length,
        percent: 0,
      });
    };
    tick();
    const t = setInterval(tick, 600);
    return () => {
      stop = true;
      clearInterval(t);
    };
  }, [phase]);

  const onLoadedMetadata = () => {
    const v = videoRef.current;
    if (!v) return;
    setMeta((m) => ({
      ...m,
      duration: Number.isFinite(v.duration) ? v.duration : null,
      width: v.videoWidth || null,
      height: v.videoHeight || null,
    }));
  };

  // Backend probe values after a run are the authoritative FPS/frame count.
  const fps = result?.input_fps || meta.fps;
  const frames = result?.input_frames || meta.frames;
  const duration = result?.video_duration_sec ?? meta.duration;
  const width = result?.input_width || meta.width;
  const height = result?.input_height || meta.height;

  const showStageCount =
    phase === "processing" && live && live.stage_index > 0 && live.stage_total > 0;

  return (
    <section className="card" aria-label="Input video">
      <div className="metrics-head">
        <h2>Input Video</h2>
        {busy && (
          <span className="pill pill-busy">
            <span className="pill-dot" />
            PROCESSING
          </span>
        )}
      </div>
      <div className="input-split">
        <div className="input-pane">
          <label className="file-row">
            <input
              type="file"
              accept="video/mp4,.mp4"
              disabled={busy}
              onChange={(e) => pick(e.target.files?.[0] ?? null)}
            />
          </label>
          <button
            className="btn btn-sample"
            type="button"
            disabled={busy || sampleLoading}
            onClick={onUseSample}
            title={`Load ${SAMPLE_VIDEO_FILENAME} from the backend as the input`}
          >
            {sampleLoading
              ? "Loading sample…"
              : `Use default testing video (${SAMPLE_VIDEO_FILENAME})`}
          </button>
          {previewUrl && (
            <video
              ref={videoRef}
              className="preview preview-compact"
              src={previewUrl}
              controls
              muted
              playsInline
              preload="metadata"
              onLoadedMetadata={onLoadedMetadata}
            />
          )}
          {!file && <p className="muted">Select an .mp4 file to begin.</p>}
        </div>
        <div className="input-pane">
          {file ? (
            <div className="file-meta">
              <div className="file-name" title={file.name}>
                {file.name}
              </div>
              <div className="file-grid-split">
                <div className="file-grid">
                  <span className="muted">Size</span>
                  <span>{(file.size / 1024 / 1024).toFixed(1)} MB</span>
                  <span className="muted">Duration</span>
                  <span>{duration !== null && duration !== undefined ? fmtDuration(Number(duration)) : "—"}</span>
                  <span className="muted">Resolution</span>
                  <span>{width && height ? `${width}×${height}` : "—"}</span>
                </div>
                <div className="file-grid">
                  <span className="muted">FPS</span>
                  <span>{fps ? Number(fps).toFixed(2) : "—"}</span>
                  <span className="muted">Frames</span>
                  <span>{frames ? Number(frames).toLocaleString() : "—"}</span>
                </div>
              </div>
            </div>
          ) : (
            <p className="muted">File details appear here.</p>
          )}
          <button className="btn btn-run" disabled={!file || busy} onClick={onRun}>
            {busy ? "Running SLAM…" : "Run SLAM"}
          </button>
        </div>
      </div>

      {phase === "uploading" && progress !== null && (
        <div className="run-strip">
          <div className="bar">
            <div className="bar-fill" style={{ width: `${progress}%` }} />
          </div>
          <p className="muted">{progress}% uploaded</p>
        </div>
      )}

      {phase === "processing" && (
        <div className="run-strip">
          <div className="status-row">
            <span className="spinner" aria-label="processing" />
            <strong>{stageLabel(live?.stage ?? "features")}</strong>
          </div>
          {showStageCount ? (
            <p className="muted">
              Stage {live!.stage_index}/{live!.stage_total}
              {live!.percent > 0 && ` · ${live!.percent.toFixed(0)}% phases complete`} ·{" "}
              {elapsedSec.toFixed(1)} s elapsed
            </p>
          ) : (
            <p className="muted">{elapsedSec.toFixed(1)} s elapsed</p>
          )}
          <div className="bar bar-indeterminate" aria-hidden="true">
            <div className="bar-fill bar-slide" />
          </div>
        </div>
      )}

      {phase === "error" && <p className="error-text run-strip">{message || "Processing failed."}</p>}
    </section>
  );
}
