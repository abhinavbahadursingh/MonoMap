"use client";

import { useEffect, useRef, useState } from "react";
import {
  SAMPLE_VIDEO_FILENAME,
  STAGE_ORDER,
  fetchProgress,
  skipOptimization,
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
  sampleProgress: number | null; // 0-100 while fetching default video
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
  sampleProgress,
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
  const [skipping, setSkipping] = useState(false);
  const [skipDone, setSkipDone] = useState(false);

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

  // Detect backend skip (after result lands: phase_params or result flags)
  useEffect(() => {
    if (result?.optimization_skipped) setSkipDone(true);
    else if (phase === "idle") setSkipDone(false);
  }, [result, phase]);

  const handleSkip = async () => {
    if (skipping || skipDone) return;
    // If SLAM hasn't started, start it and then request skip (faster on weak CPU/GPU)
    if (phase === "idle" || phase === "error" || phase === "done") {
      if (!file) {
        alert("Select an .mp4 or use the default testing video first.");
        return;
      }
      setSkipping(true);
      setSkipDone(true);
      // Fire a pre-flag (in case the backend hasn't yet reset) and start the job
      skipOptimization();
      onRun();
      // The pipeline resets the skip flag at job start, so re-request shortly after
      setTimeout(async () => {
        const ok = await skipOptimization();
        if (!ok) setSkipDone(false);
        setSkipping(false);
      }, 600);
      return;
    }
    setSkipping(true);
    const ok = await skipOptimization();
    if (ok) setSkipDone(true);
    setTimeout(() => setSkipping(false), 1200);
    if (!ok) setSkipping(false);
  };

  // Backend probe values after a run are the authoritative FPS/frame count.
  // Known specs for the bundled sample (backend/video1.mp4) so Size /
  // Duration / Resolution / FPS / Frames can render beside the preview
  // even before the browser reports metadata or SLAM has run.
  const isSample = file?.name === SAMPLE_VIDEO_FILENAME;
  const fps = result?.input_fps || meta.fps || (isSample ? 25 : null);
  const duration = result?.video_duration_sec ?? meta.duration ?? (isSample ? 8.96 : null);
  const width = result?.input_width || meta.width || (isSample ? 3840 : null);
  const height = result?.input_height || meta.height || (isSample ? 2160 : null);
  const frames =
    result?.input_frames ||
    meta.frames ||
    (duration && fps ? Math.round(Number(duration) * Number(fps)) : null) ||
    (isSample ? 224 : null);
  const sizeMb = file ? file.size / 1024 / 1024 : isSample ? 42.8 : null;

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
      <div className="input-stack">
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
            ? sampleProgress !== null
              ? `Fetching video ${sampleProgress}% fetched`
              : "Fetching video…"
            : `Use default testing video (${SAMPLE_VIDEO_FILENAME})`}
        </button>
        {sampleLoading && (
          <div className="run-strip">
            {sampleProgress !== null ? (
              <>
                <div className="bar">
                  <div className="bar-fill" style={{ width: `${sampleProgress}%` }} />
                </div>
                <p className="muted">Fetching video {sampleProgress}% fetched</p>
              </>
            ) : (
              <>
                <div className="bar bar-indeterminate" aria-hidden="true">
                  <div className="bar-fill bar-slide" />
                </div>
                <p className="muted">Fetching video…</p>
              </>
            )}
          </div>
        )}
        {previewUrl ? (
          <div className="video-side">
            <video
              ref={videoRef}
              className="preview preview-side"
              src={previewUrl}
              controls
              muted
              playsInline
              preload="metadata"
              onLoadedMetadata={onLoadedMetadata}
            />
            <div className="file-meta file-meta-side">
              <div className="file-name" title={file?.name ?? ""}>
                {file?.name ?? "video1.mp4"}
              </div>
              <dl className="spec-list">
                <div className="spec-row">
                  <dt className="muted">Size</dt>
                  <dd>{sizeMb !== null ? `${sizeMb.toFixed(1)} MB` : "—"}</dd>
                </div>
                <div className="spec-row">
                  <dt className="muted">Duration</dt>
                  <dd>{duration !== null && duration !== undefined ? fmtDuration(Number(duration)) : "—"}</dd>
                </div>
                <div className="spec-row">
                  <dt className="muted">Resolution</dt>
                  <dd>{width && height ? `${width}×${height}` : "—"}</dd>
                </div>
                <div className="spec-row">
                  <dt className="muted">FPS</dt>
                  <dd>{fps ? Number(fps).toFixed(2) : "—"}</dd>
                </div>
                <div className="spec-row">
                  <dt className="muted">Frames</dt>
                  <dd>{frames ? Number(frames).toLocaleString() : "—"}</dd>
                </div>
              </dl>
            </div>
          </div>
        ) : (
          <p className="muted">Select an .mp4 file to begin.</p>
        )}
        <button className="btn btn-run" disabled={!file || busy} onClick={onRun}>
          {busy ? "Running SLAM…" : "Run SLAM"}
        </button>
        <div className="run-slam-tip-row">
          <p className="muted run-slam-tip">
            Tip: you can skip the 11th step (Optimization) by clicking the Skip Optimization button down
            in Phase Parameters — faster result on Render&apos;s weak CPU/GPU.
          </p>
          <button
            className="btn btn-skip-mini"
            type="button"
            onClick={handleSkip}
            disabled={skipping || skipDone || (!file && (phase === "idle" || phase === "error"))}
            title="Skip pose optimization (Phase 11) and use trajectory from previous phases — starts processing if idle"
          >
            {skipDone ? "Skipped ✓" : skipping ? "Skipping…" : "Skip 11th step"}
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
