"""SLAM entry point: CLI + FastAPI app (one file, two doors).

CLI:
    1. Set INPUT_VIDEO at the top of this file, then run:
           python main.py
    2. Or pass it in the terminal (overrides the variable):
           python main.py --input path/to/video.mp4
           python main.py --input in.mp4 --output out.mp4 --fps 10 --width 640 --height 360

API (served as `main:app`, run from backend/):
           python -m uvicorn main:app --reload --port 8000
    GET  /health     liveness probe
    POST /api/slam   multipart form, field "video", one .mp4 file

The flow itself (Video -> Features -> Matching -> Motion -> Keyframes ->
Triangulation -> Local Map -> Optimization -> Final Map) lives in
pipeline.py (`run_slam`). To attach future work, add a stage to
SLAM_PHASES there — nothing here changes.
"""
from __future__ import annotations

# ================= SET YOUR VIDEO HERE =================
# Put your input video path below, then just run:  python main.py
# (Terminal flags still work and override these values.)
INPUT_VIDEO = r"E:\videoSparse\backend\video1.mp4"          # e.g. r"C:\videos\myvideo.mp4"
OUTPUT_VIDEO = r""         # optional; leave empty for auto (backend/output/<name>_10fps_640x360.mp4)
TARGET_FPS = 10.0
TARGET_WIDTH = 640
TARGET_HEIGHT = 360
# =======================================================

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
# `src/` is not an installed package, so its parent goes on the import path.
# (Deliberate: avoids requiring `pip install -e .` before first run.)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from video_pipeline.context import VideoContext  # noqa: E402
from video_pipeline.stages import probe_video  # noqa: E402

from pipeline import run_slam  # noqa: E402  (orchestration layer, same folder)
from pipeline import SLAM_PHASES  # noqa: E402  (phase names for --skip choices)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extensible video normalization pipeline")
    p.add_argument("--input", "-i", default=None, help="Path to input video file (overrides INPUT_VIDEO)")
    p.add_argument("--output", "-o", default=None,
                   help="Path for normalized output (overrides OUTPUT_VIDEO; default: backend/output/<input_stem>_10fps_640x360.mp4)")
    p.add_argument("--fps", type=float, default=None, help="Target FPS (overrides TARGET_FPS)")
    p.add_argument("--width", type=int, default=None, help="Target width (overrides TARGET_WIDTH)")
    p.add_argument("--height", type=int, default=None, help="Target height (overrides TARGET_HEIGHT)")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=[name for name, _ in SLAM_PHASES],
                   help="Phase(s) to skip, e.g. --skip optimization")
    return p.parse_args(argv)


def default_output(input_path: Path, fps: float, w: int, h: int) -> Path:
    fps_tag = ("%g" % fps).replace(".", "p")
    output_dir = Path(__file__).resolve().parent / "output"  # separate folder in backend
    return output_dir / f"{input_path.stem}_{fps_tag}fps_{w}x{h}.mp4"


def main(argv=None) -> int:
    args = parse_args(argv)
    ctx = resolve_context(args)
    if ctx is None:
        return 2
    try:
        result = run_slam(ctx, skip=args.skip)
    except (OSError, ValueError, RuntimeError) as e:
        # OSError: missing/unreadable files; ValueError: bad settings;
        # RuntimeError: stage invariant violations (loud by design).
        print(f"\n[ERROR] {e}")
        return 1
    print(f"Normalized video saved to: {ctx.output_path}")
    print(f"Artifacts ({len(result['artifacts'])} files) in: "
          f"{ctx.output_path.parent}")
    return 0


def resolve_context(args: argparse.Namespace) -> VideoContext | None:
    """Merge terminal flags over the file variables into a VideoContext.

    Returns None (after printing why) when no input video was provided
    by either source.
    """
    input_str = args.input or INPUT_VIDEO
    output_str = args.output or OUTPUT_VIDEO
    fps = args.fps if args.fps is not None else TARGET_FPS
    width = args.width if args.width is not None else TARGET_WIDTH
    height = args.height if args.height is not None else TARGET_HEIGHT
    if not input_str or not str(input_str).strip():
        print("\n[ERROR] No input video set. Put your video path in INPUT_VIDEO "
              "at the top of main.py, or pass --input <path>.")
        return None 
    input_path = Path(str(input_str).strip())
    if output_str and str(output_str).strip():
        output_path = Path(str(output_str).strip())
    else:
        output_path = default_output(input_path, fps, width, height)
    # VideoContext validates fps/resolution itself (fail fast on bad values).
    return VideoContext(
        input_path=input_path,
        output_path=output_path,
        target_fps=fps,
        target_width=width,
        target_height=height,
    )


# ------------------------------------------------------------------ #
# FastAPI: POST /api/slam over the unchanged SLAM engine.
#
# The SLAM algorithm is only CALLED here — no stage logic lives below,
# and nothing here can change pipeline behavior (same stages, same
# defaults, same outputs). Plain `def` endpoints: FastAPI runs sync
# endpoints in a threadpool, exactly right for a ~10 s CPU-bound run.
# ------------------------------------------------------------------ #

API_TARGET_FPS = 10.0
API_TARGET_WIDTH = 640
API_TARGET_HEIGHT = 360


class SlamResult(BaseModel):
    """POST /api/slam response shape (units: arbitrary, never meters).

    Core fields are the original SS-2 contract (unchanged names/types).
    Extended fields are all derived from the same single pipeline run —
    no second SLAM execution, no fake values.
    """

    processing_time_sec: float = Field(description="Full pipeline wall time")
    frame_count: int = Field(description="Processed (normalized) frames")
    input_frames: int = Field(description="Frames in the uploaded video")
    video_duration_sec: float = Field(description="Uploaded video duration")
    keyframe_count: int = Field(description="Selected keyframes")
    point_count: int = Field(description="Filtered 3D landmarks returned")
    realtime_factor: float = Field(description="video_duration / processing_time")
    trajectory: List[List[float]] = Field(
        description="Per-frame camera centers [x, y, z], frame 0 at origin")
    points: List[List[float]] = Field(
        description="Filtered landmark positions [x, y, z]")
    # --- dashboard aliases + derived real metrics (same run, no re-compute) ---
    avg_fps: float = Field(default=0.0, description="Processed frames per second")
    total_frames: int = Field(default=0, description="Alias of frame_count")
    map_points: int = Field(default=0, description="Alias of point_count")
    input_fps: float = Field(default=0.0, description="Uploaded video FPS")
    input_width: int = Field(default=0, description="Uploaded video width")
    input_height: int = Field(default=0, description="Uploaded video height")
    tracking_status: str = Field(
        default="UNKNOWN",
        description="ACTIVE / DEGRADED / LOST from ORB reliability")
    tracked_features: int = Field(
        default=0, description="Mean ORB keypoints per processed frame")
    tracking_frame_index: int = Field(
        default=0, description="Frame that tracking_keypoints belong to")
    tracking_keypoints: List[List[float]] = Field(
        default_factory=list,
        description="Sampled [x, y] ORB keypoints (normalized 640x360 px)")
    tracking_track: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-frame keypoint sets over time: "
                    "[{frame, points: [[x, y]]}] (normalized 640x360 px)")
    orb_counts: List[int] = Field(
        default_factory=list, description="ORB keypoints per processed frame")
    orb_stats: Dict[str, Any] = Field(
        default_factory=dict, description="min/mean/max + reliable frames")
    match_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Good-match min/mean/max per pair")
    phase_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured parameters reported by each pipeline phase")
    phase_times: Dict[str, float] = Field(
        default_factory=dict, description="Wall time per pipeline phase")
    tracking_video_url: str = Field(
        default="",
        description="Backend-annotated ORB tracking video for this run "
                    "(empty when unavailable)")


# ------------------------------------------------------------------ #
# Live progress state (polled by the frontend while POST /api/slam runs).
#
# POST /api/slam stays synchronous (same SLAM engine, same temp-dir flow).
# FastAPI serves sync endpoints in a threadpool, so GET /api/slam/progress
# keeps answering from the event loop while the upload crunches — the
# percentages below are real completed-phase counts, not estimates.
# ------------------------------------------------------------------ #

_PROGRESS_LOCK = threading.Lock()
_PROGRESS: Dict[str, Any] = {
    "active": False,
    "stage": "idle",
    "stage_index": 0,
    "stage_total": len(SLAM_PHASES),
    "percent": 0.0,
}


def _set_progress(active: bool, stage: str, index: int = 0,
                  total: int = 0, percent: float = 0.0) -> None:
    with _PROGRESS_LOCK:
        _PROGRESS.update({"active": active, "stage": stage,
                          "stage_index": index, "stage_total": total,
                          "percent": percent})


def _report_phase(name: str, index_1_based: int, total: int) -> None:
    percent = round((index_1_based - 1) / total * 100.0, 1) if total else 0.0
    _set_progress(True, name, index_1_based, total, percent)
    _mark_phase_running(name, index_1_based, total)


# ------------------------------------------------------------------ #
# Live per-phase parameters + terminal output (polled while running).
#
# pipeline.run_slam captures each phase's stdout and reports it here via
# phase_log_callback. The frontend polls GET /api/slam/phases to render
# every phase's parameters live — the text below is the phases' own
# terminal output, never synthesized.
# ------------------------------------------------------------------ #

PHASE_LABELS: Dict[str, str] = {
    "probe": "Probe input video",
    "normalize": "Normalize frames",
    "features": "ORB feature extraction",
    "matching": "Descriptor matching",
    "motion": "Camera motion estimation",
    "trajectory": "Trajectory chaining",
    "triangulation": "Triangulation (sparse map)",
    "filter": "Point cloud filtering",
    "keyframes": "Keyframe selection",
    "local_map": "Local mapping",
    "optimization": "Pose optimization",
}

MAX_PHASE_LOG_CHARS = 8000

_PHASE_LOCK = threading.Lock()
_PHASES: List[Dict[str, Any]] = [
    {"name": name, "label": PHASE_LABELS.get(name, name), "index": i + 1,
     "total": len(SLAM_PHASES), "status": "pending", "log": "",
     "elapsed_sec": 0.0}
    for i, (name, _) in enumerate(SLAM_PHASES)
]
_PHASES_ACTIVE = False


def _reset_phases() -> None:
    global _PHASES_ACTIVE
    with _PHASE_LOCK:
        _PHASES_ACTIVE = True
        for i, (name, _) in enumerate(SLAM_PHASES):
            _PHASES[i] = {"name": name, "label": PHASE_LABELS.get(name, name),
                          "index": i + 1, "total": len(SLAM_PHASES),
                          "status": "pending", "log": "", "elapsed_sec": 0.0}


def _mark_phase_running(name: str, index_1_based: int, total: int) -> None:
    with _PHASE_LOCK:
        for entry in _PHASES:
            if entry["name"] == name:
                entry["status"] = "running"
                entry["index"] = index_1_based
                entry["total"] = total
            elif entry["index"] < index_1_based and entry["status"] == "pending":
                # A phase that never reported (e.g. skipped ordering edge)
                # stays pending rather than silently flipping.
                pass


def _on_phase_log(name: str, log: str, elapsed: float, ok: bool) -> None:
    if len(log) > MAX_PHASE_LOG_CHARS:
        log = ("... [truncated to last "
               f"{MAX_PHASE_LOG_CHARS} chars] ...\n" + log[-MAX_PHASE_LOG_CHARS:])
    with _PHASE_LOCK:
        for entry in _PHASES:
            if entry["name"] == name:
                entry["log"] = log
                entry["elapsed_sec"] = round(elapsed, 3)
                entry["status"] = "done" if ok else "failed"
                break


def _finish_phases() -> None:
    global _PHASES_ACTIVE
    with _PHASE_LOCK:
        _PHASES_ACTIVE = False


app = FastAPI(title="videoSparse SLAM API",
              description="Monocular visual-odometry SLAM over uploaded video.",
              version="1.0.0")

# Frontend origin (local dev). Tighten before any non-local deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/slam/progress")
def slam_progress() -> Dict[str, Any]:
    """Live stage info for the currently running POST /api/slam (if any)."""
    with _PROGRESS_LOCK:
        return dict(_PROGRESS)


@app.get("/api/slam/phases")
def slam_phases() -> Dict[str, Any]:
    """Live per-phase parameters + terminal output for the current run."""
    with _PHASE_LOCK:
        return {"active": _PHASES_ACTIVE,
                "phases": [dict(entry) for entry in _PHASES]}


# ------------------------------------------------------------------ #
# Servable backend-annotated tracking video (the ORB stage's vis video).
#
# The per-request temp dir is deleted after POST /api/slam, so the
# annotated `_orb_vis.mp4` (normalized video with keypoints + counts
# drawn on every frame) is copied into a small LRU disk cache and
# streamed here. Nothing is re-rendered or synthesized.
# ------------------------------------------------------------------ #

TRACKING_VIDEO_MAX_BYTES = 30 * 1024 * 1024
_TRACKING_CACHE_DIR = Path(__file__).resolve().parent / "output" / "tracking_cache"
_TRACKING_CACHE_MAX = 3
_TRACKING_LOCK = threading.Lock()
_TRACKING_VIDEOS: Dict[str, Path] = {}


def _tracking_cache_init() -> None:
    with _TRACKING_LOCK:
        _TRACKING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for stale in _TRACKING_CACHE_DIR.glob("*.mp4"):
            with contextlib.suppress(OSError):
                stale.unlink()
        _TRACKING_VIDEOS.clear()


def _transcode_to_h264(src: Path, dest: Path, fps: float) -> bool:
    """Re-encode an mp4v video as browser-playable H.264 (avc1, Baseline).

    The pipeline writes mp4v (MPEG-4 Part 2), which browsers cannot play in
    <video>. This converts the SERVED COPY ONLY — pipeline artifacts are
    untouched. Validates the output (reopen + decode a frame with matching
    dimensions) and returns False (leaving no file) on any failure. Never
    raises — the caller falls back to no tracking video.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            return False
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            cap.release()
            return False
        writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"avc1"),
                                 float(fps), (w, h))
        if not writer.isOpened():
            cap.release()
            with contextlib.suppress(OSError):
                dest.unlink()
            return False
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
        finally:
            cap.release()
            writer.release()
        # Validate: non-empty, under cap, and actually decodable.
        if (not dest.is_file() or dest.stat().st_size == 0
                or dest.stat().st_size > TRACKING_VIDEO_MAX_BYTES):
            with contextlib.suppress(OSError):
                dest.unlink()
            return False
        check = cv2.VideoCapture(str(dest))
        ok, frame = check.read() if check.isOpened() else (False, None)
        good = (ok and frame is not None
                and int(frame.shape[1]) == w and int(frame.shape[0]) == h)
        check.release()
        if not good:
            with contextlib.suppress(OSError):
                dest.unlink()
            return False
        return True
    except Exception:
        traceback.print_exc()
        with contextlib.suppress(OSError):
            dest.unlink()
        return False


def _publish_tracking_video(vis_path: Path) -> str:
    """Copy a run's annotated ORB video into the servable cache.

    The pipeline's mp4v file is transcoded to H.264 so browsers can play
    it; the served copy is validated before publishing. Returns the API
    path ("" when unavailable/oversize/unplayable). Never raises —
    the tracking video is a bonus; the JSON result must survive without it.
    """
    try:
        if not vis_path.is_file() or vis_path.stat().st_size == 0:
            return ""
        if vis_path.stat().st_size > TRACKING_VIDEO_MAX_BYTES * 4:
            return ""  # absurd input; transcoding would waste time
        job_id = uuid.uuid4().hex
        with _TRACKING_LOCK:
            _TRACKING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            dest = _TRACKING_CACHE_DIR / f"{job_id}.mp4"
            tmp = _TRACKING_CACHE_DIR / f"{job_id}.tmp.mp4"
        # Transcode outside the lock (CPU work); register only on success.
        if not _transcode_to_h264(vis_path, tmp, API_TARGET_FPS):
            return ""
        with _TRACKING_LOCK:
            tmp.rename(dest)
            _TRACKING_VIDEOS[job_id] = dest
            while len(_TRACKING_VIDEOS) > _TRACKING_CACHE_MAX:
                old_id, old_path = next(iter(_TRACKING_VIDEOS.items()))
                del _TRACKING_VIDEOS[old_id]
                with contextlib.suppress(OSError):
                    old_path.unlink()
        return f"/api/slam/tracking-video/{job_id}"
    except Exception:
        traceback.print_exc()
        return ""


@app.get("/api/slam/tracking-video/{job_id}")
def tracking_video(job_id: str):
    """Stream the backend-annotated ORB tracking video for a finished run."""
    with _TRACKING_LOCK:
        path = _TRACKING_VIDEOS.get(job_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404,
                            detail="Tracking video not found or expired.")
    return FileResponse(str(path), media_type="video/mp4",
                        filename="tracking.mp4")


_tracking_cache_init()


def _run_pipeline_quiet(ctx: VideoContext,
                        progress_callback=None,
                        phase_log_callback=None) -> Dict[str, Any]:
    """run_slam with its phase chatter captured instead of printed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return run_slam(ctx, progress_callback=progress_callback,
                        phase_log_callback=phase_log_callback)


def _tracking_sample(orb_path: Path, max_points: int = 300) -> tuple:
    """Real [x, y] keypoints of the middle frame + per-frame counts.

    Reads the `_orb.npz` written by this same run (no re-detection).
    Returns (frame_index, keypoints_xy, counts). Empty inputs yield
    (0, [], []) — never fake points.
    """
    if not orb_path.is_file():
        return 0, [], []
    with np.load(str(orb_path), allow_pickle=False) as z:
        counts = [int(c) for c in z["counts"].tolist()]
        keypoints = np.array(z["keypoints"], dtype=np.float64).reshape(-1, 6)
    if not counts:
        return 0, [], []
    frame_index = len(counts) // 2
    offsets = np.cumsum([0] + counts).tolist()
    start, end = offsets[frame_index], offsets[frame_index + 1]
    xy = keypoints[start:end, :2] if end > start else np.zeros((0, 2))
    if len(xy) > max_points:
        pick = np.linspace(0, len(xy) - 1, max_points).astype(int)
        xy = xy[pick]
    return (frame_index,
            [[float(x), float(y)] for x, y in xy.tolist()],
            counts)


def _tracking_track(orb_path: Path, max_frames: int = 60,
                    max_points: int = 80) -> List[Dict[str, Any]]:
    """Real per-frame keypoint sets sampled across the whole run.

    Reads the `_orb.npz` written by this same run (no re-detection).
    Frames are stride-sampled to at most `max_frames` entries and points
    per frame to at most `max_points` (evenly spaced, strongest-first is
    not stored so index order is kept). Returns
    [{"frame": i, "points": [[x, y], ...]}, ...] — never fake points.
    """
    if not orb_path.is_file():
        return []
    with np.load(str(orb_path), allow_pickle=False) as z:
        counts = [int(c) for c in z["counts"].tolist()]
        keypoints = np.array(z["keypoints"], dtype=np.float64).reshape(-1, 6)
    n = len(counts)
    if n == 0:
        return []
    import math
    stride = max(1, math.ceil(n / max_frames))
    offsets = np.cumsum([0] + counts).tolist()
    track: List[Dict[str, Any]] = []
    for idx in range(0, n, stride):
        start, end = offsets[idx], offsets[idx + 1]
        xy = keypoints[start:end, :2] if end > start else np.zeros((0, 2))
        if len(xy) > max_points:
            pick = np.linspace(0, len(xy) - 1, max_points).astype(int)
            xy = xy[pick]
        track.append({"frame": int(idx),
                      "points": [[float(x), float(y)] for x, y in xy.tolist()]})
    return track


def _build_phase_params(shared: Dict[str, Any], ctx: VideoContext,
                        phase_times: Dict[str, float]) -> Dict[str, Any]:
    """Structured per-phase parameters from THIS run's shared context.

    Mirrors what each phase prints to the terminal (same source dicts),
    repackaged as JSON so the frontend can render every phase's numbers
    without parsing log text. Missing phases yield {} — never fake values.
    """
    def meta_dict(m) -> Dict[str, Any]:
        if m is None:
            return {}
        return {"fps": round(float(m.fps), 3),
                "width": int(m.width), "height": int(m.height),
                "frame_count": int(m.frame_count),
                "duration_sec": round(float(m.duration_sec), 3)}

    orb = shared.get("orb", {}) or {}
    orb_stats = orb.get("stats", {}) or {}
    matches = shared.get("matches", {}) or {}
    match_stats = matches.get("stats", {}) or {}
    motion = shared.get("motion", {}) or {}
    motion_stats = motion.get("stats", {}) or {}
    traj = shared.get("trajectory", {}) or {}
    traj_stats = traj.get("stats", {}) or {}
    pts = shared.get("points3d", {}) or {}
    pts_stats = pts.get("stats", {}) or {}
    filt = shared.get("points3d_filtered", {}) or {}
    filt_stats = filt.get("stats", {}) or {}
    kf = shared.get("keyframes", {}) or {}
    kf_stats = kf.get("stats", {}) or {}
    lmap = shared.get("local_map", {}) or {}
    lmap_stats = lmap.get("stats", {}) or {}
    opt = shared.get("optimization", {}) or {}
    scipy_info = opt.get("scipy", {}) or {}

    return {
        "probe": meta_dict(ctx.source),
        "normalize": {"frames_in": shared.get("frames_read", 0),
                      "frames_out": shared.get("frames_written", 0),
                      **meta_dict(ctx.result)},
        "features": {"frames": orb_stats.get("frames", 0),
                     "min": orb_stats.get("min", 0),
                     "mean": orb_stats.get("mean", 0.0),
                     "max": orb_stats.get("max", 0),
                     "reliable_frames": orb_stats.get("reliable_frames", 0),
                     "unreliable_frames": orb.get("unreliable_frames", []),
                     "tier_use": orb_stats.get("tier_use", {})},
        "matching": {"pairs": match_stats.get("pairs", 0),
                     "min": match_stats.get("min", 0),
                     "mean": match_stats.get("mean", 0.0),
                     "max": match_stats.get("max", 0),
                     "reliable_pairs": match_stats.get("reliable_pairs", 0),
                     "weak_pairs": match_stats.get("weak_pairs", []),
                     "worst_pair": list(match_stats.get("worst_pair", []))},
        "motion": {"pairs": motion_stats.get("pairs", 0),
                   "ok_pairs": motion_stats.get("ok_pairs", 0),
                   "failed_pairs": motion_stats.get("failed_pairs", []),
                   "inliers_min": motion_stats.get("inliers_min", 0),
                   "inliers_mean": motion_stats.get("inliers_mean", 0.0),
                   "inliers_max": motion_stats.get("inliers_max", 0),
                   "focal_length": motion.get("focal_length", 0.0)},
        "trajectory": {"frames": traj_stats.get("frames", 0),
                       "pairs": traj_stats.get("pairs", 0),
                       "valid_pairs": traj_stats.get("valid_pairs", 0),
                       "failed_pairs": traj_stats.get("failed_pairs", []),
                       "segments": traj_stats.get("segments", [])},
        "triangulation": {"pairs_attempted": pts_stats.get("pairs_attempted", 0),
                          "input_matches": pts_stats.get("input_matches", 0),
                          "final_points": pts_stats.get("final_points", 0),
                          "rejected_total": pts_stats.get("rejected_total", 0),
                          "rejected": pts_stats.get("rejected", {})},
        "filter": {"input_points": filt_stats.get("input_points", 0),
                   "final_points": filt_stats.get("final_points", 0),
                   "rejected_total": filt_stats.get("rejected_total", 0),
                   "tiers": filt_stats.get("tiers", [])},
        "keyframes": {"total_keyframes": kf_stats.get("total_keyframes", 0),
                      "total_frames": kf_stats.get("total_frames", 0),
                      "ids": kf.get("ids", []),
                      "gap_min": kf_stats.get("gap_min", 0),
                      "gap_mean": kf_stats.get("gap_mean", 0.0),
                      "gap_max": kf_stats.get("gap_max", 0),
                      "reason_hist": kf_stats.get("reason_hist", {})},
        "local_map": {"n_keyframes": lmap_stats.get("n_keyframes", 0),
                      "n_landmarks": lmap_stats.get("n_landmarks", 0),
                      "n_mapped": lmap_stats.get("n_mapped", 0),
                      "n_orphans": lmap_stats.get("n_orphans", 0),
                      "kf_per_landmark_hist": lmap_stats.get("kf_per_landmark_hist", {})},
        "optimization": {"n_keyframes": opt.get("n_keyframes", 0),
                         "n_optimized": opt.get("n_optimized", 0),
                         "n_obs": opt.get("n_obs", 0),
                         "n_params": opt.get("n_params", 0),
                         "rms_before": opt.get("rms_before", 0.0),
                         "rms_after": opt.get("rms_after", 0.0),
                         "scipy_success": scipy_info.get("success", None),
                         "scipy_nfev": scipy_info.get("nfev", 0),
                         "scipy_status": scipy_info.get("status", "")},
        "timing": {k: round(float(v), 3) for k, v in (phase_times or {}).items()},
    }


def _slam_response(result: Dict[str, Any], elapsed: float,
                   input_meta=None) -> SlamResult:
    from video_pipeline.utils import sibling_output
    shared = result["context"].shared
    ctx = result["context"]
    trajectory = shared.get("trajectory", {}).get("positions", [])
    keyframe_ids = shared.get("keyframes", {}).get("ids", [])
    points_path = Path(shared.get("points3d_filtered", {}).get(
        "filtered_path", ""))
    if not points_path.is_file():
        raise RuntimeError("Pipeline finished without a filtered point cloud.")
    with np.load(str(points_path), allow_pickle=False) as z:
        points = np.array(z["points"], dtype=np.float64)
    frame_count = len(trajectory)
    point_count = len(points)
    avg_fps = round(frame_count / elapsed, 2) if elapsed > 0 else 0.0
    # --- tracking metrics from THIS run's ORB + matching outputs ---
    orb_info = shared.get("orb", {})
    orb_counts: List[int] = [int(c) for c in orb_info.get("counts", [])]
    orb_stats_raw = orb_info.get("stats", {}) or {}
    match_stats_raw = (shared.get("matches", {}) or {}).get("stats", {}) or {}
    if orb_counts:
        mean_kp = round(float(sum(orb_counts) / len(orb_counts)), 1)
        tracked_features = int(round(sum(orb_counts) / len(orb_counts)))
        reliable = orb_info.get("reliable", [])
        ratio = (sum(1 for r in reliable if r) / len(reliable)) if reliable else 0.0
        if ratio >= 0.8:
            tracking_status = "ACTIVE"
        elif ratio >= 0.5:
            tracking_status = "DEGRADED"
        else:
            tracking_status = "LOST"
    else:
        mean_kp = 0.0
        tracked_features = 0
        tracking_status = "UNKNOWN"
    orb_path = sibling_output(ctx.output_path, "_orb.npz")
    tracking_frame_index, tracking_keypoints, sampled_counts = _tracking_sample(orb_path)
    tracking_track = _tracking_track(orb_path)
    if not orb_counts:
        orb_counts = sampled_counts
    orb_stats: Dict[str, Any] = {
        "min": int(orb_stats_raw.get("min", min(orb_counts) if orb_counts else 0)),
        "mean": float(orb_stats_raw.get("mean", mean_kp)),
        "max": int(orb_stats_raw.get("max", max(orb_counts) if orb_counts else 0)),
        "reliable_frames": int(orb_stats_raw.get(
            "reliable_frames", sum(1 for r in orb_info.get("reliable", []) if r))),
        "total_frames": int(orb_stats_raw.get("frames", frame_count)),
    }
    match_stats: Dict[str, Any] = {
        "min": match_stats_raw.get("min", 0),
        "mean": match_stats_raw.get("mean", 0.0),
        "max": match_stats_raw.get("max", 0),
        "pairs": match_stats_raw.get("pairs", max(frame_count - 1, 0)),
    }
    phase_times = {k: round(float(v), 3)
                   for k, v in (result.get("phase_times") or {}).items()}
    phase_params = _build_phase_params(shared, ctx, result.get("phase_times") or {})
    # Every value below is a plain Python type: JSON-safe by construction.
    return SlamResult(
        processing_time_sec=round(elapsed, 3),
        frame_count=frame_count,
        input_frames=int(shared.get("probe_frames") or frame_count),
        video_duration_sec=0.0,  # filled by the caller, which probed the upload
        keyframe_count=len(keyframe_ids),
        point_count=point_count,
        realtime_factor=0.0,  # filled by the caller
        trajectory=[[float(v) for v in row] for row in trajectory],
        points=[[float(v) for v in row] for row in points],
        avg_fps=avg_fps,
        total_frames=frame_count,
        map_points=point_count,
        input_fps=round(float(getattr(input_meta, "fps", 0.0) or 0.0), 3),
        input_width=int(getattr(input_meta, "width", 0) or 0),
        input_height=int(getattr(input_meta, "height", 0) or 0),
        tracking_status=tracking_status,
        tracked_features=tracked_features,
        tracking_frame_index=int(tracking_frame_index),
        tracking_keypoints=tracking_keypoints,
        tracking_track=tracking_track,
        orb_counts=orb_counts,
        orb_stats=orb_stats,
        match_stats=match_stats,
        phase_params=phase_params,
        phase_times=phase_times,
    )


@app.post("/api/slam", response_model=SlamResult)
def run_slam_api(video: UploadFile = File(...)) -> SlamResult:
    """Upload .mp4 -> full SLAM -> JSON (files cleaned up afterwards)."""
    filename = (video.filename or "").lower()
    if not filename.endswith(".mp4"):
        raise HTTPException(status_code=400,
                            detail="Only .mp4 uploads are accepted.")
    request_tmp = tempfile.mkdtemp(prefix="slam_api_")
    try:
        return _handle_upload(video, Path(request_tmp))
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500,
                            detail=f"SLAM processing failed: {exc}")
    finally:
        # The whole request footprint (upload + every stage artifact) dies here.
        shutil.rmtree(request_tmp, ignore_errors=True)


def _handle_upload(video: UploadFile, workdir: Path) -> SlamResult:
    upload_path = workdir / "input.mp4"
    with upload_path.open("wb") as out:
        shutil.copyfileobj(video.file, out)  # streamed to disk, not RAM
    if upload_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    meta = probe_video(upload_path)
    output_path = workdir / f"input_10fps_{API_TARGET_WIDTH}x{API_TARGET_HEIGHT}.mp4"
    ctx = VideoContext(input_path=upload_path, output_path=output_path,
                       target_fps=API_TARGET_FPS, target_width=API_TARGET_WIDTH,
                       target_height=API_TARGET_HEIGHT)
    ctx.shared["probe_frames"] = meta.frame_count

    _set_progress(True, "loading video", 0, len(SLAM_PHASES), 0.0)
    _reset_phases()
    start = time.perf_counter()
    try:
        result = _run_pipeline_quiet(ctx, progress_callback=_report_phase,
                                     phase_log_callback=_on_phase_log)
    except Exception:
        _set_progress(False, "error", 0, len(SLAM_PHASES), 0.0)
        _finish_phases()
        raise
    elapsed = time.perf_counter() - start

    response = _slam_response(result, elapsed, input_meta=meta)
    response.video_duration_sec = round(meta.duration_sec, 3)
    response.input_frames = meta.frame_count
    response.realtime_factor = (round(meta.duration_sec / elapsed, 3)
                                if elapsed > 0 else 0.0)
    # Publish the annotated tracking video BEFORE the temp dir is removed.
    from video_pipeline.utils import sibling_output as _sibling_output
    response.tracking_video_url = _publish_tracking_video(
        _sibling_output(ctx.output_path, "_orb_vis.mp4"))
    _set_progress(False, "done", len(SLAM_PHASES), len(SLAM_PHASES), 100.0)
    _finish_phases()
    print(f"[api] {video.filename}: {meta.frame_count} frames in, "
          f"{response.point_count} points out, {elapsed:.1f}s")
    return response


if __name__ == "__main__":
    raise SystemExit(main())
