# videoSparse — Monocular RGB Sparse Point-Cloud SLAM

Upload an `.mp4` video and get a full monocular visual-odometry SLAM run:
probe → normalize (10 FPS, 640×360) → ORB features → matching → camera
motion → trajectory → triangulation → filtering → keyframes → local map →
pose optimization. Inspect the result in the browser — live metrics
dashboard, 2D feature-tracking video, 3D sparse point cloud, camera
trajectory, and per-phase parameters — or grab the `.npz` artifacts for
your own code.

> All geometry is in **arbitrary units, never meters**. Monocular scale is
> unobservable with a single camera (see
> [Coordinate frame & scale](#11-coordinate-frame--scale)).

---

## Table of contents

1. [Features](#1-features)
2. [How it works](#2-how-it-works)
3. [Repository structure](#3-repository-structure)
4. [Prerequisites](#4-prerequisites)
5. [Quick start (web app)](#5-quick-start-web-app)
6. [Backend in detail](#6-backend-in-detail)
7. [Frontend in detail](#7-frontend-in-detail)
8. [Configuration reference](#8-configuration-reference)
9. [Utility scripts](#9-utility-scripts)
10. [Outputs & artifacts](#10-outputs--artifacts)
11. [Coordinate frame & scale](#11-coordinate-frame--scale)
12. [Troubleshooting](#12-troubleshooting)
13. [Development notes](#13-development-notes)

---

## 1. Features

**SLAM engine (Python / OpenCV / NumPy / SciPy, CPU-only)**

- 11-phase monocular visual-odometry pipeline with per-phase timing,
  per-phase terminal logs, and failure attribution (the failing phase is
  named on error).
- ORB feature extraction with a 3-tier reliability ladder
  (plain → CLAHE → sensitive detector); unreliable frames are *reported*,
  never padded with fake keypoints.
- Descriptor matching (BFMatcher + Hamming + Lowe ratio test), Essential
  matrix motion estimation with RANSAC, segmented trajectory chaining,
  pair-local triangulation with reprojection gating, SOR-filtered sparse
  map, motion/overlap/interval keyframe selection, keyframe–landmark local
  map, and Huber-robust pose-only bundle adjustment (SciPy).
- Every stage persists `.npz` artifacts plus OpenCV-rendered plots and
  preview videos (no display/GPU needed).

**Web app (Next.js 14 + React Three Fiber + FastAPI)**

- **SLAM Dashboard** — compact metric cards fed by the real backend
  response: process time, avg FPS, map points, keyframes, frames processed,
  plus an IDLE / PROCESSING / DONE status pill.
- **Input Video** — upload + preview with real file metadata (name, size,
  duration, resolution, FPS, frames) and a prominent Run SLAM button with
  a live run strip (upload %, backend stage, elapsed time).
- **2D Feature Tracking** — autoplaying loop of the backend-annotated
  tracking video (ORB keypoints + counts drawn per frame by the pipeline
  itself), with a live frame/points/playing HUD and a keypoints-per-frame
  sparkline with playhead.
- **Tracking & Map Summary** — mean features/frame, ORB range, reliable
  frames, mean matches/pair, realtime factor, duration.
- **Phase Parameters** — all 11 phases with status, elapsed time,
  structured parameters, and collapsible real terminal output, streamed
  live while processing via polling.
- **3D Sparse Point Cloud** — orbitable height-colored point cloud
  (React Three Fiber), with point/keyframe/frame metadata.
- **Camera Trajectory** — orbitable camera-center path with start/end
  markers, with pose/keyframe/FPS metadata.
- A floating **"View 3D Results"** button appears on completion and hides
  itself once the 3D views are scrolled into view.

---

## 2. How it works

```
.mp4 upload
  │  probe            read FPS / resolution / frame count / duration (OpenCV)
  ▼
  │  normalize        resample to 10 FPS @ 640×360 (exact rational sampling)
  ▼
  │  features         ORB keypoints + descriptors per frame (nfeatures=500)
  ▼
  │  matching         consecutive-frame BFMatcher + Lowe ratio (0.70)
  ▼
  │  motion           Essential matrix + RANSAC → (R, unit-t) per pair
  ▼
  │  trajectory       chain relative poses; gaps become new segments
  ▼
  │  triangulation    triangulate good matches → sparse 3D cloud
  ▼
  │  filter           strict reprojection + relative-depth + SOR filtering
  ▼
  │  keyframes        translation / rotation / overlap / interval selection
  ▼
  │  local_map        keyframe ↔ landmark association map
  ▼
  │  optimization     Huber-robust pose-only bundle adjustment
  ▼
JSON result + annotated tracking video → dashboard + 3D viewers
```

The 11 canonical phases (stable identifiers used in timing reports,
`--skip` flags, and the `/api/slam/phases` endpoint):

| # | Phase | What it computes |
|---|-------|------------------|
| 1 | `probe` | Input FPS, resolution, frame count, duration |
| 2 | `normalize` | Frames in/out, output FPS + resolution |
| 3 | `features` | Keypoint min/mean/max, reliable frames, tier usage |
| 4 | `matching` | Good matches min/mean/max, reliable pairs, weak/worst pair |
| 5 | `motion` | Valid/failed pairs, RANSAC inlier stats, focal length |
| 6 | `trajectory` | Frames, valid pairs, per-segment frame ranges |
| 7 | `triangulation` | Pairs attempted, input matches, final points, rejection causes |
| 8 | `filter` | Input/final points, rejected total, per-tier waterfall |
| 9 | `keyframes` | Keyframe ids, gaps, selection-reason histogram |
| 10 | `local_map` | Keyframes, landmarks, mapped/orphan counts |
| 11 | `optimization` | Observations, parameters, reprojection RMS before/after, SciPy stats |

---

## 3. Repository structure

```
videoSparse/
├── README.md                 # this file
├── instruction.txt           # project working agreement (root-cause fixes)
├── .gitignore                # repo-wide ignores (Python, Node, media, artifacts)
│
├── backend/                  # SLAM engine + FastAPI + scripts
│   ├── main.py               # CLI entry AND FastAPI app (one file, two doors)
│   ├── pipeline.py           # orchestration: canonical 11-phase SLAM flow
│   ├── benchmark.py          # timing + realtime verdict + baseline compare
│   ├── visualize_3d.py       # standalone Open3D export/render
│   ├── requirements.txt
│   ├── src/video_pipeline/   # pipeline stages
│   │   ├── __init__.py
│   │   ├── context.py        # VideoContext + VideoMetadata (shared state)
│   │   ├── stages.py         # Stage base class, probe + frame sampling
│   │   ├── pipeline.py       # chainable VideoPipeline helper
│   │   ├── orb_features.py   # ORB extraction + reliability ladder
│   │   ├── orb_matching.py   # BFMatcher + Lowe ratio matching
│   │   ├── camera_motion.py  # Essential-matrix motion estimation
│   │   ├── trajectory.py     # pose chaining + segments
│   │   ├── triangulation.py  # sparse 3D reconstruction
│   │   ├── point_filter.py   # reprojection/depth/SOR filtering
│   │   ├── keyframes.py      # keyframe selection
│   │   ├── local_map.py      # keyframe–landmark association map
│   │   ├── optimization.py   # pose-only bundle adjustment
│   │   └── utils.py          # metadata formatting, charts, tri-view renderer
│   └── output/               # run artifacts (git-ignored, regenerable)
│       └── tracking_cache/   # last 3 servable annotated tracking videos
│
└── frontend/                 # Next.js 14 + React Three Fiber UI
    ├── app/
    │   ├── layout.tsx        # root layout + metadata
    │   ├── page.tsx          # dashboard page + run orchestration + results
    │   └── globals.css       # dark SLAM theme + all component styles
    ├── components/
    │   ├── DashboardMetrics.tsx     # top metric cards + status pill
    │   ├── VideoUpload.tsx          # upload, metadata, Run SLAM, live run strip
    │   ├── FeatureTrackingPanel.tsx # annotated tracking video + HUD + overlay
    │   ├── PhaseParameters.tsx      # live per-phase params + terminal logs
    │   ├── PointCloudViewer.tsx     # 3D sparse point cloud (R3F)
    │   └── TrajectoryViewer.tsx     # camera trajectory (R3F)
    ├── lib/
    │   └── api.ts            # typed API client, stage/phase labels, polling
    ├── package.json
    ├── tsconfig.json
    ├── next.config.mjs
    └── .env.example          # sample backend-URL override
```

---

## 4. Prerequisites

- **Python 3.11+** (`python --version`)
- **Node.js 18+** (`node --version`), npm bundled with it
- No ffmpeg / CUDA / GPU needed — CPU-only, OpenCV-based
- A browser with WebGL enabled (required by the 3D viewers)
- Ports **8000** (backend) and **3000** (frontend) free — see
  [Troubleshooting](#12-troubleshooting) if occupied

---

## 5. Quick start (web app)

Open **two separate terminal windows**.

### Terminal 1 — backend (FastAPI on port 8000)

```powershell
cd backend

# Optional: create and activate a virtual environment first
# python -m venv venv
# .\venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

> Use `main:app` (the FastAPI app object), **not** `main:main`
> (which is the CLI function).

Verify it is live:

```powershell
curl.exe http://127.0.0.1:8000/health
# {"status":"ok"}
```

### Terminal 2 — frontend (Next.js on port 3000)

```powershell
cd frontend
npm install
npm run dev
```

### Open the application

Navigate to **http://localhost:3000**, upload an `.mp4` video, and click
**Run SLAM**. While it runs you will see live stage progress, per-phase
parameters streaming in, and on completion the dashboard metrics, the
annotated 2D tracking video, the 3D point cloud, and the camera
trajectory.

---

## 6. Backend in detail

### 6.1 CLI usage (no UI)

```powershell
cd backend
pip install -r requirements.txt
```

Pick the input video **one** of two ways (terminal flags win):

```powershell
# Option A: edit INPUT_VIDEO at the top of main.py, then just:
python main.py

# Option B: pass it in the terminal:
python main.py --input "C:\videos\myvideo.mp4"
python main.py --input in.mp4 --output out.mp4 --fps 10 --width 640 --height 360
```

From the repo root the same works as `python backend\main.py`.
Converted videos and every stage artifact land in `backend/output/`.

Skip any phase without deleting it:

```powershell
python main.py --skip optimization
# valid names: probe, normalize, features, matching, motion, trajectory,
# triangulation, filter, keyframes, local_map, optimization
```

To attach future work, add a stage to `SLAM_PHASES` in
`backend/pipeline.py` — `main.py` only *calls* the engine and never
contains stage logic. (A loop-closure stage plugs in between `local_map`
and `optimization`; see the note at the top of `pipeline.py`.)

### 6.2 API reference

Base URL defaults to `http://127.0.0.1:8000`.

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness probe → `{"status":"ok"}` |
| `POST /api/slam` | Multipart upload (`video` field, one `.mp4`). Runs the full pipeline in an isolated temp dir (deleted afterwards) and returns the `SlamResult` JSON below |
| `GET /api/slam/progress` | Live stage info for the running job: `{active, stage, stage_index, stage_total, percent}` (real completed-phase counts, not estimates) |
| `GET /api/slam/phases` | Live per-phase state: `{active, phases: [{name, label, index, total, status, log, elapsed_sec}]}`. `log` is that phase's real terminal output (tail-truncated at 8000 chars) |
| `GET /api/slam/tracking-video/{job_id}` | Streams the run's annotated ORB tracking video (`video/mp4`, H.264 Baseline — browser-playable). The pipeline writes MPEG-4 Part 2, which browsers cannot play, so the served copy is transcoded and validated; the last 3 runs are cached |

Smoke tests:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/api/slam/progress
curl.exe http://127.0.0.1:8000/api/slam/phases
curl.exe -F "video=@C:\videos\myvideo.mp4;type=video/mp4" http://127.0.0.1:8000/api/slam
```

### 6.3 `POST /api/slam` response schema

Core fields (original contract) plus extended fields — **all derived from
the same single pipeline run**, no second execution, no synthetic values:

```jsonc
{
  // ---- core ----
  "processing_time_sec": 8.5,   // full pipeline wall time
  "frame_count": 90,            // processed (normalized) frames
  "input_frames": 224,          // frames in the uploaded video
  "video_duration_sec": 8.96,
  "keyframe_count": 41,
  "point_count": 12467,         // filtered 3D landmarks
  "realtime_factor": 1.05,      // video_duration / processing_time
  "trajectory": [[0, 0, 0], "..."],  // per-frame camera centers, frame 0 at origin
  "points": [[x, y, z], "..."],      // filtered landmark positions
  // ---- dashboard ----
  "avg_fps": 10.6,              // frame_count / processing_time
  "total_frames": 90,           // alias of frame_count
  "map_points": 12467,          // alias of point_count
  "input_fps": 25.0, "input_width": 3840, "input_height": 2160,
  // ---- 2D tracking (real ORB data from this run) ----
  "tracking_status": "ACTIVE",  // ACTIVE / DEGRADED / LOST from ORB reliability
  "tracked_features": 427,      // mean ORB keypoints per processed frame
  "tracking_frame_index": 257,
  "tracking_keypoints": [[x, y], "..."],  // sample frame, normalized 640×360 px
  "tracking_track": [{"frame": 0, "points": [[x, y], "..."]}, "..."],
  "orb_counts": [511, "..."],   // keypoints per processed frame
  "orb_stats": {"min": 53, "mean": 427.3, "max": 579,
                "reliable_frames": 514, "total_frames": 514},
  "match_stats": {"min": 0, "mean": 113.2, "max": 498, "pairs": 513},
  // ---- per-phase detail ----
  "phase_params": {"probe": {...}, "normalize": {...}, "features": {...},
                   "matching": {...}, "motion": {...}, "trajectory": {...},
                   "triangulation": {...}, "filter": {...}, "keyframes": {...},
                   "local_map": {...}, "optimization": {...}, "timing": {...}},
  "phase_times": {"probe": 0.006, "...": "..."},
  "tracking_video_url": "/api/slam/tracking-video/<job_id>"  // "" when unavailable
}
```

### 6.4 Request lifecycle (`POST /api/slam`)

1. Reject non-`.mp4` filenames (400) and empty files (400).
2. Stream the upload to an isolated temp dir; probe it with OpenCV.
3. Run the 11 phases with stdout captured; per-phase progress + logs are
   published to the in-memory polling state (safe: FastAPI runs the sync
   endpoint in a threadpool, so the poll endpoints keep answering).
4. Publish the transcoded annotated tracking video into the LRU cache.
5. Delete the entire temp dir and return the JSON above.

---

## 7. Frontend in detail

### 7.1 Screens & components (`frontend/app/page.tsx` + `components/`)

| Area | Component | What it shows |
|---|---|---|
| Top metrics | `DashboardMetrics.tsx` | Process time, avg FPS, map points, keyframes, frames processed + IDLE/PROCESSING/DONE pill (animated count-up) |
| Aside row (3 columns) | `VideoUpload.tsx` | File picker, compact preview, name/size/duration/resolution/FPS/frames, Run SLAM, live run strip (upload %, backend stage, elapsed), inline errors |
| Aside row | *Tracking & Map Summary* (in `page.tsx`) | Tracking status pill + mean features, ORB range, reliable frames, mean matches, realtime factor, duration |
| Aside row | `FeatureTrackingPanel.tsx` | Autoplaying annotated tracking video (loop), live FRAME/count/PLAYING HUD, keypoints sparkline with playhead, canvas keypoint overlay fallback when no annotated video |
| Full width | `PhaseParameters.tsx` | 11 phases: status, elapsed, structured params, collapsible terminal logs; polls `/api/slam/phases` every 700 ms while busy, auto-scrolls to the running phase |
| Results | `page.tsx` + `PointCloudViewer.tsx` | **3D Sparse Point Cloud**: height-ramped points (blue→cyan→yellow), orbit/zoom/pan, point/keyframe/frame metadata |
| Results | `page.tsx` + `TrajectoryViewer.tsx` | **Camera Trajectory**: green path, green start / red end markers, pose/keyframe/FPS metadata |
| Floating | `page.tsx` | **View 3D Results** button on completion; hides once the 3D views are on screen (IntersectionObserver) |

Page column is 90% of the viewport (centered, 1400 px cap); the aside
row collapses to one column below 1100 px, the 3D views below 1024 px.

### 7.2 API client (`frontend/lib/api.ts`)

- `SlamResult` / `SlamProgress` / `PhaseInfo` types mirroring the backend.
- `API_BASE` (default `http://127.0.0.1:8000`), `uploadVideo()` over XHR
  for real upload progress, `fetchProgress()` / `fetchPhases()` polling,
  `STAGE_LABELS` / `PHASE_ORDER` / `PHASE_LABELS` display maps.

### 7.3 Configuration & production build

The UI calls the backend directly. To override the backend URL, create
`frontend/.env.local` (see `.env.example`):

```powershell
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

```powershell
cd frontend
npm run dev              # development server (port 3000)
npm run build            # production build (also type-checks)
npm run start -- --port 3000
```

> If you ever see `Cannot find module './<n>.js'` from
> `.next/server/webpack-runtime.js`, it is a stale Next.js dev cache —
> stop the dev server, delete `frontend/.next`, and restart (this bites
> especially after deleting/renaming component files while dev is running).

---

## 8. Configuration reference

| Setting | Where | Default | Notes |
|---|---|---|---|
| Backend URL for UI | `frontend/.env.local` → `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | Rebuild/restart dev after changing |
| CLI input video | `backend/main.py` → `INPUT_VIDEO`, or `--input` | `backend/video1.mp4` | Flags override the file variables |
| Target FPS / resolution | `main.py` (`TARGET_FPS/WIDTH/HEIGHT`), or `--fps/--width/--height` | 10 FPS, 640×360 | API always uses these values |
| ORB budget | `backend/pipeline.py` `SLAM_PHASES` (`nfeatures=500`) | 500 | Deployment speed/quality tradeoff; revert to 1000 if maps thin out |
| Skip phases | `--skip <names...>` | none | Names listed in §2 |
| Tracking cache size | `main.py` `_TRACKING_CACHE_MAX` | 3 runs | LRU-evicted from `backend/output/tracking_cache/` |
| Served video cap | `main.py` `TRACKING_VIDEO_MAX_BYTES` | 30 MB | Oversize runs omit `tracking_video_url` |
| Phase log cap | `main.py` `MAX_PHASE_LOG_CHARS` | 8000 chars | Tail-kept per phase for `/api/slam/phases` |
| Tracking track sample | `main.py` `_tracking_track()` | ≤60 frames × 80 pts | Bounds the JSON payload |

---

## 9. Utility scripts

All run from `backend/`:

| Command | What it does |
|---|---|
| `python -m uvicorn main:app --reload --port 8000` | Start the API server for the UI |
| `python benchmark.py` | Time the full flow, print realtime verdict (`total <= video duration`) |
| `python benchmark.py --save-as output/bench_base.json` | Snapshot a timing baseline |
| `python benchmark.py --compare output/bench_base.json` | BEFORE-vs-AFTER table after any change |
| `python visualize_3d.py` | Export Open3D cloud/cameras + captures to `output/` (needs a display for PNGs) |

---

## 10. Outputs & artifacts

CLI runs write next to the normalized video in `backend/output/`
(API runs use a temp dir that is deleted; only the tracking video is
kept — see §6.2):

| Artifact | Content |
|---|---|
| `<stem>_10fps_640x360.mp4` | Normalized video |
| `<stem>_orb.npz` | Per-frame keypoints, descriptors, counts, tiers |
| `<stem>_orb_vis.mp4` | Normalized video with keypoints + counts overlaid (mp4v codec) |
| `<stem>_orb_preview.jpg`, `_orb_timeline.png`, `_orb_desc.png` | First-vs-weakest frame, counts timeline, descriptor mosaic |
| `<stem>_orb_matches.npz` + `_orb_match_*.png/.mp4` | Good matches per pair + montage/timeline/video |
| `<stem>_motion.npz` + `_motion_timeline.png` | Per-pair (R, unit-t), inliers, statuses |
| `<stem>_trajectory.npz` + `_trajectory_xz.png` | Camera centers, rotations, segment ids |
| `<stem>_points3d.npz` + `_points3d_triview.png` | Raw triangulated cloud |
| `<stem>_points3d_filtered.npz` + `..._filtered_triview.png` | SOR-filtered cloud (this is what the UI renders) |
| `<stem>_keyframes.npz` + `_keyframes_plot.png` | Keyframe ids, reasons, gaps |
| `<stem>_local_map.npz` + `_local_map_plot.png` | Keyframe–landmark associations |
| `<stem>_slam_optimized.npz`, `_slam_traj_plot.png`, `_slam_map_plot.png` | Optimized poses + final map |
| `tracking_cache/<job>.mp4` | Last 3 H.264 tracking videos served to the UI |

---

## 11. Coordinate frame & scale

- Frame 0 defines the world frame (position `[0,0,0]`, identity rotation).
- Trajectory `rotations` map world → camera frame; `positions` are camera
  centers in the world frame.
- Translations are **direction-only unit vectors**: consecutive `t`s must
  never be summed into meters, and nothing may be reported, plotted, or
  consumed as metric distance. Trajectory shape is indicative only.
- Failed pairs break the chain into **segments** with local origins;
  segments are never joined by lines.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `unrecognized arguments: main:main --reload` | Ran `uvicorn main:main` instead of `uvicorn main:app` | `app` is the FastAPI instance; `main` is the CLI function |
| `[WinError 10013]` port 8000 in use | Another backend process holds the port | Check `curl.exe http://127.0.0.1:8000/health`; kill it or use `--port 8001` (+ matching `NEXT_PUBLIC_API_URL`) |
| `API unreachable … is the backend running?` in the UI | Backend not running on port 8000 | Start it per §5 |
| `Only .mp4 uploads are accepted` (400) | Non-mp4 file | Rename/convert to `.mp4` |
| `No input video set` from `main.py` | CLI ran without input | Set `INPUT_VIDEO` or pass `--input <path>` |
| `Cannot find module './<n>.js'` in `.next` | Stale Next.js dev cache | Stop dev, `Remove-Item -Recurse -Force frontend\.next`, restart dev |
| Next.js page blank / WebGL error | Hardware acceleration off | Enable WebGL; 3D viewers require it |
| Tracking panel falls back to overlay (no badge) | Annotated video oversize/unavailable, or backend predates the endpoint | Check backend logs; overlay mode uses the same real keypoint data |
| Slow first API call | Cold start | Normal — imports + full SLAM run on first request |

---

## 13. Development notes

- **Root-cause-first:** per `instruction.txt`, fix root causes through the
  codebase — no temporary patches. (Example: the black tracking video was
  traced to the `mp4v` codec browsers can't decode, fixed by serving a
  validated H.264 transcode — not by hiding the player.)
- **Verify with evidence:** `npm run build` for the frontend;
  `python -m py_compile` plus a real `POST /api/slam` run for the
  backend. Key numbers (avg FPS, counts, codec tags) have been
  cross-checked against actual responses in this repo's history.
- **No fake data:** every dashboard/tracking/phase value must come from
  the backend run; unavailable metrics are exposed through the API, never
  invented on the client.
- **Single engine:** the SLAM algorithm lives in `src/video_pipeline/`
  and is only *called* by `main.py` — no duplicated processing logic.
#   M o n o M a p  
 