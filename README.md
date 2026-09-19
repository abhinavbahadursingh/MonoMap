# videoSparse — Monocular RGB Sparse Point-Cloud SLAM

<div align="center">

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://monomapp.onrender.com/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abhinavbahadursingh/MonoMap)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.2-black?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![Three.js](https://img.shields.io/badge/Three.js-R3F-049EF4?style=for-the-badge&logo=threedotjs&logoColor=white)](https://threejs.org/)
[![Hardware](https://img.shields.io/badge/Compute-CPU--Only-FF6B6B?style=for-the-badge)](#4-prerequisites)

<p align="center">
  <b>A production-grade, end-to-end monocular visual odometry and sparse SLAM platform.</b><br>
  Upload any standard <code>.mp4</code> video to reconstruct 3D sparse point clouds, estimate continuous 6-DoF camera trajectories, track ORB features in 2D with live telemetry, and explore interactive 3D WebGL visualizations directly in your browser.
</p>

</div>

---

> [!WARNING]
> **Monocular Scale Ambiguity:** All reconstructed geometry and camera positions are calculated in **arbitrary relative units, never meters**. Monocular RGB visual odometry cannot observe absolute physical scale without external metric sensors (such as stereo cameras, depth sensors, IMU, or ground-truth fiducials). See [Coordinate Frame, Geometry & Scale](#10-coordinate-frame-geometry--scale).

---

## 🌐 Deployment & Repository

| Target | URL | Notes |
| :--- | :--- | :--- |
| **Deployed Web Application** | **[https://monomapp.onrender.com/](https://monomapp.onrender.com/)** | Full UI with live SLAM execution, 2D tracking & 3D WebGL viewers |
| **Git Source Repository** | **[https://github.com/abhinavbahadursingh/MonoMap](https://github.com/abhinavbahadursingh/MonoMap)** | Complete codebase (FastAPI backend + Next.js frontend + SLAM engine) |
| **Local API Fallback** | `http://127.0.0.1:8000` | Local FastAPI backend endpoint |
| **Local Web Fallback** | `http://localhost:3000` | Local Next.js 14 frontend dashboard |

---

## 📑 Table of Contents

- [1. System Architecture](#1-system-architecture)
- [2. Core Features & Capabilities](#2-core-features--capabilities)
- [3. The 11-Phase SLAM Pipeline](#3-the-11-phase-slam-pipeline)
- [4. Repository Structure](#4-repository-structure)
- [5. Prerequisites](#5-prerequisites)
- [6. Quick Start (Single Command)](#6-quick-start-single-command)
- [7. Web Application Tour](#7-web-application-tour)
- [8. Backend Engine & CLI Reference](#8-backend-engine--cli-reference)
- [9. REST API Specification](#9-rest-api-specification)
- [10. Coordinate Frame, Geometry & Scale](#10-coordinate-frame-geometry--scale)
- [11. Output Artifacts & Data Formats](#11-output-artifacts--data-formats)
- [12. Benchmarking & Performance](#12-benchmarking--performance)
- [13. Production Deployment Guide](#13-production-deployment-guide)
- [14. Engineering Philosophy & Root-Cause Fixes](#14-engineering-philosophy--root-cause-fixes)
- [15. Troubleshooting & FAQ](#15-troubleshooting--faq)
- [16. Known Limitations & Roadmap](#16-known-limitations--roadmap)
- [17. Attribution & AI Usage](#17-attribution--ai-usage)

---

## 1. System Architecture

videoSparse is structured as a **decoupled, modular monocular SLAM system**. It pairs a high-performance, CPU-only OpenCV/NumPy/SciPy geometry engine with an asynchronous FastAPI application layer and a responsive Next.js 14 React Three Fiber frontend.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                BROWSER FRONTEND (Next.js 14)                            │
│                                                                                        │
│  ┌───────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────────┐ │
│  │   Dashboard Metrics   │  │   Video Upload & Run  │  │   2D Feature Tracking HUD   │ │
│  │  (Animated Count-Up)  │  │  (Live Sync Run Strip) │  │  (H.264 Stream + Sparkline) │ │
│  └───────────────────────┘  └───────────────────────┘  └─────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────┐  ┌─────────────────────────────┐ │
│  │       3D WebGL Viewers (React Three Fiber)       │  │   Live Phase Parameters &   │ │
│  │  Height-Ramped Cloud  │  Camera Trajectory Path  │  │    Real-time Terminal Logs  │ │
│  └──────────────────────────────────────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │ HTTP / JSON Polling & Video Stream
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI BACKEND (Python 3.11+)                            │
│                                                                                        │
│  • POST /api/slam (Multipart upload, isolated ephemeral execution dir)                 │
│  • POST /api/slam/skip-optimization (Cooperative cancel of Phase 11, preserves trajectory) │
│  • GET  /api/slam/progress & /phases (700ms polling for non-blocking UI updates)       │
│  • GET  /api/slam/tracking-video/{id} (Browser-playable H.264 transcode cache)          │
│  • GET  /api/sample-video & /info (Bundled video1.mp4 one-click demo testing)          │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │ Shared VideoContext State
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             CANONICAL 11-PHASE SLAM PIPELINE                           │
│                                                                                        │
│  [1. Probe] ──► [2. Normalize] ──► [3. ORB Features] ──► [4. Matcher] ──► [5. Motion] │
│                                                                                 │      │
│  [10. Local Map] ◄── [9. Keyframes] ◄── [8. Filter] ◄── [7. Triangulate] ◄─────┘      │
│         │                                                                              │
│         ▼                                                                              │
│  [11. Huber Pose BA*] ─► Final 3D Point Cloud + Camera Trajectory + Telemetry JSON    │
│         (*skippable via UI/CLI/API — uses trajectory from Phase 6 when skipped)       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Architectural Tenets
1. **Single Engine, Dual Interfaces:** The SLAM algorithm in `backend/src/video_pipeline/` is the single source of truth. Both the CLI (`python main.py`) and Web API (`main:app`) invoke the same underlying pipeline without duplicated logic.
2. **Deterministic Metric Integrity:** Zero synthetic values. Every frame count, point coordinate, feature yield, and timing benchmark returned to the client is computed directly by the SLAM pipeline.
3. **Responsive Polling via Threadpool Isolation:** Heavy CPU-bound SLAM runs execute in FastAPI's background threadpool, allowing synchronous polling endpoints (`/progress`, `/phases`) to deliver real-time terminal output and progress percentages with zero UI blocking.
4. **Transient Storage & Resource Hygiene:** Each API job operates inside an isolated temporary directory that is automatically purged upon completion, retaining only the cached browser-compatible tracking preview video.

---

## 2. Core Features & Capabilities

### 🔬 SLAM & Computer Vision Engine
- **100% CPU-Only Execution:** Operates cleanly on standard multi-core CPUs without requiring CUDA, specialized GPUs, or complex deep learning runtimes.
- **Adaptive 3-Tier ORB Reliability Ladder:** Features are extracted using a progressive fallback ladder:
  - *Tier 1 (Nominal):* Standard fast ORB detection.
  - *Tier 2 (Low Contrast):* Automatic CLAHE (Contrast Limited Adaptive Histogram Equalization).
  - *Tier 3 (Challenging Footage):* High-sensitivity detector with lower response thresholds.
  - *Integrity Guarantee:* Hard-to-track frames are transparently flagged as degraded/unreliable; synthetic keypoints are never injected.
- **Robust 5-Point Motion Estimation:** Epipolar geometry estimation using OpenCV's 5-point Essential Matrix algorithm coupled with RANSAC outlier rejection and Cheirality depth verification.
- **Segmented Trajectory Chaining:** Recovers continuous 6-DoF camera poses ($R, \mathbf{t}$). Tracking disruptions gracefully generate isolated trajectory segments with local reference frames, preventing unrealistic interpolation.
- **Multi-Stage Point Cloud Filtering:**
  - Strict reprojection error thresholds ($\le 2.0$ px).
  - Positive depth and baseline parallax gating.
  - Statistical Outlier Removal (SOR) filtering based on mean $k$-nearest-neighbor distance distribution.
- **Keyframe Selection & Local Mapping:** Extracts keyframes based on translation distance, angular rotation change, feature overlap ratios, and forced periodic intervals.
- **Huber-Robust Pose-Only Bundle Adjustment:** Pose-graph refinement utilizing SciPy's `least_squares` with Huber loss weighting to dampen the effect of residual feature drift.
- **One-Click Skip Optimization (Phase 11):** The expensive Huber bundle adjustment can be skipped entirely. When skipped, the pipeline preserves the trajectory from Phase 6 (Trajectory Chaining) as the final trajectory, never fakes optimized poses, and continues directly to finalization. Cooperative cancellation via `skip_control` checks at phase entry and on every `least_squares` residual/Jacobian evaluation (next safe checkpoint), never killing the whole SLAM job. Enabled via CLI (`--skip optimization`), API (`POST /api/slam/skip-optimization`), or UI (Skip buttons).

### 🖥️ Next.js Web Dashboard
- **Real-Time SLAM Dashboard:** Live animated count-up metric cards displaying total processing time, average throughput (FPS), total reconstructed 3D map points, keyframe count, and processed frames.
- **Accurate Synchronized Telemetry:** Execution timers measure backend processing only (starting at the exact millisecond the upload finishes and the first pipeline phase begins).
- **Interactive 2D Feature Tracking Panel:** Plays back the backend-annotated tracking video (ORB keypoints with per-frame counts) in a smooth loop, paired with an interactive keypoint sparkline, live playhead, and real-time status HUD.
- **Browser-Safe H.264 Playback:** Automatically transcodes OpenCV's raw `mp4v` codec to standards-compliant H.264 Baseline, ensuring seamless cross-browser playback without black screens.
- **Dynamic 3D WebGL Explorers (React Three Fiber):**
  - **3D Sparse Point Cloud Viewer:** Real-time orbit, zoom, and pan navigation with height-ramped color gradients (blue $\to$ cyan $\to$ yellow).
  - **Camera Trajectory Viewer:** Interactive 3D path visualization complete with green origin/start indicators, red termination markers, and frame orientation indices.
- **Live Per-Phase Diagnostic Stream:** Accordion panel detailing all 11 phases with live status pills (`pending`, `running`, `done`, `failed`, `skipped`), per-phase wall clock execution times, and complete terminal output logs streamed live via polling. Phase 11 shows “Pose optimization skipped / Using trajectory from previous phase” when skipped and a final `Pose Optimization: Skipped | Completed` indicator.
- **Skip Optimization Controls (Render-aware):** Two single-click `Skip Optimization` buttons — a compact `Skip 11th step` next to `Run SLAM` in the Input Video card (auto-starts the job if idle) and a `Skip Optimization` button inside the Phase 11 card — both hit `POST /api/slam/skip-optimization` for instant cancellation on weak Render CPU/GPU. Helpful tip rendered directly below `Run SLAM`.
- **One-Click Sample Verification:** Built-in default sample video button instantly tests the system using `video1.mp4` without requiring manual file uploads.

---

## 3. The 11-Phase SLAM Pipeline

Every uploaded or CLI-supplied video executes through the canonical 11-phase sequence defined in `backend/pipeline.py`:

```
   Video File (.mp4)
         │
         ▼
 ┌───────────────┐
 │   1. PROBE    │   Extract duration, native FPS, container frame count, resolution
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │ 2. NORMALIZE  │   Rational frame decimation & bicubic resize to 10 FPS @ 640×360
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │ 3. FEATURES   │   ORB detection & 32-byte BRIEF descriptors (nfeatures=500, 3-tier ladder)
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │  4. MATCHING  │   Consecutive-frame BFMatcher (Hamming distance) + Lowe's ratio test (0.70)
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │   5. MOTION   │   Essential Matrix (5-point) + RANSAC + Cheirality test -> Relative R, unit-t
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │ 6. TRAJECTORY │   Pose chaining into global world frame; automatic segment splitting on failure
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │7. TRIANGULATE │   Two-view linear DLT triangulation on verified feature matches
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │  8. FILTER    │   Reprojection gating (<= 2px) + relative depth sanity + SOR filtering
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │ 9. KEYFRAMES  │   Translation, rotation, feature overlap decay & temporal interval selection
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │ 10. LOCAL MAP │   Keyframe-to-landmark spatial association matrix & visibility graph
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │11. OPTIMIZATION│  Pose-only Bundle Adjustment via SciPy least_squares with Huber robust loss
 └───────┬───────┘
          │
          ▼
    Final Output: 3D Point Cloud (.npz) + Trajectory (.npz) + JSON Result + H.264 Video
         (Phase 11 skipped → trajectory from Phase 6 is returned as final)
```

### Skipping Pose Optimization (Phase 11) — Render Performance Control

Pose optimization is the most expensive phase (SciPy `least_squares` over all keyframe–landmark reprojection residuals) and can take **many minutes on Render's weak CPU/GPU**. The pipeline supports a safe, cooperative skip that never fakes results:

* **What it does:** Stops/never starts the Huber bundle adjustment, keeps the **existing trajectory produced by Phase 6 (Trajectory Chaining)** as the final trajectory, and continues directly `Phase 11 → Finalization → Results`. No trajectory recalculation, no synthetic optimized poses.
* **Cancellation safety:** If Phase 11 is already running, the backend checks `skip_control.should_skip()` at phase entry **and on every residual/Jacobian evaluation** inside `least_squares`. At the next safe checkpoint it raises `SkipOptimization`, preserves the last valid trajectory, and proceeds to finalization — never killing the entire SLAM job.
* **How to trigger:**
  * **Web UI (preferred, one-click):** `Input Video → Skip 11th step` mini button below `Run SLAM` (enabled even when idle — it auto-starts SLAM then skips) **or** `Phase Parameters → Phase 11 → Skip Optimization` button (enabled for `pending`/`running` and also `idle`). Both call `POST /api/slam/skip-optimization` instantly.
  * **REST API:** `POST /api/slam/skip-optimization` → `{ok:true}`; polling `GET /api/slam/phases` then shows `"status":"skipped"` with logs `Pose optimization skipped / Using trajectory from previous phase`.
  * **CLI:** `python main.py --skip optimization` (or any subset of the 11 phase names; see [CLI Execution](#8-backend-engine--cli-reference)).
* **Result signalling:** `POST /api/slam` response includes `optimization_skipped` (`bool`) + `optimization_status` (`"skipped"|"completed"`) and `phase_params.optimization = {skipped, status, ...}`. The UI renders `Pose Optimization: Skipped|Completed` and the Phase 11 card shows the two-line skipped notice.
* **Nothing else changes:** ORB tracking, pose estimation, triangulation, point cloud, keyframe selection, local mapping and Three.js visualization are untouched.

### Phase Details & Mathematical Foundations

| # | Phase | Module | Mathematical Operation / Method | Output Produced |
| :-: | :--- | :--- | :--- | :--- |
| **1** | `probe` | `stages.py` | OpenCV `VideoCapture` container metadata probe | FPS, resolution, total frames, duration |
| **2** | `normalize` | `stages.py` | Exact rational downsampling ($t_k = k \cdot \Delta t$) + bicubic interpolation | Standardized $640\times360$ video @ 10 FPS |
| **3** | `features` | `orb_features.py` | FAST corner detection + oriented intensity centroid + 256-bit BRIEF descriptor | Keypoint coordinates, descriptors, tier stats |
| **4** | `matching` | `orb_matching.py` | Brute-force Hamming distance with Lowe's ratio test ($d_1 / d_2 < 0.70$) | Verified 2D match pairs between adjacent frames |
| **5** | `motion` | `camera_motion.py` | 5-point Essential Matrix $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$ with RANSAC & SVD recovery | Relative rotation $\mathbf{R}_{k,k-1}$, unit translation $\hat{\mathbf{t}}_{k,k-1}$ |
| **6** | `trajectory` | `trajectory.py` | Forward chaining: $\mathbf{T}_{k} = \mathbf{T}_{k-1} \begin{bmatrix} \mathbf{R} & \hat{\mathbf{t}} \\ 0 & 1 \end{bmatrix}^{-1}$; segment reset on tracking loss | 6-DoF camera poses and trajectory segments |
| **7** | `triangulation` | `triangulation.py` | Direct Linear Transform (DLT) on normalized camera rays $\mathbf{x} \times (\mathbf{P}\mathbf{X}) = \mathbf{0}$ | Raw 3D landmark point coordinates |
| **8** | `filter` | `point_filter.py` | Reprojection error $\| \mathbf{x} - \pi(\mathbf{P}\mathbf{X}) \|_2 \le 2\text{px}$ + Statistical Outlier Removal (SOR) | Filtered, clean 3D sparse point cloud |
| **9** | `keyframes` | `keyframes.py` | Multi-criteria gating: $\|\mathbf{t}\| > \theta_t \lor \Delta\theta > \theta_R \lor \text{overlap} < \theta_o$ | Sparse keyframe indices and selection causes |
| **10** | `local_map` | `local_map.py` | Cross-frame landmark indexing and keyframe co-visibility graph generation | Keyframe-to-landmark association graph |
| **11** | `optimization`| `optimization.py` | Huber-weighted non-linear least squares pose BA: $\min_{\mathbf{T}_k} \sum \rho_H(\| \mathbf{r}_{ij} \|^2)$ — **skippable** (see below) | Refined camera poses and convergence metrics — or `skipped` preserves Phase-6 trajectory |

---

## 4. Repository Structure

```text
videoSparse/
├── README.md                      # Comprehensive project documentation
├── instruction.txt                # Working agreement: root-cause fixes only
├── .gitignore                     # Git ignore rules (builds, cache, video files)
├── package.json                   # Root orchestrator: starts backend & frontend simultaneously
│
├── backend/                       # Python SLAM engine & FastAPI web service
│   ├── main.py                    # Dual-mode entrypoint: CLI command & FastAPI application
│   ├── pipeline.py                # SLAM pipeline orchestrator (11 canonical phases)
│   ├── benchmark.py               # Performance timing harness & baseline comparison
│   ├── visualize_3d.py            # Standalone Open3D point cloud & camera path exporter
│   ├── requirements.txt           # Python dependency manifest
│   ├── video1.mp4                 # Bundled reference test video
│   ├── src/video_pipeline/        # Core SLAM algorithmic implementations
│   │   ├── __init__.py
│   │   ├── context.py             # VideoContext & VideoMetadata state containers
│   │   ├── stages.py              # Base Stage class, VideoProbe, FrameSampling
│   │   ├── pipeline.py            # Chainable VideoPipeline helper
│   │   ├── orb_features.py        # 3-tier ORB feature extraction
│   │   ├── orb_matching.py        # Descriptor matching & Lowe ratio filtering
│   │   ├── camera_motion.py       # Essential matrix & RANSAC motion estimation
│   │   ├── trajectory.py          # Relative pose chaining & segment partitioning
│   │   ├── triangulation.py       # DLT triangulation & ray angle verification
│   │   ├── point_filter.py        # Reprojection gating & SOR filtering
│   │   ├── keyframes.py           # Adaptive keyframe selection logic
│   │   ├── local_map.py           # Co-visibility graph & landmark indexing
│   │   ├── optimization.py        # Huber-loss pose-only bundle adjustment (skippable via skip_control)
│   │   ├── skip_control.py        # Thread-safe skip flag for Phase 11 (requested by /skip-optimization)
│   │   └── utils.py               # Geometry math, debug plotters & H.264 transcode
│   └── output/                    # Generated CLI artifacts, plots, and cached videos
│       └── tracking_cache/        # H.264 transcoded 2D tracking videos (LRU cache)
│
└── frontend/                      # Next.js 14 Web Application
    ├── app/
    │   ├── layout.tsx             # Root HTML layout, dark theme, and metadata
    │   ├── page.tsx               # Main dashboard, state machine, and 3D viewers
    │   └── globals.css            # Dark glassmorphic SLAM styling system
    ├── components/
│   ├── DashboardMetrics.tsx   # Top animated count-up KPI cards & status pill
│   ├── VideoUpload.tsx        # File picker, video preview, live progress strip, Skip-11th-step mini button + tip
│   ├── FeatureTrackingPanel.tsx# Annotated 2D tracking player, HUD & sparkline
│   ├── PhaseParameters.tsx    # Live 11-phase cards (fully expanded), timing, streamed logs, Skip Optimization button
    │   ├── PointCloudViewer.tsx   # 3D height-colored point cloud (React Three Fiber)
    │   └── TrajectoryViewer.tsx   # 3D 6-DoF camera trajectory path (React Three Fiber)
    ├── lib/
    │   └── api.ts                 # Typed API client, polling engine & phase metadata
    ├── package.json               # Frontend dependencies & Next.js scripts
    ├── tsconfig.json              # TypeScript strict configuration
    └── .env.example               # Example backend API URL override
```

---

## 5. Prerequisites

Before running the project locally, verify that your machine meets the following runtime requirements:

| Component | Minimum Version | Check Command | Purpose |
| :--- | :--- | :--- | :--- |
| **Python** | `3.11+` | `python --version` | Backend API & SLAM geometry engine |
| **Node.js** | `18.0+` | `node --version` | Frontend Next.js application |
| **npm** | `9.0+` | `npm --version` | Package management & dev server orchestration |
| **WebGL** | Enabled Browser | `chrome://gpu` | Required for 3D point cloud & trajectory exploration |
| **Network Ports** | `8000` & `3000` | — | `8000` (FastAPI backend), `3000` (Next.js frontend) |

> [!NOTE]
> **No GPU or CUDA Required:** All computer vision and mathematical optimization algorithms are optimized for standard multi-core CPUs.

---

## 6. Quick Start (Single Command)

You can launch both the FastAPI backend and Next.js frontend with a single command from the project root.

### 1. One-Time Dependency Installation

Clone the repository and install both backend and frontend dependencies:

```bash
# Clone the repository
git clone https://github.com/abhinavbahadursingh/MonoMap.git
cd MonoMap

# Install Python backend packages
pip install -r backend/requirements.txt

# Install root runner and frontend packages
npm install
npm run install:frontend
```

### 2. Start Both Servers

Run the root dev command:

```bash
npm run dev
```

This starts:
- 🟢 **FastAPI Backend:** Running on `http://127.0.0.1:8000`
- 🔵 **Next.js Frontend:** Running on `http://localhost:3000`

Verify that the backend is responding:
```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

<details>
<summary><b>Prefer running in separate terminal windows?</b> (Click to expand)</summary>

**Terminal 1 — Backend:**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm install
npm run dev
```
</details>

### 3. Run Your First SLAM Reconstruction

1. Open your browser and navigate to **`http://localhost:3000`**.
2. Click **"Use default testing video (video1.mp4)"** to instantly load the bundled sample file, or upload any `.mp4` video.
3. Click the cyan **"Run SLAM"** button.
4. Watch the live progress strip:
   - Upload progress bar ($0 \to 100\%$).
   - Synchronized execution timer kicks off when Phase 1 starts.
   - Live phase parameters and real-time terminal logs stream in.
5. Explore the interactive results:
   - Live 2D tracking video with keypoint HUD and sparkline.
   - Interactive 3D sparse point cloud colored by height.
   - 3D camera trajectory path showing poses and keyframes.

---

## 7. Web Application Tour

The web application is tailored for high-density visual odometry inspection:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [● IDLE/DONE]  SLAM DASHBOARD                                                          │
│  Process Time: 8.5 s │ Avg FPS: 10.6 │ Map Points: 12,467 │ Keyframes: 41 │ Frames: 90 │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│ VIDEO INPUT & METRICS                    │ 2D FEATURE TRACKING                         │
│ • Drag & Drop / Default Video Button     │ • Autoplaying loop of annotated ORB video   │
│ • File metadata (FPS, Res, Duration)     │ • Real-time HUD: Frame #, Feature Count     │
│ • "Run SLAM" + Live execution strip      │ • Keypoints-per-frame sparkline & playhead  │
│ • Tip + "Skip 11th step" mini button     │                                             │
├──────────────────────────────────────────┴─────────────────────────────────────────────┤
│ TRACKING & MAP SUMMARY                                                                 │
│ Mean Features: 427 │ ORB Range: 53-579 │ Reliable: 100% │ Matches: 113 │ Realtime: ×1.05│
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│ 3D SPARSE POINT CLOUD (R3F)              │ CAMERA TRAJECTORY (R3F)                     │
│ • Orbit / Zoom / Pan Controls            │ • 3D 6-DoF trajectory curve in world space  │
│ • Height-ramped colormap (Blue/Cyan/Gold)│ • Green start marker & Red end marker       │
├──────────────────────────────────────────┴─────────────────────────────────────────────┤
│ 11-PHASE PARAMETERS & LIVE TERMINAL OUTPUT (fully expanded, no scroll)                 │
│ • All 11 phases always visible (no scroll) with live pills (idle/running/done/skipped)│
│ • Phase 11 card: "Skip Optimization" button (also before start; idle click starts job)│
│ • Skipped banner: "Pose optimization skipped / Using trajectory from previous phase"    │
│ • Real terminal stdout stream captured directly from the Python backend                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key UI Capabilities
- **Skip-11th-Step Affordance:** Directly below `Run SLAM` a tip explains `Tip: you can skip the 11th step (Optimization)… Render's weak CPU/GPU` with a compact `Skip 11th step` button. When `IDLE`, clicking it auto-starts SLAM and immediately requests the skip (`POST /api/slam/skip-optimization`), yielding a faster result. The same control exists inside the Phase 11 card — single-click, no confirmation required.
- **Synchronized Backend Timing:** The timer on the upload panel does not prematurely start on upload click; it triggers at the exact moment the backend acknowledges receipt and initializes Phase 1 (`probe`).
- **Resilient 2D Tracking Player:** Seamlessly displays the backend-generated H.264 preview video. If video playback is unavailable, it automatically switches to a high-speed HTML5 2D canvas overlay using sampled ORB coordinates.
- **Smart Viewport Positioning:** When SLAM completes, a floating badge *"View 3D Results"* appears, automatically disappearing when the user scrolls the 3D canvases into view via an `IntersectionObserver`.
- **Fully Expanded Phase List:** Phase Parameters no longer caps height or scrolls — all 11 phases are always visible; only terminal `<pre>` tails inside `<details>` scroll.

---

## 8. Backend Engine & CLI Reference

In addition to serving the web application, the backend functions as a standalone command-line tool.

### CLI Execution
Run SLAM directly from your terminal:

```bash
cd backend

# Option A: Execute on default video specified in main.py
python main.py

# Option B: Pass explicit input and output arguments
python main.py --input "path/to/video.mp4" --output "output/custom_run.mp4"

# Option C: Customize normalization parameters
python main.py --input test.mp4 --fps 15 --width 1280 --height 720
```

### Skipping Specific Phases
For targeted debugging or research experiments, any phase can be skipped without modifying source code:

```bash
# Skip optimization to inspect raw triangulated trajectory (also skippable live via UI/API)
python main.py --skip optimization

# Skip heavy geometric stages
python main.py --skip local_map optimization
```

*Valid `--skip` identifiers:* `probe`, `normalize`, `features`, `matching`, `motion`, `trajectory`, `triangulation`, `filter`, `keyframes`, `local_map`, `optimization`.

> **Web/API equivalent:** `POST /api/slam/skip-optimization` — cooperative skip of Phase 11 while the job is running (or immediately before `probe` when `IDLE` via the `Skip 11th step` button below `Run SLAM`). The pipeline preserves the trajectory from previous phases as final, never faking BA results, and shows `Pose Optimization: Skipped` in the final JSON and UI. See [Skipping Pose Optimization (Phase 11)](#skipping-pose-optimization-phase-11--render-performance-control).

### Standalone 3D Visualization
To render and inspect point clouds in native Open3D windows:

```bash
python visualize_3d.py
```

---

## 9. REST API Specification

The FastAPI server provides typed endpoints for video processing, live execution polling, and preview streaming.

### Endpoints Overview

| Method | Endpoint | Description | Query / Body |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Liveness health check | None $\to$ `{"status": "ok"}` |
| `GET` | `/` | Service root probe & docs link | None $\to$ Status & metadata |
| `POST` | `/api/slam` | Execute full 11-phase SLAM run | Multipart Form: `video` (`.mp4` binary) |
| `POST` | `/api/slam/skip-optimization` | **Skip Phase 11 (pose optimization) cooperatively** — preserves trajectory from Phase 6, continues to finalization | None $\to$ `{"ok":true}`; polled `GET /api/slam/phases` shows `status:"skipped"` with `Pose optimization skipped` log |
| `GET` | `/api/slam/progress` | Poll overall progress status | None $\to$ Current stage & percentage |
| `GET` | `/api/slam/phases` | Poll live per-phase states & logs | None $\to$ All 11 phases with stdout tails (`status` includes `skipped`) |
| `GET` | `/api/slam/tracking-video/{job_id}` | Stream browser-compatible H.264 video | Path: `job_id` $\to$ `video/mp4` stream |
| `GET` | `/api/sample-video/info` | Query bundled sample video status | None $\to$ `{available, filename, size_bytes}` |
| `GET` | `/api/sample-video` | Download bundled sample video | None $\to$ `video/mp4` binary stream |

### Response Schema (`POST /api/slam`)

```jsonc
{
  // Core Visual Odometry Telemetry
  "processing_time_sec": 8.52,          // Total wall-clock execution time
  "frame_count": 90,                    // Processed normalized frames
  "input_frames": 224,                  // Original raw input video frames
  "video_duration_sec": 8.96,           // Raw input video duration
  "keyframe_count": 41,                 // Number of selected keyframes
  "point_count": 12467,                 // Reconstructed 3D sparse landmark count
  "realtime_factor": 1.05,              // Ratio: video_duration / processing_time
  "avg_fps": 10.56,                     // Processed frames per second

  // 3D Geometry
  "trajectory": [                       // 6-DoF camera centers [x, y, z] per frame
    [0.0, 0.0, 0.0],
    [0.012, -0.003, 0.045]
  ],
  "points": [                           // 3D landmark coordinates [x, y, z]
    [0.42, 1.15, 3.84],
    [-0.81, 0.32, 2.19]
  ],

  // 2D Tracking Telemetry
  "tracking_status": "ACTIVE",          // ACTIVE | DEGRADED | LOST
  "tracked_features": 427,              // Mean ORB features detected per frame
  "orb_stats": {
    "min": 53,
    "mean": 427.3,
    "max": 579,
    "reliable_frames": 90,
    "total_frames": 90
  },
  "match_stats": {
    "min": 24,
    "mean": 113.2,
    "max": 284,
    "pairs": 89
  },

  // Pose Optimization result (never faked)
  "optimization_skipped": false,        // true when Phase 11 was skipped via UI/CLI/API
  "optimization_status": "completed",   // "completed" | "skipped"

  // Per-Phase Execution Breakdown
  "phase_times": {
    "probe": 0.005,
    "normalize": 1.12,
    "features": 2.45,
    "matching": 0.88,
    "motion": 0.62,
    "trajectory": 0.04,
    "triangulation": 1.34,
    "filter": 0.45,
    "keyframes": 0.12,
    "local_map": 0.21,
    "optimization": 1.28             // ~0.02 when skipped (cancellation at checkpoint)
  },
  "phase_params": {
    "optimization": {
      "skipped": false,              // true when skipped — UI shows amber "Skipped" banner
      "status": "completed",         // "completed" | "skipped"
      "n_optimized": 38,
      "n_obs": 4210
    }
  },
  "tracking_video_url": "/api/slam/tracking-video/550e8400-e29b-41d4-a716-446655440000"
}
```

---

## 10. Coordinate Frame, Geometry & Scale

Understanding the mathematical constraints of monocular visual odometry is critical for interpreting the reconstructed models:

### 1. The Monocular Scale Ambiguity
When capturing the world with a single pinhole camera, depth and scale are fundamentally ambiguous:
$$\lambda \mathbf{x} = \mathbf{K} [\mathbf{R} \mid \mathbf{t}] \mathbf{X}$$
Scaling the world coordinates $\mathbf{X}$ by an arbitrary scalar $\alpha$ while scaling translation $\mathbf{t}$ by $\alpha$ produces identical pixel projections $\mathbf{x}$. Because of this:
- All translation vectors $\hat{\mathbf{t}}$ are estimated as **unit-length directions** ($\|\hat{\mathbf{t}}\|_2 = 1.0$).
- Output coordinates represent **relative spatial structure**, not physical meters.

### 2. Coordinate Conventions
- **World Frame:** Defined by Frame 0 with camera center at $\mathbf{C}_0 = [0, 0, 0]^T$ and rotation $\mathbf{R}_0 = \mathbf{I}_{3\times3}$.
- **Camera Poses:** Poses represent the transformation from world frame to camera coordinates: $\mathbf{x}_c = \mathbf{R}_{cw} \mathbf{X}_w + \mathbf{t}_{cw}$. Camera centers plotted in 3D represent $\mathbf{C}_w = -\mathbf{R}_{cw}^T \mathbf{t}_{cw}$.
- **Axis System:** Standard computer vision coordinates: $+X$ points right, $+Y$ points down, and $+Z$ points forward along the optical axis.

### 3. Segmented Trajectories on Tracking Loss
When rapid motion or low texture causes epipolar RANSAC inliers to drop below safety thresholds, the pipeline does **not** hallucinate a trajectory bridge. Instead:
- It terminates the current segment.
- It initiates a new segment with a fresh local origin upon tracking recovery.
- Trajectory breaks are visualized as disconnected paths rather than distorted jumps.

---

## 11. Output Artifacts & Data Formats

When running via CLI or persistent server modes, all intermediate and final outputs are saved to `backend/output/`:

| File Name Pattern | Type | Contents & Use Case |
| :--- | :--- | :--- |
| `<name>_10fps_640x360.mp4` | Video | Normalized, resampled base video file |
| `<name>_orb.npz` | NumPy Archive | Keypoint coordinates $(N \times 2)$, 32-byte descriptors, detection tiers |
| `<name>_orb_vis.mp4` | Video | Rendered video with detected ORB features and stats overlaid |
| `<name>_orb_matches.npz` | NumPy Archive | Adjacent frame match indices and Lowe distance ratios |
| `<name>_motion.npz` | NumPy Archive | Relative rotation matrices $\mathbf{R}$, unit translations $\hat{\mathbf{t}}$, RANSAC inlier masks |
| `<name>_trajectory.npz` | NumPy Archive | Chained world camera centers, rotation matrices, segment IDs |
| `<name>_trajectory_xz.png` | Image | Bird's-eye view (Top-Down $X$-$Z$) 2D camera path plot |
| `<name>_points3d.npz` | NumPy Archive | Raw triangulated 3D landmark points and ray intersection angles |
| `<name>_points3d_filtered.npz` | NumPy Archive | Final SOR-filtered 3D sparse point cloud (consumed by frontend) |
| `<name>_points3d_filtered_triview.png`| Image | Orthographic 3-view projection plot (XY, XZ, YZ planes) |
| `<name>_keyframes.npz` | NumPy Archive | Selected keyframe frame indices and triggering criteria |
| `<name>_local_map.npz` | NumPy Archive | Keyframe-to-landmark visibility association graph |
| `<name>_slam_optimized.npz` | NumPy Archive | Final Huber-refined camera poses and converged residual errors — **not written when Phase 11 is skipped** (final trajectory is the Phase-6 chain) |
| `tracking_cache/<job_id>.mp4` | H.264 Video | Validated browser-playable MP4 tracking video stream |

---

## 12. Benchmarking & Performance

The repository includes a dedicated benchmarking tool (`backend/benchmark.py`) to measure throughput, compare performance across commits, and establish timing baselines.

### Running Benchmarks
```bash
cd backend

# Execute benchmark on the default test video
python benchmark.py

# Save current run as a baseline snapshot
python benchmark.py --save-as output/baseline_v1.json

# Compare current execution against an existing baseline
python benchmark.py --compare output/baseline_v1.json
```

### Performance Metrics Reference
Throughput is evaluated using the **Realtime Factor** ($\text{RTF}$):
$$\text{RTF} = \frac{\text{Video Duration (seconds)}}{\text{Processing Time (seconds)}}$$
- $\text{RTF} \ge 1.0$: Pipeline operates **faster than real-time**.
- $\text{RTF} < 1.0$: Pipeline operates slower than real-time.

```text
================================================================================
                           BENCHMARK EXECUTION SUMMARY                          
================================================================================
 Input Video:         video1.mp4 (Normalized: 90 frames @ 10 FPS, 640x360)
 Duration:            8.96 seconds
 Processing Time:     8.52 seconds
 Average Throughput:  10.56 FPS
 Realtime Factor:     x1.05  [REALTIME VERDICT: PASSED]
 Map Points Yield:    12,467 points
 Keyframes Generated: 41 keyframes
================================================================================
```

---

## 13. Production Deployment Guide

The application is engineered for zero-friction deployment to modern cloud hosting platforms (such as Render, Vercel, or Railway) without requiring Docker or root system privileges.

### 1. Backend Deployment (Render / VPS)
- **Environment:** Python 3.11+
- **Root Directory:** `backend`
- **Build Command:**
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command:**
  ```bash
  python -m uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
  ```
- **Environment Variables:**
  - `FRONTEND_URL`: `https://monomapp.onrender.com` (or comma-separated allowed CORS origins)
  - `SAMPLE_VIDEO_PATH`: `video1.mp4` (optional custom path to bundled test video)

> [!IMPORTANT]
> **Single Worker Recommendation:** Keep `--workers 1` unless you attach an external distributed store (Redis / S3) for the in-memory progress polling state and the tracking video cache. **Required for skip:** `POST /api/slam/skip-optimization` flips the in-process `skip_control` flag; with multiple workers the flag would land on the wrong worker.

> [!TIP]
> **Render / weak CPU/GPU:** Pose optimization (Phase 11) can dominate wall time. Users can skip it from the UI (`Skip 11th step` below `Run SLAM` or `Phase 11 → Skip Optimization` — single-click, works even before the run and auto-starts it) or via `POST /api/slam/skip-optimization` mid-run. The backend cancels at the next `least_squares` checkpoint, preserves the Phase-6 trajectory, and returns `optimization_status:"skipped"`.

### 2. Frontend Deployment (Vercel / Render Static)
- **Framework:** Next.js 14
- **Root Directory:** `frontend`
- **Build Command:**
  ```bash
  npm run build
  ```
- **Start Command:**
  ```bash
  npm run start -- --port $PORT
  ```
- **Environment Variables:**
  - `NEXT_PUBLIC_API_URL`: Base URL of the deployed FastAPI backend (e.g., `https://monomap-backend.onrender.com`). Set this before executing `npm run build`.

---

## 14. Engineering Philosophy & Root-Cause Fixes

In strict accordance with the repository's foundational guidelines (`instruction.txt`), all updates and fixes prioritize **identifying root causes over applying superficial patches**:

### Case Studies in Root-Cause Problem Solving

1. **The Browser Black Screen Bug (`mp4v` vs `H.264`)**
   - *Symptom:* The 2D feature tracking video rendered as a black screen in Google Chrome and Safari.
   - *Temporary Patch (Rejected):* Hide the video player and only show static images or canvas dots.
   - *Root Cause:* OpenCV's default `cv2.VideoWriter_fourcc(*'mp4v')` outputs MPEG-4 Part 2, which modern web browsers cannot decode due to lack of hardware decoding support.
   - *Permanent Fix:* Implemented an automatic server-side transcode pass using `cv2.CAP_FFMPEG` / H.264 baseline encoder with parameter verification, ensuring 100% browser-compliant streaming.

2. **CORS Rejection on Deployed Origins**
   - *Symptom:* The deployed frontend was blocked from communicating with the backend API.
   - *Temporary Patch (Rejected):* Setting `allow_origins=["*"]` with insecure credentials.
   - *Root Cause:* Dynamic cloud host domains require pre-flight `OPTIONS` handling and origin matching across environment variables.
   - *Permanent Fix:* Built dynamic CORS resolver `_cors_origins()` supporting `FRONTEND_URL` and `CORS_ORIGINS` environment variables while pre-whitelisting local dev and the production URL (`https://monomapp.onrender.com`).

3. **Inaccurate Execution Timing**
   - *Symptom:* Reported processing time included client upload latency, skewing performance metrics on slow network connections.
   - *Temporary Patch (Rejected):* Guessing upload overhead or deducting a fixed constant.
   - *Root Cause:* The UI timer started on file upload button click rather than backend execution start.
   - *Permanent Fix:* Re-engineered timer lifecycle in `frontend/app/page.tsx` and `VideoUpload.tsx`: the timer starts strictly when the backend confirms receipt and initiates Phase 1 (`probe`).

4. **Render Timeout on Pose Optimization (Phase 11)**
   - *Symptom:* Huber BA runs many minutes on Render's weak CPU, causing slow final results or apparent hangs.
   - *Temporary Patch (Rejected):* Hide the Phase 11 card or fake optimized poses after timeout.
   - *Root Cause:* Phase 11 (`scipy.optimize.least_squares` with analytic sparse Jacobian) is CPU-bound by design and cannot finish fast on free-tier hosts.
   - *Permanent Fix:* Introduced cooperative `skip_control` flag + `POST /api/slam/skip-optimization`. The optimization stage checks at entry and on every residual/Jacobian call; `SkipOptimization` aborts `least_squares` at the next checkpoint, preserves the trajectory from Phase 6 as final, writes no fake `_slam_optimized.npz`, and returns `optimization_status:"skipped"` with `Pose optimization skipped / Using trajectory from previous phase` in the phase log and Phase 11 UI banner.

---

## 15. Troubleshooting & FAQ

| Symptom | Probable Cause | Corrective Action |
| :--- | :--- | :--- |
| **`API unreachable` in web UI** | Backend service is not active on port 8000 | Run `curl http://127.0.0.1:8000/health`. If down, start backend via `npm run dev:backend`. |
| **`Port 8000 already in use`** | Another background uvicorn or python instance holds the port | Run `netstat -ano \| findstr :8000` (Windows) and kill the PID, or pass `--port 8001` (update `NEXT_PUBLIC_API_URL` accordingly). |
| **Optimization slow on Render / timeout** | Phase 11 Huber BA is CPU-bound | Use `Skip 11th step` below `Run SLAM` (works even when `IDLE` — auto-starts then skips) or `Phase 11 → Skip Optimization`, or `POST /api/slam/skip-optimization`, or `python main.py --skip optimization`. Skipped jobs return the Phase-6 trajectory as final with `optimization_status:"skipped"`. |
| **`Cannot find module './<n>.js'` in `.next`** | Stale Next.js webpack build cache | Delete `frontend/.next` directory and restart dev server (`npm run dev:frontend`). |
| **Skip button does nothing** | Wrong worker or no active job | Ensure backend runs with `--workers 1` and the job is `running`/`pending`. After skip, `GET /api/slam/phases` shows `optimization: {status:"skipped"}` and the result JSON has `optimization_skipped:true`. |
| **`Only .mp4 uploads are accepted`** | Uploaded file has an invalid extension or MIME type | Ensure the uploaded video is encapsulated in an `.mp4` container. |
| **`Cannot find module './<n>.js'` in `.next`** | Stale Next.js webpack build cache | Delete `frontend/.next` directory and restart dev server (`npm run dev:frontend`). |
| **3D view is blank or crashes** | Hardware acceleration disabled or WebGL blocked | Enable hardware acceleration in browser settings and verify via `chrome://gpu`. |
| **Tracking player falls back to overlay** | Tracking video cache expired or file exceeds 30 MB cap | Normal behavior: the high-speed 2D canvas overlay continues to display real keypoint telemetry. |
| **Cold start latency on first run** | Initial module loading and Python JIT warm-up | Subsequent runs process immediately; cold starts take ~2-3 seconds for OpenCV initialization. |

---

## 16. Known Limitations & Roadmap

### Known Limitations
- **Monocular Scale:** Scale cannot be observed without external metric sensors; all distances are relative unit vectors.
- **Pure Rotational Motion:** Severe camera rotation without translational baseline causes DLT triangulation degeneracy.
- **Low Texture / Overexposure:** Extremely dark or featureless sequences can trigger tracking loss and segment splits.
- **Dynamic Entities:** Moving pedestrians or vehicles are treated as static scene features and can introduce slight motion bias.
- **Skipped Optimization:** When Phase 11 is skipped, final poses are unrefined (Phase-6 chain) — accuracy may drop slightly but runtime is cut drastically on weak hosts (by design).

### Planned Roadmap
- [ ] **Loop Closure Engine:** Visual bag-of-words (DBoW2) loop detection placed between `local_map` and `optimization`.
- [ ] **Global Pose-Graph BA:** Full pose-graph optimization across closed loops to eliminate long-term drift.
- [ ] **Dense Depth Elevation:** Optional semi-dense / dense depth estimation pass for surface mesh generation.
- [ ] **Persistent Job Queue:** Redis + Celery worker queue replacing in-memory progress polling for high-concurrency multi-tenant clusters.
- [ ] **Export Formats:** Direct export buttons for standard 3D formats (`.ply`, `.obj`, `.las`) and TUM trajectory formats.

---

## 17. Attribution & AI Usage

### Software & Open-Source Libraries
- **Computer Vision & SLAM:** [OpenCV](https://opencv.org/) (`opencv-python`), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Open3D](http://www.open3d.org/)
- **Backend Web Framework:** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [Pydantic](https://pydantic.dev/)
- **Frontend & 3D Engine:** [Next.js](https://nextjs.org/), [React](https://react.dev/), [Three.js](https://threejs.org/), [React Three Fiber](https://r3f.docs.pmnd.rs/), [@react-three/drei](https://github.com/pmndrs/drei)

### AI Usage Disclosure
- **AI Tooling Utilized:** OpenCode and Antigravity IDE coding agents.
- **Scope of Application:**
  - Codebase exploration, architectural visualization, and modular refactoring assistance.
  - Formulating root-cause fixes (such as H.264 transcoding pipelines and CORS origin handling).
  - Authoring and structuring comprehensive technical documentation grounded strictly in real code implementations.
- **Human Oversight & Verification:** All geometric derivations, SLAM pipeline sequences, and performance validations were reviewed, tested, and verified by the repository maintainer. No fabricated metrics or synthetic data have been introduced.

---

<div align="center">
  <sub>Engineered with precision for robust monocular visual odometry. Built by <a href="https://github.com/abhinavbahadursingh">Abhinav Bahadur Singh</a>.</sub>
</div>
