/** Typed client for the SLAM backend. Field names mirror backend/main.py. */

export interface SlamResult {
  processing_time_sec: number;
  frame_count: number;
  input_frames: number;
  video_duration_sec: number;
  keyframe_count: number;
  point_count: number;
  realtime_factor: number;
  /** Per-frame camera centers [x, y, z], frame 0 at origin. */
  trajectory: number[][];
  /** Filtered landmark positions [x, y, z]. */
  points: number[][];
  // --- extended real metrics from the same pipeline run (backend/main.py) ---
  avg_fps: number;
  total_frames: number;
  map_points: number;
  input_fps: number;
  input_width: number;
  input_height: number;
  tracking_status: string;
  tracked_features: number;
  tracking_frame_index: number;
  /** Sampled [x, y] ORB keypoints in normalized 640x360 px. */
  tracking_keypoints: number[][];
  /** Per-frame keypoint sets over time: [{frame, points}] (640x360 px). */
  tracking_track: { frame: number; points: number[][] }[];
  /** ORB keypoints per processed frame (real, for sparklines). */
  orb_counts: number[];
  orb_stats: {
    min: number;
    mean: number;
    max: number;
    reliable_frames: number;
    total_frames: number;
  };
  match_stats: { min: number; mean: number; max: number; pairs: number };
  /** Structured per-phase parameters from the backend run. */
  phase_params: Record<string, Record<string, unknown>>;
  /** Wall time per pipeline phase, in seconds. */
  phase_times: Record<string, number>;
  /** Backend-annotated ORB tracking video path ("" when unavailable). */
  tracking_video_url: string;
}

export type Phase = "idle" | "uploading" | "processing" | "done" | "error";

export interface SlamProgress {
  active: boolean;
  stage: string;
  stage_index: number;
  stage_total: number;
  percent: number;
}

export interface PhaseInfo {
  name: string;
  label: string;
  index: number;
  total: number;
  status: "pending" | "running" | "done" | "failed";
  log: string;
  elapsed_sec: number;
}

/** Canonical pipeline phases in execution order. */
export const PHASE_ORDER = [
  "probe",
  "normalize",
  "features",
  "matching",
  "motion",
  "trajectory",
  "triangulation",
  "filter",
  "keyframes",
  "local_map",
  "optimization",
];

export const PHASE_LABELS: Record<string, string> = {
  probe: "Probe input video",
  normalize: "Normalize frames",
  features: "ORB feature extraction",
  matching: "Descriptor matching",
  motion: "Camera motion estimation",
  trajectory: "Trajectory chaining",
  triangulation: "Triangulation (sparse map)",
  filter: "Point cloud filtering",
  keyframes: "Keyframe selection",
  local_map: "Local mapping",
  optimization: "Pose optimization",
};

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

/** Human-readable labels for the 11 canonical pipeline phases. */
export const STAGE_LABELS: Record<string, string> = {
  probe: "Loading video...",
  normalize: "Normalizing frames...",
  features: "Extracting ORB features...",
  matching: "Tracking features...",
  motion: "Estimating camera pose...",
  trajectory: "Building trajectory...",
  triangulation: "Triangulating 3D points...",
  filter: "Filtering map points...",
  keyframes: "Selecting keyframes...",
  local_map: "Building sparse map...",
  optimization: "Finalizing trajectory...",
};

/** Ordered fallback stage list while the progress endpoint is unreachable. */
export const STAGE_ORDER = [
  "probe",
  "normalize",
  "features",
  "matching",
  "motion",
  "trajectory",
  "triangulation",
  "filter",
  "keyframes",
  "local_map",
  "optimization",
];

export function stageLabel(stage: string): string {
  if (!stage || stage === "idle") return "Idle";
  if (stage === "done") return "Finalizing trajectory...";
  if (stage === "loading video") return "Loading video...";
  return STAGE_LABELS[stage] ?? stage;
}

export async function fetchProgress(): Promise<SlamProgress | null> {
  try {
    const res = await fetch(`${API_BASE}/api/slam/progress`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as SlamProgress;
  } catch {
    return null;
  }
}

export async function fetchPhases(): Promise<PhaseInfo[] | null> {
  try {
    const res = await fetch(`${API_BASE}/api/slam/phases`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    const body = await res.json();
    return (body?.phases ?? null) as PhaseInfo[] | null;
  } catch {
    return null;
  }
}

/**
 * POST the video as multipart form data. XHR (not fetch) so the upload
 * phase can report real progress for large .mp4 files.
 */
export function uploadVideo(
  file: File,
  onProgress: (percent: number) => void,
): Promise<SlamResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/api/slam`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(body as SlamResult);
        } else {
          reject(new Error(body?.detail ?? `Server error (${xhr.status})`));
        }
      } catch {
        reject(new Error(`Bad server response (${xhr.status})`));
      }
    };
    xhr.onerror = () =>
      reject(new Error(`API unreachable at ${API_BASE} — is the backend running?`));
    const form = new FormData();
    form.append("video", file, file.name);
    xhr.send(form);
  });
}
