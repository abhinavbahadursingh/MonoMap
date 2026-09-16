"""Triangulation stage: sparse 3D map from good matches + relative poses.

For every motion-valid consecutive pair, reconstructs the matched points
with `cv2.triangulatePoints` and aggregates them into one sparse cloud.

Geometry choices (why, not just how):

* Pair-local triangulation: P1 = K[I|0], P2 = K[R_rel|t_rel], i.e. the
  first camera of the pair is the temporary origin. Only that pair's
  (R, t) enters the reconstruction, so drift accumulated elsewhere in
  the chain cannot distort it. Points are then rigidly moved into the
  trajectory world frame with that frame's pose.
* All good matches are triangulated (not just RANSAC inliers): the
  motion stage owns inliers for POSE, this stage owns point validity for
  MAPPING. Mismatches are caught here by depth + reprojection filters,
  which use only the pair's own data.
* Reprojection gate: a triangulated point must reproject within
  `reproj_thresh` px in BOTH views. Outliers (wrong matches) almost
  never survive this; it is the primary garbage filter.
* Points land in the trajectory world frame, tagged with their segment:
  pairs from different segments live in different frames (see trajectory
  stage), and the tag keeps that explicit instead of silently mixing.
* Far-field sign flips: with tiny per-pair motion, cheirality can lock the
  wrong baseline sign for a pair (see motion stage). Such pairs contribute
  almost nothing here — their true matches fall behind the cameras and are
  rejected as bad_depth — which is visible as a low per-pair survival rate,
  not as corrupted structure.

UNITS — arbitrary, never meters (monocular scale is unobserved; the
trajectory poses already carry that caveat, and it propagates here).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from video_pipeline.camera_motion import STATUS_OK
from video_pipeline.context import VideoContext
from video_pipeline.stages import Stage
from video_pipeline.utils import sibling_output

#: Stamped on terminal output, the .npz, and shared context.
POINTS_SCALE_NOTE = (
    "3D points are in ARBITRARY UNITS, not meters: monocular scale is "
    "unobserved. Points are valid for structure/matching use, not measurement."
)


@dataclass
class TriangulationConfig:
    """Tuning knobs for triangulation.

    Attributes:
        reproj_thresh: Max reprojection error in pixels (worst of both
            views) for a point to survive. Catches wrong matches.
        min_depth: Points at or below this depth in either camera frame
            are degenerate (behind/at the camera) and rejected.
        max_abs_coord: Sanity bound on |X|,|Y|,|Z|; beyond it a point is
            triangulation noise from near-parallel rays, not structure.
        progress_every: Terminal progress cadence in pairs.
        visualize: Master switch for the tri-view image
            (the .npz cloud is always saved).
    """

    reproj_thresh: float = 2.0
    min_depth: float = 1e-6
    max_abs_coord: float = 1e4
    progress_every: int = 25
    visualize: bool = True


def _projection_matrices(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pair-local projections: first camera at origin, second at (R, t)."""
    return (K @ np.hstack([np.eye(3), np.zeros((3, 1))]),
            K @ np.hstack([R, t.reshape(3, 1)]))


def _to_cartesian(homogeneous: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """4xN homogeneous -> (3xN Cartesian, N-vector w). w ~ 0 means at infinity."""
    w = homogeneous[3]
    with np.errstate(divide="ignore", invalid="ignore"):
        points = homogeneous[:3] / np.where(w == 0, np.nan, w)
    return points, w


def _depths(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Depths of 3xN pair-local points in the first and second camera frames."""
    return points[2], (R[2] @ points + t[2])


def _reprojection_errors(points: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray,
                         K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Worst-of-both-views reprojection error in pixels for 3xN points."""
    def project(P: np.ndarray) -> np.ndarray:
        uvw = P @ np.vstack([points, np.ones((1, points.shape[1]))])
        return (uvw[:2] / np.where(uvw[2] == 0, np.nan, uvw[2])).T  # Nx2
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    err_a = np.linalg.norm(project(P1) - pts_a, axis=1)
    err_b = np.linalg.norm(project(P2) - pts_b, axis=1)
    return np.maximum(err_a, err_b)


def _validity_mask(points: np.ndarray, R: np.ndarray, t: np.ndarray,
                   config: TriangulationConfig,
                   reproj: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Accept/reject mask + rejection-cause counts (causes checked in order)."""
    finite = np.isfinite(points).all(axis=0)
    depth_a, depth_b = _depths(points, R, t)
    deep_enough = (depth_a > config.min_depth) & (depth_b > config.min_depth)
    bounded = np.abs(points).max(axis=0) < config.max_abs_coord
    reprojects = reproj <= config.reproj_thresh
    mask = finite & deep_enough & bounded & reprojects
    causes = {
        "non_finite": int((~finite).sum()),
        "bad_depth": int((finite & ~deep_enough).sum()),
        "extreme_coord": int((finite & deep_enough & ~bounded).sum()),
        "reprojection": int((finite & deep_enough & bounded & ~reprojects).sum()),
    }
    return mask, causes


class TriangulationStage(Stage):
    """Stage 7 — triangulate good matches into a sparse 3D point cloud.

    Loads `_orb.npz` (keypoints), `_orb_matches.npz` (correspondences),
    `_motion.npz` (relative poses + validity) and `_trajectory.npz`
    (world placement) — all written by earlier stages, none modified.
    One sequential video pass samples point colors. Records on
    `ctx.shared["points3d"]`:

        shared["points3d"] = {
            "count": ..., "stats": {...}, "rejected": {...},
            "scale_note": ..., "points_path": ..., "triview_path": ...,
        }
    """

    name = "triangulation"

    def __init__(self, config: Optional[TriangulationConfig] = None):
        self.config = config or TriangulationConfig()

    # ------------------------------------------------------------------ #
    # stage entry point
    # ------------------------------------------------------------------ #
    def process(self, ctx: VideoContext) -> VideoContext:
        video_path = ctx.output_path
        orb = self._load_npz(sibling_output(video_path, "_orb.npz"), "ORB")
        matches = self._load_npz(sibling_output(video_path, "_orb_matches.npz"), "matching")
        motion = self._load_npz(sibling_output(video_path, "_motion.npz"), "motion")
        trajectory = self._load_npz(sibling_output(video_path, "_trajectory.npz"), "trajectory")
        K = np.array(motion["K"], dtype=np.float64)

        pair_slices = self._pair_slices(matches)
        world_poses = self._world_poses(trajectory)
        valid = [str(s) == STATUS_OK for s in motion["status"]]

        cloud = self._reconstruct(video_path, orb, pair_slices, motion, world_poses, valid, K)

        points_path = sibling_output(video_path, "_points3d.npz")
        self._save_points(points_path, cloud, K)
        visuals: Dict[str, str] = {}
        if self.config.visualize and cloud["points"]:
            visuals["triview_path"] = str(self._save_triview(video_path, cloud))

        self._report(ctx, points_path, visuals, cloud)
        return ctx

    # ------------------------------------------------------------------ #
    # loading (earlier stages' outputs are read-only inputs here)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_npz(path: Path, stage: str) -> Dict:
        if not path.exists():
            raise IOError(f"{stage} data missing for triangulation (run it first): {path}")
        with np.load(str(path), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    @staticmethod
    def _pair_slices(matches: Dict) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Per-pair (query_idx, train_idx) from the owner mapping."""
        owner = matches["owner_pair"].astype(int)
        n_pairs = int(owner.max(initial=-1)) + 1 if len(owner) else 0
        n_pairs = max(n_pairs, len(matches["good_counts"]))
        return [(matches["query_idx"][owner == i], matches["train_idx"][owner == i])
                for i in range(n_pairs)]

    @staticmethod
    def _world_poses(trajectory: Dict) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """Per-frame (R world->cam, camera center, segment id)."""
        rotations = [np.array(m, dtype=np.float64) for m in trajectory["rotations"]]
        centers = [np.array(p, dtype=np.float64) for p in trajectory["positions"]]
        segments = [int(s) for s in trajectory["segment_ids"]]
        return rotations, centers, segments

    # ------------------------------------------------------------------ #
    # reconstruction (single sequential video pass for colors)
    # ------------------------------------------------------------------ #
    def _reconstruct(self, video_path: Path, orb: Dict,
                     pair_slices: List[Tuple[np.ndarray, np.ndarray]],
                     motion: Dict, world_poses: Tuple, valid: List[bool],
                     K: np.ndarray) -> Dict:
        cfg = self.config
        counts = orb["counts"].astype(int).tolist()
        kp_all = np.array(orb["keypoints"], dtype=np.float64)
        offsets = np.cumsum([0] + counts).tolist()

        R_rel = [np.array(m, dtype=np.float64) for m in motion["R"]]
        t_rel = [np.array(v, dtype=np.float64).reshape(3) for v in motion["t"]]
        rotations, centers, segments = world_poses

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video for triangulation colors: {video_path}")

        points: List[np.ndarray] = []      # world-frame XYZ rows
        colors: List[np.ndarray] = []      # RGB rows
        pairs: List[int] = []
        seg_ids: List[int] = []
        reprojs: List[float] = []
        rejected = {"non_finite": 0, "bad_depth": 0, "extreme_coord": 0, "reprojection": 0}
        n_input = 0
        try:
            ok, frame = cap.read()
            index = 0
            while ok:
                if index < len(pair_slices) and valid[index]:
                    pts_a, pts_b = self._pair_pixels(
                        kp_all, offsets, index, pair_slices[index])
                    n_input += len(pts_a)
                    kept = self._triangulate_pair(
                        pts_a, pts_b, R_rel[index], t_rel[index], K,
                        rotations[index], centers[index], frame, rejected)
                    for xyz, rgb, err in kept:
                        points.append(xyz)
                        colors.append(rgb)
                        pairs.append(index)
                        seg_ids.append(segments[index])
                        reprojs.append(err)
                ok, frame = cap.read()
                index += 1
                if index % cfg.progress_every == 0:
                    print(f"    ... triangulated through frame {index}", end="\r")
        finally:
            cap.release()
        print()
        return {"points": points, "colors": colors, "pairs": pairs,
                "segments": seg_ids, "reproj": reprojs,
                "rejected": rejected, "n_input": n_input,
                "n_pairs_attempted": sum(valid)}

    @staticmethod
    def _pair_pixels(kp_all: np.ndarray, offsets: List[int], frame_index: int,
                     sl: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Matched pixel coordinates (Nx2 float64), pair (frame_index, frame_index+1)."""
        query_idx, train_idx = sl
        xy_a = kp_all[offsets[frame_index]:offsets[frame_index + 1]][:, :2]
        xy_b = kp_all[offsets[frame_index + 1]:offsets[frame_index + 2]][:, :2]
        return xy_a[query_idx].astype(np.float64), xy_b[train_idx].astype(np.float64)

    def _triangulate_pair(self, pts_a: np.ndarray, pts_b: np.ndarray,
                          R: np.ndarray, t: np.ndarray, K: np.ndarray,
                          R_wc: np.ndarray, C_w: np.ndarray,
                          frame_bgr: np.ndarray,
                          rejected: Dict[str, int]) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Triangulate one pair -> [(world XYZ, RGB, reproj_err)] survivors."""
        if len(pts_a) == 0:
            return []
        P1, P2 = _projection_matrices(K, R, t)
        homogeneous = cv2.triangulatePoints(P1, P2, pts_a.T, pts_b.T)
        local, _ = _to_cartesian(homogeneous)
        reproj = _reprojection_errors(local, pts_a, pts_b, K, R, t)
        mask, causes = _validity_mask(local, R, t, self.config, reproj)
        for cause, n in causes.items():
            rejected[cause] += n
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        # Pair-local -> world: X_w = R_wc^T (X_local - c), c = -R_wc C_w.
        # Batched (same arithmetic as the per-point loop); colors by gather.
        h, w = frame_bgr.shape[:2]
        c_wc = -R_wc @ C_w
        worlds = (R_wc.T @ (local[:, idx] - c_wc[:, None])).T
        px = np.clip(np.round(pts_a[idx]).astype(int), [0, 0], [w - 1, h - 1])
        rgbs = frame_bgr[px[:, 1], px[:, 0]][:, ::-1]
        return list(zip(worlds, rgbs, (float(e) for e in reproj[idx])))

    # ------------------------------------------------------------------ #
    # persistence (.npz)
    # ------------------------------------------------------------------ #
    def _save_points(self, path: Path, cloud: Dict, K: np.ndarray) -> None:
        pts = np.array(cloud["points"], dtype=np.float64).reshape(-1, 3)
        cols = np.array(cloud["colors"], dtype=np.uint8).reshape(-1, 3)
        np.savez_compressed(
            path,
            points=pts,                       # (M, 3) world frame, ARBITRARY UNITS
            colors=cols,                      # (M, 3) RGB sampled from first view
            pair=np.array(cloud["pairs"], dtype=np.int32),
            segment=np.array(cloud["segments"], dtype=np.int32),
            reproj_error=np.array(cloud["reproj"], dtype=np.float32),
            K=K,
            scale_note=np.array(POINTS_SCALE_NOTE),
        )

    # ------------------------------------------------------------------ #
    # visualization (orthographic tri-view; OpenCV only)
    # ------------------------------------------------------------------ #
    def _save_triview(self, video_path: Path, cloud: Dict) -> Path:
        path = sibling_output(video_path, "_points3d_triview.png")
        pts = np.array(cloud["points"])
        bgr = np.array(cloud["colors"])[:, ::-1].copy()
        order = np.argsort(pts[:, 2])  # far points first, near drawn over
        pts, bgr = pts[order], bgr[order]
        stride = max(1, len(pts) // 20000)

        views = (("X", "Y", (0, 1)), ("X", "Z", (0, 2)), ("Y", "Z", (1, 2)))
        S, MH = 400, 46
        canvas = np.full((MH + S, 3 * S + 40, 3), 24, dtype=np.uint8)
        cv2.putText(canvas,
                    f"Sparse 3D map: {len(pts)} points "
                    f"(ARBITRARY UNITS - NOT meters)", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for v, (xlabel, ylabel, axes) in enumerate(views):
            ox, oy = 10 + v * S, MH
            ax, ay = pts[::stride, axes[0]], pts[::stride, axes[1]]
            lo_x, hi_x = ax.min(), ax.max()
            lo_y, hi_y = ay.min(), ay.max()
            span = max(hi_x - lo_x, hi_y - lo_y) or 1.0

            def loc(px: float, py: float) -> Tuple[int, int]:
                return (round(ox + (px - lo_x) / span * (S - 20)),
                        round(oy + S - 10 - (py - lo_y) / span * (S - 20)))

            for (px, py), color in zip(zip(ax, ay), bgr[::stride]):
                cv2.circle(canvas, loc(px, py), 2, tuple(int(c) for c in color), -1)
            cv2.putText(canvas, f"{xlabel}-{ylabel}", (ox + 8, oy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            cv2.putText(canvas, f"[{lo_x:+.1f},{hi_x:+.1f}]", (ox + 110, oy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.imwrite(str(path), canvas)
        return path

    # ------------------------------------------------------------------ #
    # terminal report
    # ------------------------------------------------------------------ #
    def _report(self, ctx: VideoContext, points_path: Path,
                visuals: Dict[str, str], cloud: Dict) -> None:
        n_final = len(cloud["points"])
        n_rejected = sum(cloud["rejected"].values())
        stats = {
            "pairs_attempted": cloud["n_pairs_attempted"],
            "input_matches": cloud["n_input"],
            "triangulated": cloud["n_input"],  # one 3D candidate per input match
            "rejected": dict(cloud["rejected"]),
            "rejected_total": n_rejected,
            "final_points": n_final,
        }
        ctx.shared["points3d"] = {
            "count": n_final,
            "stats": stats,
            "scale_note": POINTS_SCALE_NOTE,
            "points_path": str(points_path),
            **visuals,
        }
        line = "-" * 52
        print(f"\n{line}\nTRIANGULATION (sparse 3D map)\n{line}")
        print(f"  Pairs attempted   : {stats['pairs_attempted']}")
        print(f"  Input matches     : {stats['input_matches']}")
        print(f"  Triangulated      : {stats['triangulated']}")
        print(f"  Rejected total    : {n_rejected} {cloud['rejected']}")
        print(f"  Final 3D points   : {n_final}")
        print(f"  NOTE: {POINTS_SCALE_NOTE}")
        print(f"  Points saved      : {points_path}")
        for label, vis_path in visuals.items():
            print(f"  {label:<15}: {vis_path}")
        print(line)
