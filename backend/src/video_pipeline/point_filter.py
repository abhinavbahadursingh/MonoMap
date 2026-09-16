"""Point-cloud filtering stage: clean the Phase 6 sparse map.

Phase 6 already gates candidates (finite, positive depth, absolute bounds,
2 px reprojection), so repeating those checks here would be a no-op stage.
This stage adds only what Phase 6 lacks, cheapest first, with a survivor
waterfall so every tier justifies itself in the terminal:

1. Integrity recheck — finite coordinates and positive pair-local depth,
   recomputed through the trajectory world poses. Phase 6 guarantees
   these, so any rejection here is LOUD: it means a stored transform or
   array got corrupted, not that the scene is bad.
2. Strict reprojection tier — the per-point errors stored in Phase 6 were
   measured against the original 2D observations; re-measuring identical
   arithmetic would add no information, so this stage applies the strict
   map tier (default 1 px vs Phase 6's 2 px candidate tier) on those
   stored errors. Coarse-to-fine, on purpose.
3. Relative depth bound — depth beyond `depth_median_factor` x the pair's
   own median depth is far-flung triangulation noise. Relative to the
   scene's scale, never an absolute constant (monocular units are
   arbitrary, so an absolute bound cannot mean anything).
4. Statistical outlier removal (SOR) — per segment (segments live in
   different frames; cross-segment neighbors are meaningless): drop
   points whose mean distance to their `sor_k` nearest neighbors exceeds
   the segment mean + `sor_std_ratio` std. Exact brute-force neighbors,
   chunked for memory — no randomness anywhere.

Output schema is identical to the Phase 6 `.npz` (same keys), so anything
downstream that reads a cloud keeps working on the filtered file.
UNITS — arbitrary, never meters (propagated from the motion stage).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from video_pipeline.context import VideoContext
from video_pipeline.stages import Stage
from video_pipeline.utils import render_colored_triview, sibling_output

#: Stamped on terminal output, the .npz, and shared context.
FILTER_SCALE_NOTE = (
    "Filtered points remain in ARBITRARY UNITS, not meters: filtering changes "
    "membership, never scale. Not for metric measurement."
)


@dataclass
class PointFilterConfig:
    """Tuning knobs for cloud filtering.

    Attributes:
        reproj_strict: Strict map-tier reprojection cap in pixels on the
            stored Phase 6 errors (must be <= the 2 px candidate tier to
            mean anything; lower is cleaner but sparser).
        depth_median_factor: Keep depths within this multiple of their
            pair's median depth. Generous by design: true scenes span a
            few multiples, noise spans orders of magnitude.
        min_depth: Integrity floor for recomputed pair-local depth.
        sor_k: Neighbors per point for statistical outlier removal.
        sor_std_ratio: SOR cutoff in standard deviations above the
            segment's mean neighbor distance (higher keeps more).
        visualize: Master switch for the filtered tri-view image
            (the filtered .npz is always saved).
    """

    reproj_strict: float = 1.0
    depth_median_factor: float = 10.0
    min_depth: float = 1e-6
    sor_k: int = 8
    sor_std_ratio: float = 2.0
    visualize: bool = True


def _pair_local_depths(points: np.ndarray, rotations: np.ndarray,
                       centers: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Depth of each world point in its own pair's first camera frame."""
    depths = np.empty(len(points))
    for pair in np.unique(pairs):
        sel = pairs == pair
        R_wc = rotations[int(pair)]
        C_w = centers[int(pair)]
        depths[sel] = (R_wc @ (points[sel].T - C_w[:, None]))[2]
    return depths


def _relative_depth_mask(depths: np.ndarray, pairs: np.ndarray,
                         factor: float) -> np.ndarray:
    """Keep depths within `factor` x their pair's median (pairs need >= 3 points)."""
    keep = np.ones(len(depths), bool)
    for pair in np.unique(pairs):
        sel = np.where(pairs == pair)[0]
        if len(sel) < 3:
            continue  # no scale estimate possible; keep, don't guess
        keep[sel] = depths[sel] <= factor * np.median(depths[sel])
    return keep


def _sor_mask(points: np.ndarray, segments: np.ndarray,
              k: int, std_ratio: float) -> np.ndarray:
    """Statistical outlier removal, per segment, exact neighbors.

    Uses a kd-tree for exact k-nearest neighbors — the same distances as
    brute force in O(N log N) instead of O(N^2) per segment. Neighbors are
    exact (eps=0), so the mask matches brute force; only the scholar's
    patience is spared.
    """
    keep = np.ones(len(points), bool)
    for seg in np.unique(segments):
        idx = np.where(segments == seg)[0]
        if len(idx) <= k:
            continue  # too few for neighbor statistics; keep, don't guess
        block = np.array(points[idx], dtype=np.float64)
        dists, _ = cKDTree(block).query(block, k=k + 1)
        mean_dist = dists[:, 1:].mean(axis=1)
        cutoff = mean_dist.mean() + std_ratio * mean_dist.std()
        keep[idx] = mean_dist <= cutoff
    return keep


class PointFilterStage(Stage):
    """Stage 8 — filter the Phase 6 cloud; save the cleaned map.

    Loads `_points3d.npz` + `_trajectory.npz` (read-only inputs) and writes
    `<stem>_points3d_filtered.npz` in the same schema. Records on
    `ctx.shared["points3d_filtered"]`:

        shared["points3d_filtered"] = {
            "count": ..., "stats": {... waterfall ...},
            "scale_note": ..., "filtered_path": ..., "triview_path": ...,
        }
    """

    name = "point_filter"

    def __init__(self, config: Optional[PointFilterConfig] = None):
        self.config = config or PointFilterConfig()

    # ------------------------------------------------------------------ #
    # stage entry point
    # ------------------------------------------------------------------ #
    def process(self, ctx: VideoContext) -> VideoContext:
        video_path = ctx.output_path
        points_path = sibling_output(video_path, "_points3d.npz")
        if not points_path.exists():
            raise IOError(
                f"3D points missing for filtering (run triangulation first): {points_path}"
            )
        traj_path = sibling_output(video_path, "_trajectory.npz")
        if not traj_path.exists():
            raise IOError(
                f"Trajectory missing for depth recheck (run it first): {traj_path}"
            )
        with np.load(str(points_path), allow_pickle=False) as z:
            cloud = {k: z[k] for k in z.files}
        with np.load(str(traj_path), allow_pickle=False) as z:
            rotations = [np.array(m, dtype=np.float64) for m in z["rotations"]]
            centers = [np.array(p, dtype=np.float64) for p in z["positions"]]

        waterfall = self._filter(cloud, rotations, centers)

        filtered_path = sibling_output(video_path, "_points3d_filtered.npz")
        self._save_filtered(filtered_path, cloud, waterfall["mask"])
        visuals: Dict[str, str] = {}
        if self.config.visualize and waterfall["mask"].any():
            visuals["triview_path"] = str(self._save_triview(
                video_path, cloud, waterfall["mask"]))

        self._report(ctx, filtered_path, visuals, cloud, waterfall)
        return ctx

    # ------------------------------------------------------------------ #
    # filtering (waterfall: every tier reports its survivors)
    # ------------------------------------------------------------------ #
    def _filter(self, cloud: Dict, rotations: List[np.ndarray],
                centers: List[np.ndarray]) -> Dict:
        cfg = self.config
        points = np.array(cloud["points"], dtype=np.float64)
        n = len(points)
        pairs = np.array(cloud["pair"]).astype(int)
        segments = np.array(cloud["segment"]).astype(int)
        reproj = np.array(cloud["reproj_error"], dtype=np.float64)
        tiers: List[tuple] = []

        alive = np.isfinite(points).all(axis=1)
        tiers.append(("finite", alive))

        depths = _pair_local_depths(points, rotations, centers, pairs)
        alive = alive & (depths > cfg.min_depth)
        tiers.append(("depth_recheck", alive))

        alive = alive & (reproj <= cfg.reproj_strict)
        tiers.append((f"reproj<={cfg.reproj_strict:g}px", alive))

        alive = alive & _relative_depth_mask(depths, pairs, cfg.depth_median_factor)
        tiers.append((f"depth<{cfg.depth_median_factor:g}x pair-median", alive))

        alive = alive & _sor_mask(points, segments, cfg.sor_k, cfg.sor_std_ratio)
        tiers.append((f"SOR(k={cfg.sor_k},>{cfg.sor_std_ratio:g}std)", alive))

        counts = [n] + [int(t[1].sum()) for t in tiers]
        return {"mask": alive,
                "tiers": [(name, before - after)
                          for (name, _), before, after
                          in zip(tiers, counts[:-1], counts[1:])],
                "survivors": counts[1:]}

    # ------------------------------------------------------------------ #
    # persistence (.npz, same schema as the input cloud)
    # ------------------------------------------------------------------ #
    def _save_filtered(self, path: Path, cloud: Dict, mask: np.ndarray) -> None:
        carry = {}
        for key in ("points", "colors", "pair", "segment", "reproj_error"):
            arr = np.array(cloud[key])
            carry[key] = arr[mask] if len(arr) == len(mask) else arr
        np.savez_compressed(
            path,
            **carry,
            K=np.array(cloud["K"]),
            scale_note=np.array(FILTER_SCALE_NOTE),
            input_count=np.array(len(mask)),
            reproj_strict=np.array(self.config.reproj_strict),
            sor_k=np.array(self.config.sor_k),
            sor_std_ratio=np.array(self.config.sor_std_ratio),
        )

    # ------------------------------------------------------------------ #
    # visualization
    # ------------------------------------------------------------------ #
    def _save_triview(self, video_path: Path, cloud: Dict, mask: np.ndarray) -> Path:
        path = sibling_output(video_path, "_points3d_filtered_triview.png")
        render_colored_triview(
            path,
            np.array(cloud["points"])[mask],
            np.array(cloud["colors"])[mask],
            f"Filtered sparse map: {int(mask.sum())} points "
            f"(ARBITRARY UNITS - NOT meters)")
        return path

    # ------------------------------------------------------------------ #
    # terminal report
    # ------------------------------------------------------------------ #
    def _report(self, ctx: VideoContext, filtered_path: Path,
                visuals: Dict[str, str], cloud: Dict, waterfall: Dict) -> None:
        n_in = len(cloud["points"])
        n_out = int(waterfall["mask"].sum())
        stats = {
            "input_points": n_in,
            "tiers": [{"tier": name, "rejected": rej} for name, rej in waterfall["tiers"]],
            "rejected_total": n_in - n_out,
            "final_points": n_out,
        }
        ctx.shared["points3d_filtered"] = {
            "count": n_out,
            "stats": stats,
            "scale_note": FILTER_SCALE_NOTE,
            "filtered_path": str(filtered_path),
            **visuals,
        }
        line = "-" * 52
        print(f"\n{line}\nPOINT CLOUD FILTERING\n{line}")
        print(f"  Before            : {n_in} points")
        for i, (name, _) in enumerate(waterfall["tiers"]):
            print(f"    after {name:<28}: {waterfall['survivors'][i]}")
        print(f"  After             : {n_out} points "
              f"({n_in - n_out} rejected, "
              f"{100.0 * (n_in - n_out) / max(n_in, 1):.1f}%)")
        print(f"  NOTE: {FILTER_SCALE_NOTE}")
        print(f"  Filtered saved    : {filtered_path}")
        for label, vis_path in visuals.items():
            print(f"  {label:<15}: {vis_path}")
        print(line)
