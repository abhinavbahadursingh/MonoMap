"""Camera motion stage: per-pair rotation R and translation t from good matches.

For every consecutive frame pair, estimates the Essential matrix with
RANSAC (`cv2.findEssentialMat`) and decomposes it (`cv2.recoverPose`)
into rotation R and translation t, plus the inlier count.

SCALE — read this before using `t` downstream:

* A single uncalibrated camera cannot observe absolute scale. `t` is a
  UNIT vector (|t| = 1): it carries the direction of motion only. Two
  rides — 1 m and 100 m along the same heading — produce the same `t`.
* Consequences: consecutive `t`s must NOT be summed into meters, and any
  trajectory chaining needs an external scale source (IMU, wheel odometry,
  known object size, ...). The stage therefore reports direction only and
  stamps this note on the terminal, the .npz, and `ctx.shared["motion"]`.

Intrinsics — the Essential matrix needs K, and this video is
uncalibrated, so K is *assumed*: principal point at the image center and
focal length = max(width, height) unless overridden. That assumption is
the dominant error source for R/t; with a calibrated K, pass it via
`MotionConfig(focal_length=..., ...)`. Pairs that cannot support a pose
(< `min_points_for_pose` matches, RANSAC failure) are reported as failed,
never filled with plausible-looking numbers.

Degeneracy note — a flat fronto-parallel scene shifting uniformly is
inherently ambiguous (lateral translation vs a small rotation explain it
equally well). Trustworthy translation needs depth variation in view
(motion parallax); treat `t` from flat, texture-uniform stretches with
suspicion even when the inlier count is high.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from video_pipeline.context import VideoContext
from video_pipeline.orb_matching import _load_frame_data  # .npz layout owned by matching stage
from video_pipeline.stages import Stage
from video_pipeline.utils import save_line_chart, sibling_output

#: Stamped on terminal output, the .npz, and shared context — impossible to miss.
SCALE_NOTE = (
    "Monocular t is DIRECTION ONLY (|t| = 1): absolute scale is unobservable "
    "with a single camera. Do NOT integrate t into meters without an external scale source."
)

STATUS_OK = "ok"


@dataclass
class MotionConfig:
    """Tuning knobs for camera motion estimation.

    Attributes:
        focal_length: Assumed focal length in pixels (None -> max(width, height)
            of the normalized video). Override with a calibrated value when known.
        ransac_thresh: RANSAC inlier threshold in pixels for findEssentialMat.
        ransac_prob: Desired RANSAC success probability.
        min_points_for_pose: Minimum good matches to attempt a pose (the
            5-point solver needs >= 5; 8 keeps RANSAC stable).
        print_matrices: Print the full 3x3 R per pair (False -> one line per pair).
        progress_every: Terminal progress cadence in pairs.
        visualize: Master switch for the inlier-timeline output
            (the .npz motion is always saved).
    """

    focal_length: Optional[float] = None
    ransac_thresh: float = 1.0
    ransac_prob: float = 0.999
    min_points_for_pose: int = 8
    print_matrices: bool = True
    progress_every: int = 25
    visualize: bool = True


def _intrinsics(width: int, height: int, focal: Optional[float]) -> np.ndarray:
    """Assumed K: center principal point + given (or image-max) focal length."""
    f = float(focal) if focal else float(max(width, height))
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _pair_points(kp_a: np.ndarray, kp_b: np.ndarray,
                 query_idx: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (Nx2 float32) of the matched keypoints in both frames."""
    return (kp_a[query_idx][:, :2].astype(np.float32),
            kp_b[train_idx][:, :2].astype(np.float32))


def _estimate_pair(points_a: np.ndarray, points_b: np.ndarray,
                   K: np.ndarray, config: MotionConfig) -> Dict:
    """Essential+RANSAC -> recoverPose. Failures keep NaN pose + a status string."""
    result: Dict = {"R": np.full((3, 3), np.nan), "t": np.full(3, np.nan),
                    "inliers": 0, "status": STATUS_OK}
    if len(points_a) < config.min_points_for_pose:
        result["status"] = f"failed: only {len(points_a)} matches (< {config.min_points_for_pose})"
        return result
    essential, emask = cv2.findEssentialMat(
        points_a, points_b, K, cv2.RANSAC, config.ransac_prob, config.ransac_thresh)
    if essential is None:
        result["status"] = "failed: RANSAC found no Essential matrix"
        return result
    inliers, R, t, _ = cv2.recoverPose(essential, points_a, points_b, K, emask)
    if inliers < config.min_points_for_pose:
        # An unconverged pose is not a pose: keep NaN so no downstream
        # consumer can mistake leftovers for an estimate. The inlier count
        # and status already preserve the diagnostic information.
        result["status"] = (f"failed: only {inliers} inliers "
                            f"(< {config.min_points_for_pose})")
        return result
    result.update({"R": R, "t": t.reshape(3), "inliers": int(inliers)})
    return result


class CameraMotionStage(Stage):
    """Stage 5 — per-pair camera rotation R and translation direction t.

    Consumes the `_orb.npz` keypoints and `_orb_matches.npz` good matches
    (both written by earlier stages; loaded from disk so this stage also
    runs standalone). Records on `ctx.shared["motion"]`:

        shared["motion"] = {
            "R": [...],             # (3, 3) per pair (NaN rows where failed)
            "t": [...],             # unit vectors per pair (NaN where failed)
            "inliers": [...],
            "status": [...],
            "stats": {...},
            "K": [[...]], "focal_length": ...,
            "scale_note": ...,      # SCALE_NOTE, always
            "motion_path": ..., "timeline_path": ...,
        }
    """

    name = "camera_motion"

    def __init__(self, config: Optional[MotionConfig] = None):
        self.config = config or MotionConfig()

    # ------------------------------------------------------------------ #
    # stage entry point
    # ------------------------------------------------------------------ #
    def process(self, ctx: VideoContext) -> VideoContext:
        video_path = ctx.output_path
        orb_path = sibling_output(video_path, "_orb.npz")
        matches_path = sibling_output(video_path, "_orb_matches.npz")
        for needed in (orb_path, matches_path):
            if not needed.exists():
                raise IOError(
                    f"Motion stage input missing (run ORB + matching first): {needed}"
                )
        # Normalized dims are authored by this pipeline, so they are exact.
        K = _intrinsics(ctx.target_width, ctx.target_height, self.config.focal_length)

        kp_arrays, _, _ = _load_frame_data(orb_path)
        good_per_pair = self._load_good_matches(matches_path, len(kp_arrays) - 1)

        results = []
        for i, (query_idx, train_idx) in enumerate(good_per_pair):
            points_a, points_b = _pair_points(
                kp_arrays[i], kp_arrays[i + 1], query_idx, train_idx)
            outcome = _estimate_pair(points_a, points_b, K, self.config)
            outcome["pair"] = (i, i + 1)
            outcome["good"] = len(query_idx)
            results.append(outcome)
            self._print_pair(outcome)
            if (i + 1) % self.config.progress_every == 0:
                print(f"    ... estimated {i + 1}/{len(good_per_pair)} pairs", end="\r")
        print()

        motion_path = sibling_output(video_path, "_motion.npz")
        self._save_motion(motion_path, results, K)
        visuals: Dict[str, str] = {}
        if self.config.visualize:
            visuals["timeline_path"] = str(self._save_timeline(video_path, results))
        self._report(ctx, motion_path, visuals, results, K)
        return ctx

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    def _load_good_matches(self, matches_path: Path,
                           n_pairs: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Per-pair (query_idx, train_idx) arrays, sliced via owner_pair."""
        with np.load(str(matches_path)) as z:
            owner = z["owner_pair"].astype(int)
            query, train = z["query_idx"], z["train_idx"]
        return [(query[owner == i], train[owner == i]) for i in range(n_pairs)]

    # ------------------------------------------------------------------ #
    # persistence (.npz)
    # ------------------------------------------------------------------ #
    def _save_motion(self, path: Path, results: List[Dict], K: np.ndarray) -> None:
        np.savez_compressed(
            path,
            R=np.array([r["R"] for r in results]),            # (P, 3, 3), NaN where failed
            t=np.array([r["t"] for r in results]),            # (P, 3) unit vectors, NaN where failed
            inliers=np.array([r["inliers"] for r in results], dtype=np.int32),
            good=np.array([r["good"] for r in results], dtype=np.int32),
            status=np.array([r["status"] for r in results]),
            K=K,
            focal_length=np.array(K[0, 0]),
            scale_note=np.array(SCALE_NOTE),                  # stamped with the data
            ransac_thresh=np.array(self.config.ransac_thresh),
        )

    # ------------------------------------------------------------------ #
    # visualization
    # ------------------------------------------------------------------ #
    def _save_timeline(self, video_path: Path, results: List[Dict]) -> Path:
        path = sibling_output(video_path, "_motion_timeline.png")
        inliers = [r["inliers"] for r in results]
        failed = [i for i, r in enumerate(results) if r["status"] != STATUS_OK]
        save_line_chart(
            path,
            f"Pose inliers per pair ({len(results)} pairs, {len(failed)} failed)",
            inliers,
            threshold=self.config.min_points_for_pose,
            threshold_label=f"pose min={self.config.min_points_for_pose}",
            bad_indices=failed,
        )
        return path

    # ------------------------------------------------------------------ #
    # terminal report
    # ------------------------------------------------------------------ #
    def _print_pair(self, outcome: Dict) -> None:
        i, j = outcome["pair"]
        print(f"    pair {i}->{j}: good={outcome['good']} "
              f"inliers={outcome['inliers']} status={outcome['status']}")
        if self.config.print_matrices and outcome["status"] == STATUS_OK:
            t = outcome["t"]
            print(f"      t (unit, direction only) = "
                  f"[{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}]")
            for row in outcome["R"]:
                print(f"      R = [{row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f}]")

    def _report(self, ctx: VideoContext, motion_path: Path,
                visuals: Dict[str, str], results: List[Dict], K: np.ndarray) -> None:
        inliers = np.array([r["inliers"] for r in results], dtype=np.int32)
        failed = [i for i, r in enumerate(results) if r["status"] != STATUS_OK]
        stats = {
            "pairs": len(results),
            "ok_pairs": len(results) - len(failed),
            "failed_pairs": failed,
            "inliers_min": int(inliers.min()),
            "inliers_mean": round(float(inliers.mean()), 1),
            "inliers_max": int(inliers.max()),
        }
        ctx.shared["motion"] = {
            "R": [r["R"].tolist() for r in results],
            "t": [r["t"].tolist() for r in results],
            "inliers": inliers.tolist(),
            "status": [r["status"] for r in results],
            "stats": stats,
            "K": K.tolist(),
            "focal_length": float(K[0, 0]),
            "scale_note": SCALE_NOTE,
            "motion_path": str(motion_path),
            **visuals,
        }
        line = "-" * 52
        print(f"\n{line}\nCAMERA MOTION (Essential RANSAC + recoverPose)\n{line}")
        print(f"  Pairs           : {stats['pairs']} "
              f"({stats['ok_pairs']} ok, {len(failed)} failed)")
        print(f"  Inliers min/mean/max  : {stats['inliers_min']} / "
              f"{stats['inliers_mean']} / {stats['inliers_max']}")
        print(f"  K (assumed)     : f={K[0, 0]:.1f}, "
              f"center=({K[0, 2]:.0f}, {K[1, 2]:.0f})")
        if failed:
            print(f"  !! Failed pairs: {failed}")
            for i in failed:
                print(f"     pair {results[i]['pair'][0]}->{results[i]['pair'][1]}: "
                      f"{results[i]['status']}")
        print(f"  NOTE: {SCALE_NOTE}")
        print(f"  Motion saved    : {motion_path}")
        for label, vis_path in visuals.items():
            print(f"  {label:<15}: {vis_path}")
        print(line)
