"""Trajectory stage: chain relative (R, t) pairs into per-frame camera poses.

Frame 0 defines the world frame: position [0, 0, 0], rotation identity.
Each valid pair (i -> i+1) composes onto the running pose:

    R_{i+1} = R_rel @ R_i
    c_{i+1} = R_rel @ c_i + step_scale * t_rel
    C_{i+1} = -R_{i+1}^T @ c_{i+1}          (camera center in world frame)

where (R_i, c_i) is the world-to-camera transform. Saved convention:
``rotations`` map world to camera frame; ``positions`` are camera centers
in the world frame.

UNITS — positions are in arbitrary units, never meters: every monocular
`t` is a direction with its own unknown scale, and chaining uses one
uniform `step_scale` for all pairs. Even the trajectory SHAPE is only
indicative (true per-pair scales are unobserved). Nothing here may be
reported, plotted, or consumed as metric distance.

Failed pairs break the chain: frames after a gap cannot be connected to
frame 0, so each maximal run of valid pairs becomes its own SEGMENT with
a local origin. Segments are plotted separately and never joined by lines
— drawing a line across a failed pair would fabricate motion that was
never measured.
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
TRAJECTORY_SCALE_NOTE = (
    "Positions are in ARBITRARY UNITS, not meters: each monocular t has its own "
    "unknown scale. Shape is indicative only; do not use as metric distance."
)

_SEGMENT_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 165, 255),
]


@dataclass
class TrajectoryConfig:
    """Tuning knobs for trajectory accumulation.

    Attributes:
        step_scale: Uniform scale applied to every pair's unit `t`.
            Must stay a plain constant: per-pair true scales are
            unobservable monocularly, and estimating them is a separate
            (bundle-adjustment) phase, not a constant to tune here.
        visualize: Master switch for the X-Z plot output
            (the .npz trajectory is always saved).
    """

    step_scale: float = 1.0
    visualize: bool = True


def _chain_poses(pair_R: List[np.ndarray], pair_t: List[Optional[np.ndarray]],
                 step_scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose the chain.

    Args:
        pair_R: Relative rotation per pair (any placeholder where invalid).
        pair_t: Relative unit translation per pair, or None where the pair failed.

    Returns:
        positions (F, 3) camera centers, rotations (F, 3, 3) world-to-camera,
        segment_ids (F,) int32. Frame 0 is always segment 0 at the origin.
    """
    n_frames = len(pair_t) + 1
    positions = np.zeros((n_frames, 3))
    rotations = np.repeat(np.eye(3)[None, :, :], n_frames, axis=0)
    segment_ids = np.zeros(n_frames, dtype=np.int32)

    R_cur, c_cur, segment = np.eye(3), np.zeros(3), 0
    for f in range(1, n_frames):
        t_rel = pair_t[f - 1]
        if t_rel is None:
            segment += 1  # chain broken: fresh local origin, never a held pose
            R_cur, c_cur = np.eye(3), np.zeros(3)
        else:
            R_cur = pair_R[f - 1] @ R_cur
            c_cur = pair_R[f - 1] @ c_cur + step_scale * t_rel
        positions[f] = -R_cur.T @ c_cur
        rotations[f] = R_cur
        segment_ids[f] = segment
    return positions, rotations, segment_ids


class TrajectoryStage(Stage):
    """Stage 6 — accumulate relative poses into a (segmented) trajectory.

    Consumes `_motion.npz` (loaded from disk, so this stage also runs
    standalone). Records on `ctx.shared["trajectory"]`:

        shared["trajectory"] = {
            "positions": [...],       # (F, 3) camera centers, arbitrary units
            "rotations": [...],       # (F, 3, 3) world-to-camera
            "segment_ids": [...],
            "segments": [{"id", "first_frame", "last_frame", "length"}, ...],
            "stats": {...},
            "convention": ..., "scale_note": ...,
            "trajectory_path": ..., "xz_plot_path": ...,
        }
    """

    name = "trajectory"

    def __init__(self, config: Optional[TrajectoryConfig] = None):
        self.config = config or TrajectoryConfig()

    # ------------------------------------------------------------------ #
    # stage entry point
    # ------------------------------------------------------------------ #
    def process(self, ctx: VideoContext) -> VideoContext:
        video_path = ctx.output_path
        motion_path = sibling_output(video_path, "_motion.npz")
        if not motion_path.exists():
            raise IOError(
                f"Motion data missing for trajectory stage (run motion first): {motion_path}"
            )
        with np.load(str(motion_path)) as z:
            R_all = [np.array(m) for m in z["R"]]
            t_all = [np.array(v) for v in z["t"]]
            status = [str(s) for s in z["status"]]

        pair_R = [R_all[i] if s == STATUS_OK else np.eye(3) for i, s in enumerate(status)]
        pair_t: List[Optional[np.ndarray]] = [
            t_all[i] if s == STATUS_OK else None for i, s in enumerate(status)
        ]
        positions, rotations, segment_ids = _chain_poses(
            pair_R, pair_t, self.config.step_scale)

        trajectory_path = sibling_output(video_path, "_trajectory.npz")
        self._save_trajectory(trajectory_path, positions, rotations, segment_ids, status)
        visuals: Dict[str, str] = {}
        if self.config.visualize:
            visuals["xz_plot_path"] = str(self._save_xz_plot(
                video_path, positions, segment_ids))

        self._report(ctx, trajectory_path, visuals, positions, rotations,
                     segment_ids, status)
        return ctx

    # ------------------------------------------------------------------ #
    # persistence (.npz)
    # ------------------------------------------------------------------ #
    def _save_trajectory(self, path: Path, positions: np.ndarray,
                         rotations: np.ndarray, segment_ids: np.ndarray,
                         status: List[str]) -> None:
        np.savez_compressed(
            path,
            positions=positions,      # (F, 3) camera centers, ARBITRARY UNITS
            rotations=rotations,      # (F, 3, 3) world-to-camera per frame
            segment_ids=segment_ids,  # (F,) int32
            pair_ok=np.array([s == STATUS_OK for s in status], dtype=bool),
            step_scale=np.array(self.config.step_scale),
            convention=np.array(
                "rotations map world to camera frame; positions are camera "
                "centers in world frame; frame 0 is the origin with identity rotation"),
            scale_note=np.array(TRAJECTORY_SCALE_NOTE),
        )

    # ------------------------------------------------------------------ #
    # visualization
    # ------------------------------------------------------------------ #
    def _save_xz_plot(self, video_path: Path, positions: np.ndarray,
                      segment_ids: np.ndarray) -> Path:
        """Top-down X-Z view: one polyline per segment, breaks never joined."""
        path = sibling_output(video_path, "_trajectory_xz.png")
        W, H, M = 1200, 700, 70
        img = np.full((H, W, 3), 24, dtype=np.uint8)

        span = float(np.abs(positions[:, [0, 2]]).max(initial=0.0)) or 1.0
        scale = min((W - 2 * M), (H - 2 * M)) / (2 * span * 1.1)
        cx, cy = W // 2, H // 2

        def located(p: np.ndarray) -> Tuple[int, int]:
            return (round(cx + p[0] * scale), round(cy - p[2] * scale))

        cv2.line(img, (M, cy), (W - M, cy), (60, 60, 60), 1)  # X axis
        cv2.line(img, (cx, M), (cx, H - M), (60, 60, 60), 1)  # Z axis
        cv2.putText(img, "X", (W - M + 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(img, "Z", (cx + 8, M - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        n_segments = int(segment_ids.max()) + 1
        for seg in range(n_segments):
            idx = np.where(segment_ids == seg)[0]
            color = _SEGMENT_COLORS[seg % len(_SEGMENT_COLORS)]
            pts = np.array([located(positions[i]) for i in idx], dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts], False, color, 2)
            for i in idx:
                cv2.circle(img, located(positions[i]), 3, color, -1)
            cv2.circle(img, located(positions[idx[0]]), 7, color, 2)       # segment start
            cv2.drawMarker(img, located(positions[idx[-1]]), color,         # segment end
                           cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.putText(img,
                    f"Top-down trajectory X-Z ({len(positions)} frames, "
                    f"{n_segments} segment(s), ARBITRARY UNITS - NOT meters)",
                    (M, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(path), img)
        return path

    # ------------------------------------------------------------------ #
    # terminal report
    # ------------------------------------------------------------------ #
    def _report(self, ctx: VideoContext, trajectory_path: Path,
                visuals: Dict[str, str], positions: np.ndarray,
                rotations: np.ndarray, segment_ids: np.ndarray,
                status: List[str]) -> None:
        segments = [
            {"id": int(seg),
             "first_frame": int(idx[0]),
             "last_frame": int(idx[-1]),
             "length": len(idx)}
            for seg in range(int(segment_ids.max()) + 1)
            for idx in [np.where(segment_ids == seg)[0]]
        ]
        n_valid = sum(1 for s in status if s == STATUS_OK)
        longest = max(segments, key=lambda s: s["length"])
        stats = {
            "frames": len(positions),
            "pairs": len(status),
            "valid_pairs": n_valid,
            "failed_pairs": [i for i, s in enumerate(status) if s != STATUS_OK],
            "segments": segments,
        }
        ctx.shared["trajectory"] = {
            "positions": positions.tolist(),
            "rotations": rotations.tolist(),
            "segment_ids": segment_ids.tolist(),
            "segments": segments,
            "stats": stats,
            "convention": ("world-to-camera rotations; camera centers in world "
                           "frame; frame 0 is origin with identity rotation"),
            "scale_note": TRAJECTORY_SCALE_NOTE,
            "trajectory_path": str(trajectory_path),
            **visuals,
        }
        start = positions[0]
        end = positions[longest["last_frame"]]
        line = "-" * 52
        print(f"\n{line}\nTRAJECTORY (chained relative poses)\n{line}")
        print(f"  Start position  : [{start[0]:+.4f} {start[1]:+.4f} {start[2]:+.4f}] "
              f"(frame 0, origin)")
        print(f"  End position    : [{end[0]:+.4f} {end[1]:+.4f} {end[2]:+.4f}] "
              f"(frame {longest['last_frame']}, longest segment #{longest['id']})")
        print(f"  Pairs           : {stats['pairs']} "
              f"({n_valid} valid, {len(stats['failed_pairs'])} failed)")
        print(f"  Segments        : {len(segments)} "
              f"(failed pairs break the chain; segments are never joined)")
        for seg in segments:
            print(f"    segment #{seg['id']}: frames {seg['first_frame']}..{seg['last_frame']} "
                  f"({seg['length']} frames)")
        print(f"  NOTE: {TRAJECTORY_SCALE_NOTE}")
        print(f"  Trajectory saved: {trajectory_path}")
        for label, vis_path in visuals.items():
            print(f"  {label:<15}: {vis_path}")
        print(line)
