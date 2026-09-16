"""Keyframe selection stage: pick reference frames for mapping.

Frame 0 is always a keyframe. Every later frame is evaluated against the
last keyframe, and a new keyframe is taken when any threshold fires:

* translation — world-frame distance from the last keyframe >=
  `trans_thresh` (same trajectory segment only; units are the arbitrary
  trajectory units, so tune per scene scale).
* rotation — angle from the last keyframe >= `rot_thresh_deg`.
* overlap — direct BFMatcher+Lowe matches against the last keyframe <
  `min_matches`. Consecutive-pair counts cannot substitute here: they do
  not measure overlap with the keyframe, so each frame is re-matched
  against it. Matching policy (Hamming + Lowe ratio) is reused from the
  matching stage, including its stored ratio — not duplicated.
* interval — frames since the last keyframe >= `max_interval`, the
  backstop that bounds keyframe spacing no matter what.

A trajectory segment break (tracking gap with no measured transform)
forces a keyframe: motion against the last keyframe is unmeasurable, so
starting a fresh reference is the only honest choice. Reasons are
recorded per keyframe — every selection is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from video_pipeline.context import VideoContext
from video_pipeline.orb_matching import _good_matches, _load_frame_data
from video_pipeline.stages import Stage
from video_pipeline.utils import sibling_output

REASON_FIRST = "first"
REASON_INTERVAL = "interval"
REASON_TRANSLATION = "translation"
REASON_ROTATION = "rotation"
REASON_LOW_MATCHES = "low_matches"
REASON_SEGMENT_BREAK = "segment_break"


@dataclass
class KeyframeConfig:
    """Tuning knobs for keyframe selection.

    Attributes:
        trans_thresh: Translation since last keyframe (trajectory world
            units — arbitrary, scene-dependent; inspect the trajectory
            X-Z plot to calibrate).
        rot_thresh_deg: Rotation since last keyframe, in degrees.
        min_matches: New keyframe when keyframe-vs-frame good matches
            drop below this (overlap lost).
        max_interval: Hard cap on frames between keyframes.
    """

    trans_thresh: float = 2.0
    rot_thresh_deg: float = 10.0
    min_matches: int = 100
    max_interval: int = 10


def _relative_motion(R_kf: np.ndarray, C_kf: np.ndarray,
                     R_f: np.ndarray, C_f: np.ndarray) -> Tuple[float, float]:
    """(translation distance, rotation angle in degrees) keyframe -> frame."""
    trans = float(np.linalg.norm(C_f - C_kf))
    cos_angle = (np.trace(R_f @ R_kf.T) - 1.0) / 2.0
    rot_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    return trans, rot_deg


class KeyframeSelectionStage(Stage):
    """Stage 9 — select keyframes; save metadata + trajectory markers.

    Loads `_trajectory.npz` (poses), `_orb.npz` (descriptors) and
    `_orb_matches.npz` (Lowe ratio only) — all read-only. Records on
    `ctx.shared["keyframes"]`:

        shared["keyframes"] = {
            "ids": [...], "reasons": [...], "stats": {...},
            "keyframes_path": ..., "plot_path": ...,
        }
    """

    name = "keyframe_selection"

    def __init__(self, config: Optional[KeyframeConfig] = None):
        self.config = config or KeyframeConfig()

    # ------------------------------------------------------------------ #
    # stage entry point
    # ------------------------------------------------------------------ #
    def process(self, ctx: VideoContext) -> VideoContext:
        video_path = ctx.output_path
        orb_path = sibling_output(video_path, "_orb.npz")
        matches_path = sibling_output(video_path, "_orb_matches.npz")
        traj_path = sibling_output(video_path, "_trajectory.npz")
        for needed, stage in ((orb_path, "ORB"), (matches_path, "matching"),
                              (traj_path, "trajectory")):
            if not needed.exists():
                raise IOError(f"{stage} data missing for keyframes (run it first): {needed}")

        _, desc_arrays, _ = _load_frame_data(orb_path)
        with np.load(str(matches_path), allow_pickle=False) as z:
            ratio_thresh = float(z["ratio_thresh"])
        with np.load(str(traj_path), allow_pickle=False) as z:
            rotations = [np.array(m, dtype=np.float64) for m in z["rotations"]]
            centers = [np.array(p, dtype=np.float64) for p in z["positions"]]
            segments = [int(s) for s in z["segment_ids"]]

        selection = self._select(desc_arrays, rotations, centers, segments, ratio_thresh)

        keyframes_path = sibling_output(video_path, "_keyframes.npz")
        self._save_keyframes(keyframes_path, selection)
        visuals: Dict[str, str] = {}
        visuals["plot_path"] = str(self._save_plot(
            video_path, centers, segments, selection["ids"]))

        self._report(ctx, keyframes_path, visuals, selection)
        return ctx

    # ------------------------------------------------------------------ #
    # selection
    # ------------------------------------------------------------------ #
    def _select(self, desc_arrays: List[np.ndarray], rotations: List[np.ndarray],
                centers: List[np.ndarray], segments: List[int],
                ratio_thresh: float) -> Dict:
        cfg = self.config
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        n_frames = len(desc_arrays)

        ids = [0]
        reasons = [[REASON_FIRST]]
        trans_list = [0.0]
        rot_list = [0.0]
        match_list = [-1]  # frame 0 has no predecessor keyframe to match
        for f in range(1, n_frames):
            kf = ids[-1]
            fired: List[str] = []
            if segments[f] != segments[kf]:
                fired.append(REASON_SEGMENT_BREAK)
                trans, rot = float("nan"), float("nan")
            else:
                trans, rot = _relative_motion(
                    rotations[kf], centers[kf], rotations[f], centers[f])
                if trans >= cfg.trans_thresh:
                    fired.append(REASON_TRANSLATION)
                if rot >= cfg.rot_thresh_deg:
                    fired.append(REASON_ROTATION)
            if f - kf >= cfg.max_interval:
                fired.append(REASON_INTERVAL)
            nmatch = len(_good_matches(
                desc_arrays[kf], desc_arrays[f], matcher, ratio_thresh))
            if nmatch < cfg.min_matches:
                fired.append(REASON_LOW_MATCHES)
            if fired:
                ids.append(f)
                reasons.append(fired)
                trans_list.append(trans)
                rot_list.append(rot)
                match_list.append(nmatch)
        gaps = np.diff(np.array(ids, dtype=np.int32))
        return {"ids": ids, "reasons": reasons, "trans": trans_list,
                "rot": rot_list, "matches": match_list, "gaps": gaps,
                "total_frames": n_frames}

    # ------------------------------------------------------------------ #
    # persistence (.npz)
    # ------------------------------------------------------------------ #
    def _save_keyframes(self, path: Path, selection: Dict) -> None:
        cfg = self.config
        gaps = np.array(selection["gaps"], dtype=np.int32)
        np.savez_compressed(
            path,
            keyframe_ids=np.array(selection["ids"], dtype=np.int32),
            reasons=np.array(["+".join(r) for r in selection["reasons"]]),
            trans_since=np.array(selection["trans"], dtype=np.float64),  # NaN across breaks
            rot_since_deg=np.array(selection["rot"], dtype=np.float64),  # NaN across breaks
            matches_vs_last=np.array(selection["matches"], dtype=np.int32),
            keyframe_gaps=gaps,
            total_frames=np.array(selection["total_frames"]),
            trans_thresh=np.array(cfg.trans_thresh),
            rot_thresh_deg=np.array(cfg.rot_thresh_deg),
            min_matches=np.array(cfg.min_matches),
            max_interval=np.array(cfg.max_interval),
        )

    # ------------------------------------------------------------------ #
    # visualization (trajectory with keyframes marked)
    # ------------------------------------------------------------------ #
    def _save_plot(self, video_path: Path, centers: List[np.ndarray],
                   segments: List[int], kf_ids: List[int]) -> Path:
        path = sibling_output(video_path, "_keyframes_plot.png")
        W, H, M = 1200, 700, 70
        img = np.full((H, W, 3), 24, dtype=np.uint8)
        positions = np.array(centers)
        span = float(np.abs(positions[:, [0, 2]]).max(initial=0.0)) or 1.0
        scale = min(W - 2 * M, H - 2 * M) / (2 * span * 1.1)
        cx, cy = W // 2, H // 2

        def located(p: np.ndarray) -> Tuple[int, int]:
            return (round(cx + p[0] * scale), round(cy - p[2] * scale))

        segments_arr = np.array(segments)
        for seg in range(int(segments_arr.max()) + 1):
            idx = np.where(segments_arr == seg)[0]
            if len(idx) > 1:
                pts = np.array([located(positions[i]) for i in idx], dtype=np.int32)
                cv2.polylines(img, [pts], False, (90, 90, 90), 1)
        kf_set = set(kf_ids)
        for i, p in enumerate(positions):
            if i in kf_set:
                cv2.circle(img, located(p), 9, (0, 255, 255), 2)
                cv2.putText(img, str(i), (located(p)[0] + 11, located(p)[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            else:
                cv2.circle(img, located(p), 2, (110, 110, 110), -1)
        cv2.putText(img,
                    f"Keyframes on trajectory: {len(kf_ids)}/{len(positions)} frames "
                    f"(ARBITRARY UNITS - NOT meters)", (M, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(path), img)
        return path

    # ------------------------------------------------------------------ #
    # terminal report
    # ------------------------------------------------------------------ #
    def _report(self, ctx: VideoContext, keyframes_path: Path,
                visuals: Dict[str, str], selection: Dict) -> None:
        ids = selection["ids"]
        gaps = [int(g) for g in selection["gaps"]]
        reason_hist: Dict[str, int] = {}
        for fired in selection["reasons"]:
            for reason in fired:
                reason_hist[reason] = reason_hist.get(reason, 0) + 1
        stats = {
            "total_keyframes": len(ids),
            "total_frames": selection["total_frames"],
            "gap_min": min(gaps) if gaps else 0,
            "gap_mean": round(float(np.mean(gaps)), 1) if gaps else 0.0,
            "gap_max": max(gaps) if gaps else 0,
            "reason_hist": reason_hist,
        }
        ctx.shared["keyframes"] = {
            "ids": ids,
            "reasons": selection["reasons"],
            "stats": stats,
            "keyframes_path": str(keyframes_path),
            **visuals,
        }
        line = "-" * 52
        print(f"\n{line}\nKEYFRAME SELECTION\n{line}")
        print(f"  Keyframes       : {len(ids)} / {selection['total_frames']} frames")
        print(f"  IDs             : {ids}")
        print(f"  Gap min/mean/max     : {stats['gap_min']} / "
              f"{stats['gap_mean']} / {stats['gap_max']} (cap: {self.config.max_interval})")
        print(f"  Reasons         : {reason_hist}")
        for i, frame in enumerate(ids):
            trans = selection["trans"][i]
            rot = selection["rot"][i]
            trans_str = f"{trans:.2f}" if np.isfinite(trans) else "n/a"
            rot_str = f"{rot:.1f}deg" if np.isfinite(rot) else "n/a"
            print(f"    kf frame {frame}: {'+'.join(selection['reasons'][i])} "
                  f"(trans={trans_str}, rot={rot_str}, "
                  f"matches={selection['matches'][i]})")
        print(f"  Keyframes saved : {keyframes_path}")
        for label, vis_path in visuals.items():
            print(f"  {label:<15}: {vis_path}")
        print(line)
