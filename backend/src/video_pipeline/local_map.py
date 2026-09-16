"""Local mapping stage: keyframes + landmarks + their associations.

Consumes (read-only) the filtered cloud (Phase 7), keyframes (Phase 8),
trajectory poses (Phase 5) and ORB features — no new triangulation, no
optimization. Association rule, stated exactly:

* A filtered landmark triangulated from pair (i, i+1) was, by
  construction, observed by frames i and i+1. Its observing keyframes
  are therefore {i, i+1} intersected with the keyframe set — a fact read
  off the stored pair id, not estimated.
* Landmarks whose pair touches no keyframe are ORPHANS: kept in the
  collection with an empty observing list (no data loss, full
  transparency), but part of no keyframe's landmark set.
* Association is frame-granular by necessity: keypoint-level links would
  need per-point match-index provenance that the stored clouds do not
  carry. Re-deriving it by deterministic replay would couple this stage
  to exact code/threshold/float reproducibility of an earlier phase —
  fragile, so deliberately not done.

Topology note: landmarks come from consecutive pairs, so each is seen by
at most TWO keyframes — a chain, not a densely cross-observed map. True
multi-view constraints need keyframe-pair triangulation, which is a later
phase, not a patch applied here.

Storage uses flat arrays + CSR-style offsets (no pickles):

* keyframe `k` (in `keyframe_ids` order) owns
  `kf_landmark_ids[offsets[k]:offsets[k+1]]` and
  `kf_keypoints/kf_descriptors[feat_offsets[k]:feat_offsets[k+1]]`.
* landmark `m` (= row `m` of the filtered cloud, so IDs stay traceable)
  is observed by `lm_kf_ids[lm_offsets[m]:lm_offsets[m+1]]`
  (indices into `keyframe_ids`, i.e. keyframe ORDINALS, not frame ids).

UNITS — arbitrary, never meters (propagated from the motion stage).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from video_pipeline.context import VideoContext
from video_pipeline.stages import Stage
from video_pipeline.utils import sibling_output

#: Stamped on terminal output, the .npz, and shared context.
MAP_SCALE_NOTE = (
    "Map geometry is in ARBITRARY UNITS, not meters: filtering and mapping "
    "change membership and organization, never scale. Not for metric use."
)


@dataclass
class LocalMapConfig:
    """Tuning knobs for local mapping.

    Attributes:
        visualize: Master switch for the map plot output
            (the .npz map is always saved).
    """

    visualize: bool = True


def _csr(lengths: List[int]) -> Tuple[np.ndarray, int]:
    """Offsets array (len+1) for a CSR layout; total rows returned too."""
    offsets = np.cumsum([0] + lengths).astype(np.int32)
    return offsets, int(offsets[-1])


class LocalMappingStage(Stage):
    """Stage 10 — assemble keyframes + landmarks into a local map.

    Records on `ctx.shared["local_map"]`:

        shared["local_map"] = {
            "n_keyframes": ..., "n_landmarks": ..., "n_mapped": ...,
            "landmarks_per_kf": [...], "kf_per_landmark_hist": {...},
            "stats": {...}, "scale_note": ...,
            "map_path": ..., "plot_path": ...,
        }
    """

    name = "local_mapping"

    def __init__(self, config: Optional[LocalMapConfig] = None):
        self.config = config or LocalMapConfig()

    # ------------------------------------------------------------------ #
    # stage entry point
    # ------------------------------------------------------------------ #
    def process(self, ctx: VideoContext) -> VideoContext:
        video_path = ctx.output_path
        inputs = {
            "filtered": sibling_output(video_path, "_points3d_filtered.npz"),
            "keyframes": sibling_output(video_path, "_keyframes.npz"),
            "trajectory": sibling_output(video_path, "_trajectory.npz"),
            "orb": sibling_output(video_path, "_orb.npz"),
        }
        for name, path in inputs.items():
            if not path.exists():
                raise IOError(f"{name} data missing for local mapping: {path}")
        with np.load(str(inputs["filtered"]), allow_pickle=False) as z:
            cloud = {k: z[k] for k in z.files}
        with np.load(str(inputs["keyframes"]), allow_pickle=False) as z:
            kf_ids = [int(i) for i in z["keyframe_ids"]]
        with np.load(str(inputs["trajectory"]), allow_pickle=False) as z:
            rotations = [np.array(m, dtype=np.float64) for m in z["rotations"]]
            centers = [np.array(p, dtype=np.float64) for p in z["positions"]]
        with np.load(str(inputs["orb"]), allow_pickle=False) as z:
            orb_counts = z["counts"].astype(int).tolist()
            orb_kp, orb_desc = z["keypoints"], z["descriptors"]

        landmark_obs = self._associate(cloud, kf_ids)
        kf_sets = self._keyframe_landmarks(landmark_obs, kf_ids)

        map_path = sibling_output(video_path, "_local_map.npz")
        self._save_map(map_path, cloud, kf_ids, rotations, centers,
                       orb_counts, orb_kp, orb_desc, landmark_obs, kf_sets)
        visuals: Dict[str, str] = {}
        if self.config.visualize:
            visuals["plot_path"] = str(self._save_plot(
                video_path, cloud, kf_ids, centers, landmark_obs))

        self._report(ctx, map_path, visuals, cloud, kf_ids, landmark_obs, kf_sets)
        return ctx

    # ------------------------------------------------------------------ #
    # association (the whole point of the stage)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _associate(cloud: Dict, kf_ids: List[int]) -> List[List[int]]:
        """Observing-keyframe ORDINALS per landmark (empty = orphan).

        Landmark m triangulated from pair p was observed by frames {p, p+1};
        its observers are those frames that are keyframes. Ordinals (index
        into `kf_ids`), not frame ids, so the map stays compact.
        """
        kf_ordinal = {frame: ordinal for ordinal, frame in enumerate(kf_ids)}
        pairs = np.array(cloud["pair"]).astype(int)
        obs: List[List[int]] = []
        for p in pairs:
            observers = sorted({kf_ordinal[f] for f in (int(p), int(p) + 1)
                                if f in kf_ordinal})
            obs.append(observers)
        return obs

    @staticmethod
    def _keyframe_landmarks(landmark_obs: List[List[int]],
                            kf_ids: List[int]) -> List[List[int]]:
        """Inverse map: sorted landmark IDs per keyframe ordinal."""
        kf_sets: List[List[int]] = [[] for _ in kf_ids]
        for landmark, observers in enumerate(landmark_obs):
            for ordinal in observers:
                kf_sets[ordinal].append(landmark)
        return [sorted(s) for s in kf_sets]

    # ------------------------------------------------------------------ #
    # persistence (.npz, CSR layout, no pickles)
    # ------------------------------------------------------------------ #
    def _save_map(self, path: Path, cloud: Dict, kf_ids: List[int],
                  rotations: List[np.ndarray], centers: List[np.ndarray],
                  orb_counts: List[int], orb_kp: np.ndarray, orb_desc: np.ndarray,
                  landmark_obs: List[List[int]],
                  kf_sets: List[List[int]]) -> None:
        n_landmarks = len(landmark_obs)

        lm_lengths = [len(o) for o in landmark_obs]
        lm_offsets, _ = _csr(lm_lengths)
        lm_kf_ids = np.concatenate(
            [np.array(o, dtype=np.int32) for o in landmark_obs]
            or [np.zeros(0, np.int32)])

        kf_lengths = [len(s) for s in kf_sets]
        kf_offsets, _ = _csr(kf_lengths)
        kf_landmark_ids = np.concatenate(
            [np.array(s, dtype=np.int32) for s in kf_sets]
            or [np.zeros(0, np.int32)])

        orb_offsets = np.cumsum([0] + orb_counts).astype(np.int32)
        kf_feat_lengths = [orb_counts[frame] for frame in kf_ids]
        kf_feat_offsets = np.cumsum([0] + kf_feat_lengths).astype(np.int32)
        # Packed keyframe-only rows: offsets index THESE arrays, whose layout
        # (concatenated kf frames) differs from the full per-frame orb arrays.
        kf_keypoints = np.concatenate(
            [orb_kp[orb_offsets[frame]:orb_offsets[frame] + orb_counts[frame]]
             for frame in kf_ids] or [np.zeros((0, 6), np.float32)])
        kf_descriptors = np.concatenate(
            [orb_desc[orb_offsets[frame]:orb_offsets[frame] + orb_counts[frame]]
             for frame in kf_ids] or [np.zeros((0, 32), np.uint8)])
        np.savez_compressed(
            path,
            # keyframes
            keyframe_ids=np.array(kf_ids, dtype=np.int32),
            keyframe_R=np.array([rotations[f] for f in kf_ids]),
            keyframe_pos=np.array([centers[f] for f in kf_ids]),
            kf_feat_offsets=np.array(kf_feat_offsets, dtype=np.int32),
            kf_keypoints=kf_keypoints,              # (Kt, 6) sliced by kf_feat_offsets
            kf_descriptors=kf_descriptors,          # (Kt, 32) sliced likewise
            kf_landmark_offsets=kf_offsets,
            kf_landmark_ids=kf_landmark_ids,        # landmark IDs per keyframe
            # landmarks (ID = row in the filtered cloud)
            landmark_pos=np.array(cloud["points"], dtype=np.float64),
            landmark_colors=np.array(cloud["colors"], dtype=np.uint8),
            landmark_reproj=np.array(cloud["reproj_error"], dtype=np.float32),
            landmark_pair=np.array(cloud["pair"], dtype=np.int32),
            landmark_segment=np.array(cloud["segment"], dtype=np.int32),
            lm_kf_offsets=lm_offsets,
            lm_kf_ids=lm_kf_ids,                    # keyframe ORDINALS per landmark
            convention=np.array(
                "landmark ID = row in the filtered cloud; lm_kf_ids holds "
                "keyframe ordinals (index into keyframe_ids); keyframe_R maps "
                "world to camera frame; keyframe_pos are camera centers"),
            scale_note=np.array(MAP_SCALE_NOTE),
        )

    # ------------------------------------------------------------------ #
    # visualization (X-Z map: landmarks by support + keyframe markers)
    # ------------------------------------------------------------------ #
    def _save_plot(self, video_path: Path, cloud: Dict, kf_ids: List[int],
                   centers: List[np.ndarray],
                   landmark_obs: List[List[int]]) -> Path:
        path = sibling_output(video_path, "_local_map_plot.png")
        pts = np.array(cloud["points"], dtype=np.float64)
        support = np.array([len(o) for o in landmark_obs])
        W, H, M = 1200, 700, 70
        img = np.full((H, W, 3), 24, dtype=np.uint8)
        span = float(np.abs(pts[:, [0, 2]]).max(initial=0.0)) or 1.0
        scale = min(W - 2 * M, H - 2 * M) / (2 * span * 1.1)
        cx, cy = W // 2, H // 2

        def located(p: np.ndarray) -> Tuple[int, int]:
            return (round(cx + p[0] * scale), round(cy - p[2] * scale))

        palette = {0: (90, 90, 90), 1: (0, 255, 0), 2: (0, 255, 255)}
        for level in (0, 1, 2):  # orphans first, best-supported drawn over
            sel = np.where(support == level)[0]
            for m in sel[::max(1, len(sel) // 15000)]:
                cv2.circle(img, located(pts[m]), 2, palette[level], -1)
        for frame in kf_ids:
            at = located(np.array(centers[frame]))
            cv2.drawMarker(img, at, (0, 255, 255), cv2.MARKER_SQUARE, 14, 2)
            cv2.putText(img, str(frame), (at[0] + 10, at[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        n_mapped = int((support > 0).sum())
        cv2.putText(img,
                    f"Local map X-Z: {len(kf_ids)} keyframes, {len(pts)} landmarks "
                    f"({n_mapped} mapped, {len(pts) - n_mapped} orphan) "
                    f"(ARBITRARY UNITS - NOT meters)", (M, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imwrite(str(path), img)
        return path

    # ------------------------------------------------------------------ #
    # terminal report
    # ------------------------------------------------------------------ #
    def _report(self, ctx: VideoContext, map_path: Path,
                visuals: Dict[str, str], cloud: Dict, kf_ids: List[int],
                landmark_obs: List[List[int]],
                kf_sets: List[List[int]]) -> None:
        per_kf = [len(s) for s in kf_sets]
        per_lm = [len(o) for o in landmark_obs]
        hist = {k: sum(1 for c in per_lm if c == k) for k in (0, 1, 2)}
        stats = {
            "n_keyframes": len(kf_ids),
            "n_landmarks": len(landmark_obs),
            "n_mapped": sum(1 for c in per_lm if c > 0),
            "n_orphans": hist[0],
            "landmarks_per_kf": per_kf,
            "kf_per_landmark_hist": hist,
        }
        ctx.shared["local_map"] = {
            "n_keyframes": stats["n_keyframes"],
            "n_landmarks": stats["n_landmarks"],
            "n_mapped": stats["n_mapped"],
            "landmarks_per_kf": per_kf,
            "kf_per_landmark_hist": hist,
            "stats": stats,
            "scale_note": MAP_SCALE_NOTE,
            "map_path": str(map_path),
            **visuals,
        }
        line = "-" * 52
        print(f"\n{line}\nLOCAL MAPPING (keyframes + landmarks)\n{line}")
        print(f"  Keyframes       : {stats['n_keyframes']}")
        print(f"  Landmarks       : {stats['n_landmarks']} "
              f"({stats['n_mapped']} mapped, {stats['n_orphans']} orphan)")
        print(f"  Landmarks/keyframe min/mean/max: {min(per_kf)} / "
              f"{sum(per_kf) / max(len(per_kf), 1):.1f} / {max(per_kf)}")
        print(f"  Keyframes/landmark: {hist} (chain topology: <= 2 by construction)")
        print(f"  Landmarks per keyframe (frame: count):")
        print("   " + ", ".join(f"{f}:{c}" for f, c in zip(kf_ids, per_kf)))
        print(f"  NOTE: {MAP_SCALE_NOTE}")
        print(f"  Map saved       : {map_path}")
        for label, vis_path in visuals.items():
            print(f"  {label:<15}: {vis_path}")
        print(line)
