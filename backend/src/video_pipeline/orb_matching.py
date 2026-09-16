"""Descriptor matching stage: BFMatcher + Lowe's ratio test, consecutive frames.

Takes the ORB features of the normalized video and matches each frame's
descriptors against the next frame's. Two deliberate choices:

* Hamming distance, because ORB descriptors are binary strings — a
  Euclidean distance on bits is meaningless, so the metric follows from
  the descriptor type rather than from tuning.
* Lowe's ratio test (nearest / second-nearest < threshold) instead of an
  absolute distance cutoff, which would need retuning for every scene.
  Ambiguous matches are rejected relative to their runner-up.

Pairs with fewer than `min_good_matches` are *reported* as weak, never
hidden — a broken track is a fact downstream stages must see.

Visualization:

* `<stem>_orb_match_montage.png` — drawMatches for first / middle /
  last / weakest pair.
* `<stem>_orb_match_timeline.png` — good matches per pair with the
  reliability threshold marked (same chart style as the ORB step).
* `<stem>_orb_match_vis.mp4` — every consecutive pair side-by-side with
  its good matches drawn.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from video_pipeline.context import VideoContext
from video_pipeline.orb_features import _keypoints_from_array
from video_pipeline.stages import Stage
from video_pipeline.utils import save_line_chart, sibling_output

GREEN = (0, 255, 0)


@dataclass
class MatchConfig:
    """Tuning knobs for descriptor matching.

    Attributes:
        ratio_thresh: Lowe's threshold (Lowe's own value is 0.75; 0.70 here
            is stricter — fewer, cleaner matches).
        min_good_matches: A pair counts as reliable at or above this many
            good matches — enough for geometry estimation with margin.
        max_draw_matches: Cap on drawn matches per visual (strongest first).
        montage_rows: Max pairs shown in the montage image.
        progress_every: Terminal progress cadence in pairs.
        visualize: Master switch for montage/timeline/video outputs
            (the .npz matches are always saved).
    """

    ratio_thresh: float = 0.70
    min_good_matches: int = 10
    max_draw_matches: int = 50
    montage_rows: int = 4
    progress_every: int = 25
    visualize: bool = True


def _load_frame_data(features_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Split the concatenated .npz rows back into per-frame keypoint/descriptor arrays."""
    if not features_path.exists():
        raise IOError(
            f"ORB features missing for matching stage (run the ORB stage first): {features_path}"
        )
    with np.load(str(features_path)) as z:
        counts = z["counts"].astype(int).tolist()
        keypoints_all = z["keypoints"]
        descriptors_all = z["descriptors"]
    kp_arrays, desc_arrays, offset = [], [], 0
    for count in counts:
        kp_arrays.append(keypoints_all[offset:offset + count])
        desc_arrays.append(descriptors_all[offset:offset + count])
        offset += count
    return kp_arrays, desc_arrays, np.array(counts, dtype=np.int32)


def _good_matches(desc_a: np.ndarray, desc_b: np.ndarray,
                  matcher: cv2.BFMatcher, ratio: float) -> List[cv2.DMatch]:
    """knnMatch(k=2) + Lowe's ratio test. Empty input yields empty output."""
    if len(desc_a) == 0 or len(desc_b) == 0:
        return []
    good = []
    for pair in matcher.knnMatch(desc_a, desc_b, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return sorted(good, key=lambda m: m.distance)


def _read_frames(video_path: Path, wanted: set) -> Dict[int, np.ndarray]:
    """Read the video once, keeping only the requested frame indices."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video for match visualization: {video_path}")
    frames: Dict[int, np.ndarray] = {}
    try:
        index = 0
        while len(frames) < len(wanted):
            ok, bgr = cap.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = bgr
            index += 1
    finally:
        cap.release()
    missing = wanted - frames.keys()
    if missing:
        raise IOError(f"Video ended early; frames missing for visualization: {sorted(missing)}")
    return frames


class DescriptorMatchingStage(Stage):
    """Stage 4 — match consecutive frames, report + visualize match quality.

    Records on `ctx.shared["matches"]` for later stages:

        shared["matches"] = {
            "good_counts": [...],       # good matches per pair (i, i+1)
            "reliable": [...],          # bool per pair
            "weak_pairs": [...],
            "stats": {...},
            "matches_path": ..., "montage_path": ...,
            "timeline_path": ..., "vis_video_path": ...,
        }
    """

    name = "descriptor_matching"

    def __init__(self, config: Optional[MatchConfig] = None):
        self.config = config or MatchConfig()

    # ------------------------------------------------------------------ #
    # stage entry point
    # ------------------------------------------------------------------ #
    def process(self, ctx: VideoContext) -> VideoContext:
        video_path = ctx.output_path
        if not video_path.exists():
            raise IOError(f"Normalized video missing for matching stage: {video_path}")
        features_path = sibling_output(video_path, "_orb.npz")
        kp_arrays, desc_arrays, _ = _load_frame_data(features_path)
        if len(kp_arrays) < 2:
            raise IOError(f"Need >= 2 frames with features to match, got {len(kp_arrays)}")

        # Binary descriptors -> Hamming distance (see module docstring).
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        pairs = self._match_all(desc_arrays, matcher)

        matches_path = sibling_output(video_path, "_orb_matches.npz")
        self._save_matches(matches_path, pairs)

        visuals: Dict[str, str] = {}
        if self.config.visualize:
            visuals["montage_path"] = str(self._save_montage(video_path, kp_arrays, pairs))
            visuals["timeline_path"] = str(self._save_timeline(video_path, pairs))
            visuals["vis_video_path"] = str(self._save_vis_video(video_path, kp_arrays, pairs, ctx))

        self._report(ctx, matches_path, visuals, pairs)
        return ctx

    # ------------------------------------------------------------------ #
    # matching
    # ------------------------------------------------------------------ #
    def _match_all(self, desc_arrays: List[np.ndarray],
                   matcher: cv2.BFMatcher) -> List[Dict]:
        """Match every consecutive pair; one dict per pair (i, i+1)."""
        cfg = self.config
        pairs = []
        for i in range(len(desc_arrays) - 1):
            good = _good_matches(desc_arrays[i], desc_arrays[i + 1], matcher, cfg.ratio_thresh)
            pairs.append({
                "pair": (i, i + 1),
                "good": good,
                "good_count": len(good),
                "reliable": len(good) >= cfg.min_good_matches,
                "mean_distance": (round(float(np.mean([m.distance for m in good])), 2)
                                  if good else -1.0),
            })
            if not pairs[-1]["reliable"]:
                print(f"    pair {i}->{i + 1}: {len(good)} good matches "
                      f"(weak, min={cfg.min_good_matches})")
            if (i + 1) % cfg.progress_every == 0:
                print(f"    ... matched {i + 1}/{len(desc_arrays) - 1} pairs", end="\r")
        print()
        return pairs

    # ------------------------------------------------------------------ #
    # persistence (.npz)
    # ------------------------------------------------------------------ #
    def _save_matches(self, path: Path, pairs: List[Dict]) -> None:
        """One row per good match; `owner_pair` maps it back to its pair."""
        query = np.concatenate(
            [np.array([m.queryIdx for m in p["good"]], dtype=np.int32) for p in pairs]
            or [np.zeros(0, np.int32)])
        train = np.concatenate(
            [np.array([m.trainIdx for m in p["good"]], dtype=np.int32) for p in pairs]
            or [np.zeros(0, np.int32)])
        distance = np.concatenate(
            [np.array([m.distance for m in p["good"]], dtype=np.float32) for p in pairs]
            or [np.zeros(0, np.float32)])
        owner_pair = np.repeat(np.arange(len(pairs), dtype=np.int32),
                               [p["good_count"] for p in pairs])
        np.savez_compressed(
            path,
            good_counts=np.array([p["good_count"] for p in pairs], dtype=np.int32),
            reliable=np.array([p["reliable"] for p in pairs], dtype=bool),
            query_idx=query,
            train_idx=train,
            distance=distance,        # Hamming distances of good matches
            owner_pair=owner_pair,
            ratio_thresh=np.array(self.config.ratio_thresh),
            min_good_matches=np.array(self.config.min_good_matches),
        )

    # ------------------------------------------------------------------ #
    # visualization
    # ------------------------------------------------------------------ #
    def _draw_pair(self, img_a: np.ndarray, kp_a: List[cv2.KeyPoint],
                   img_b: np.ndarray, kp_b: List[cv2.KeyPoint],
                   pair: Dict, label: str) -> np.ndarray:
        """Side-by-side pair with strongest matches drawn (no single points)."""
        drawn = cv2.drawMatches(
            img_a, kp_a, img_b, kp_b,
            pair["good"][:self.config.max_draw_matches], None,
            matchColor=GREEN, singlePointColor=None,
            matchesThickness=2,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.putText(drawn, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
        return drawn

    def _pair_keypoints(self, kp_arrays: List[np.ndarray], i: int) -> Tuple[List, List]:
        return (_keypoints_from_array(kp_arrays[i]),
                _keypoints_from_array(kp_arrays[i + 1]))

    def _montage_pair_indices(self, pairs: List[Dict]) -> List[int]:
        """First / middle / last / weakest pair, deduplicated, in order."""
        n = len(pairs)
        weakest = min(range(n), key=lambda i: pairs[i]["good_count"])
        return sorted(set([0, n // 2, n - 1, weakest]))[:self.config.montage_rows]

    def _save_montage(self, video_path: Path,
                      kp_arrays: List[np.ndarray], pairs: List[Dict]) -> Path:
        path = sibling_output(video_path, "_orb_match_montage.png")
        chosen = self._montage_pair_indices(pairs)
        wanted = {i for p in chosen for i in pairs[p]["pair"]}
        frames = _read_frames(video_path, wanted)
        rows = []
        for p in chosen:
            i, j = pairs[p]["pair"]
            kp_a, kp_b = self._pair_keypoints(kp_arrays, i)
            rows.append(self._draw_pair(
                frames[i], kp_a, frames[j], kp_b, pairs[p],
                f"pair {i}->{j}: {pairs[p]['good_count']} good "
                f"(mean Hamming {pairs[p]['mean_distance']})"))
        cv2.imwrite(str(path), np.vstack(rows))
        return path

    def _save_timeline(self, video_path: Path, pairs: List[Dict]) -> Path:
        path = sibling_output(video_path, "_orb_match_timeline.png")
        good_counts = [p["good_count"] for p in pairs]
        weak = [i for i, p in enumerate(pairs) if not p["reliable"]]
        save_line_chart(
            path,
            f"Good matches per pair ({len(pairs)} pairs, {len(weak)} weak)",
            good_counts,
            threshold=self.config.min_good_matches,
            threshold_label=f"min={self.config.min_good_matches}",
            bad_indices=weak,
        )
        return path

    def _save_vis_video(self, video_path: Path, kp_arrays: List[np.ndarray],
                        pairs: List[Dict], ctx: VideoContext) -> Path:
        path = sibling_output(video_path, "_orb_match_vis.mp4")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not re-open video for match visualization: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 ctx.target_fps, (width * 2, height))
        if not writer.isOpened():
            cap.release()
            raise IOError(f"Could not open writer for match visualization: {path}")
        try:
            ok, prev = cap.read()
            if not ok:
                raise IOError("Video ended early during match visualization")
            for i, pair in enumerate(pairs):
                ok, curr = cap.read()
                if not ok:
                    raise IOError(f"Video ended early during match visualization at pair {i}")
                kp_a, kp_b = self._pair_keypoints(kp_arrays, i)
                writer.write(self._draw_pair(
                    prev, kp_a, curr, kp_b, pair,
                    f"pair {i}->{i + 1}: {pair['good_count']} good matches"))
                prev = curr
                if (i + 1) % self.config.progress_every == 0:
                    print(f"    ... visualized {i + 1}/{len(pairs)} pairs", end="\r")
        finally:
            cap.release()
            writer.release()
        print()
        if not path.exists() or path.stat().st_size == 0:
            raise IOError(f"Match visualization produced no output: {path}")
        return path

    # ------------------------------------------------------------------ #
    # terminal report
    # ------------------------------------------------------------------ #
    def _report(self, ctx: VideoContext, matches_path: Path,
                visuals: Dict[str, str], pairs: List[Dict]) -> None:
        good_counts = np.array([p["good_count"] for p in pairs], dtype=np.int32)
        weak = [i for i, p in enumerate(pairs) if not p["reliable"]]
        worst = int(np.argmin(good_counts))
        stats = {
            "pairs": len(pairs),
            "min": int(good_counts.min()),
            "mean": round(float(good_counts.mean()), 1),
            "max": int(good_counts.max()),
            "reliable_pairs": int(sum(p["reliable"] for p in pairs)),
            "weak_pairs": weak,
            "worst_pair": (worst, worst + 1),
        }
        ctx.shared["matches"] = {
            "good_counts": good_counts.tolist(),
            "reliable": [p["reliable"] for p in pairs],
            "weak_pairs": weak,
            "stats": stats,
            "matches_path": str(matches_path),
            **visuals,
        }
        line = "-" * 52
        print(f"\n{line}\nDESCRIPTOR MATCHING (BFMatcher + Lowe ratio "
              f"{self.config.ratio_thresh:g})\n{line}")
        print(f"  Pairs           : {stats['pairs']}")
        print(f"  Good min/mean/max    : {stats['min']} / {stats['mean']} / {stats['max']}")
        print(f"  Reliable pairs  : {stats['reliable_pairs']}/{stats['pairs']} "
              f"(min_good_matches={self.config.min_good_matches})")
        print(f"  Worst pair      : {stats['worst_pair'][0]}->{stats['worst_pair'][1]} "
              f"({stats['min']} good)")
        if weak:
            print(f"  !! Weak pairs (below minimum): {weak}")
        else:
            print("  Every pair matched reliably.")
        print(f"  Matches saved   : {matches_path}")
        for label, vis_path in visuals.items():
            print(f"  {label:<15}: {vis_path}")
        print(line)
