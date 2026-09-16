"""ORB keypoint + descriptor extraction stage.

Runs on the normalized video and records, for every frame, its ORB
keypoints and binary descriptors for downstream stages (matching,
tracking, ...).

Reliability strategy — why frames end up with no features, and what we
do about each cause instead of papering over it:

* Low local contrast starves the FAST corner detector. Fix: retry the
  frame with CLAHE contrast enhancement (addresses the cause; the pixels
  themselves gain texture).
* Sparse texture even after enhancement. Fix: retry once more with a
  more sensitive detector (more features, lower FAST threshold).
* Truly textureless input (e.g. black frames). No detector setting can
  invent corners there, so the frame is *reported* as unreliable rather
  than faked with dummy keypoints. Silent padding would corrupt every
  downstream consumer.

Visualization (all OpenCV/numpy only, no extra dependencies):

* `<stem>_orb_preview.jpg` — first frame vs weakest frame, keypoints drawn.
* `<stem>_orb_vis.mp4` — the normalized video with keypoints + counts
  overlaid on every frame (watch reliability over time).
* `<stem>_orb_desc.png` — mosaic of raw descriptor bytes (8x4 per
  descriptor): identical rows would expose a degenerate descriptor set.
* `<stem>_orb_timeline.png` — keypoint count per frame with the
  reliability threshold and unreliable frames marked.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from video_pipeline.context import VideoContext
from video_pipeline.stages import Stage
from video_pipeline.utils import save_line_chart, sibling_output

#: Tiers in escalation order. The first tier reaching `min_keypoints` wins.
TIER_PLAIN = "plain"
TIER_CLAHE = "clahe"
TIER_SENSITIVE = "sensitive"

GREEN = (0, 255, 0)
RED = (0, 0, 255)
GRAY = (180, 180, 180)


@dataclass
class OrbConfig:
    """Tuning knobs for ORB extraction.

    Attributes:
        nfeatures: Keypoint budget per frame for the base detector.
        scale_factor / nlevels: Image-pyramid shape (defaults are ORB's
            well-tested values; change only with reason).
        fast_threshold: FAST corner threshold for the base detector.
        fallback_fast_threshold: Lower threshold for the last-resort
            detector (must stay >= 2, an OpenCV requirement).
        min_keypoints: A frame counts as reliable at or above this many
            keypoints — enough for RANSAC geometry with redundancy left.
        clahe_clip_limit: Contrast-enhancement strength for fallback tiers.
        progress_every: Terminal progress cadence in frames.
        visualize: Master switch for the preview/video/mosaic/timeline
            outputs (the .npz features are always saved).
    """

    nfeatures: int = 1000
    scale_factor: float = 1.2
    nlevels: int = 8
    fast_threshold: int = 20
    fallback_fast_threshold: int = 7
    min_keypoints: int = 30
    clahe_clip_limit: float = 3.0
    progress_every: int = 25
    visualize: bool = True


def _make_detector(config: OrbConfig, nfeatures: int, fast_threshold: int) -> cv2.ORB:
    return cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=config.scale_factor,
        nlevels=config.nlevels,
        fastThreshold=fast_threshold,
    )


def _detect(gray: np.ndarray, detector: cv2.ORB) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Detect; normalize the no-keypoint case to empty (not None)."""
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if not keypoints:
        return [], np.zeros((0, 32), dtype=np.uint8)
    return list(keypoints), descriptors


def _extract_reliable(
    gray: np.ndarray,
    config: OrbConfig,
    base: cv2.ORB,
    sensitive: cv2.ORB,
    clahe: cv2.CLAHE,
) -> Tuple[List[cv2.KeyPoint], np.ndarray, str]:
    """Run the tier ladder; return (keypoints, descriptors, tier used).

    The first tier reaching `min_keypoints` wins. If none does, the best
    tier's output is returned — the caller flags the frame unreliable.
    """
    enhanced = clahe.apply(gray)
    attempts = (
        (TIER_PLAIN, gray, base),
        (TIER_CLAHE, enhanced, base),
        (TIER_SENSITIVE, enhanced, sensitive),
    )
    best: Tuple[List[cv2.KeyPoint], np.ndarray, str] = ([], np.zeros((0, 32), dtype=np.uint8), TIER_PLAIN)
    for tier, image, detector in attempts:
        keypoints, descriptors = _detect(image, detector)
        if len(keypoints) >= config.min_keypoints:
            return keypoints, descriptors, tier
        if len(keypoints) > len(best[0]):
            best = (keypoints, descriptors, tier)
    return best


def _keypoints_to_array(keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    """Serialize keypoints to an (N, 6) float32 array: x, y, size, angle, response, octave."""
    if not keypoints:
        return np.zeros((0, 6), dtype=np.float32)
    return np.array(
        [(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave) for k in keypoints],
        dtype=np.float32,
    )


def _keypoints_from_array(arr: np.ndarray) -> List[cv2.KeyPoint]:
    return [cv2.KeyPoint(float(x), float(y), float(size), float(angle), float(response), int(octave))
            for x, y, size, angle, response, octave in arr]


def _overlay_keypoints(bgr: np.ndarray, kp_arr: np.ndarray, label: str) -> np.ndarray:
    """Draw keypoints + a caption line; the shared look of every visual."""
    out = cv2.drawKeypoints(bgr, _keypoints_from_array(kp_arr), None, color=GREEN, flags=0)
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
    return out


class OrbFeatureStage(Stage):
    """Stage 3 — ORB keypoints + descriptors for every normalized frame.

    Reads `ctx.output_path` (the normalized video), extracts features
    frame by frame, saves them to `<output_stem>_orb.npz` plus
    visualizations, and records everything on `ctx.shared["orb"]` for
    later stages:

        shared["orb"] = {
            "counts": [...],        # keypoints per frame
            "tiers": [...],         # ladder tier used per frame
            "reliable": [...],      # bool per frame
            "unreliable_frames": [...],
            "stats": {"min/mean/max", ...},
            "features_path": ..., "preview_path": ...,
            "vis_video_path": ..., "desc_mosaic_path": ..., "timeline_path": ...,
        }
    """

    name = "orb_features"

    def __init__(self, config: Optional[OrbConfig] = None):
        self.config = config or OrbConfig()

    # ------------------------------------------------------------------ #
    # stage entry point
    # ------------------------------------------------------------------ #
    def process(self, ctx: VideoContext) -> VideoContext:
        video_path = ctx.output_path
        if not video_path.exists():
            raise IOError(
                f"Normalized video missing for ORB stage (run resampling first): {video_path}"
            )
        records = self._extract_all(video_path)
        if not records["counts"]:
            raise IOError(f"Video contained no decodable frames: {video_path}")

        features_path = sibling_output(video_path, "_orb.npz")
        self._save_features(features_path, records, ctx)

        visuals: Dict[str, str] = {}
        if self.config.visualize:
            visuals["preview_path"] = str(self._save_preview(video_path, records))
            visuals["vis_video_path"] = str(self._save_vis_video(video_path, records, ctx))
            visuals["desc_mosaic_path"] = str(self._save_descriptor_mosaic(video_path, records))
            visuals["timeline_path"] = str(self._save_timeline(video_path, records))

        self._report(ctx, features_path, visuals, records)
        return ctx

    # ------------------------------------------------------------------ #
    # extraction (streaming: constant memory)
    # ------------------------------------------------------------------ #
    def _extract_all(self, video_path: Path) -> Dict[str, list]:
        """Detect on every frame; keep only arrays + two thumbnails in memory."""
        cfg = self.config
        base = _make_detector(cfg, cfg.nfeatures, cfg.fast_threshold)
        sensitive = _make_detector(cfg, cfg.nfeatures * 2, cfg.fallback_fast_threshold)
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=(8, 8))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video for ORB stage: {video_path}")

        kp_arrays: List[np.ndarray] = []
        desc_arrays: List[np.ndarray] = []
        tiers: List[str] = []
        reliable: List[bool] = []
        first_bgr: Optional[np.ndarray] = None
        weakest_bgr: Optional[np.ndarray] = None
        weakest_kp = np.zeros((0, 6), dtype=np.float32)
        weakest_count = -1
        try:
            index = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors, tier = _extract_reliable(gray, cfg, base, sensitive, clahe)
                count = len(keypoints)
                is_reliable = count >= cfg.min_keypoints
                kp_arr = _keypoints_to_array(keypoints)

                kp_arrays.append(kp_arr)
                desc_arrays.append(descriptors)
                tiers.append(tier)
                reliable.append(is_reliable)

                if first_bgr is None:
                    first_bgr = bgr.copy()
                if weakest_bgr is None or count < weakest_count:
                    weakest_bgr, weakest_kp, weakest_count = bgr.copy(), kp_arr, count

                if tier != TIER_PLAIN or not is_reliable:
                    # Weak frames are the interesting ones — say so loudly.
                    print(f"    frame {index}: {count} keypoints "
                          f"(tier={tier}, reliable={is_reliable})")
                if (index + 1) % cfg.progress_every == 0:
                    print(f"    ... {index + 1} frames processed", end="\r")
                index += 1
        finally:
            cap.release()

        return {
            "counts": [len(a) for a in kp_arrays],
            "kp_arrays": kp_arrays,
            "desc_arrays": desc_arrays,
            "tiers": tiers,
            "reliable": reliable,
            "first_bgr": first_bgr,
            "weakest_bgr": weakest_bgr,
            "weakest_kp": weakest_kp,
            "weakest_count": weakest_count,
        }

    # ------------------------------------------------------------------ #
    # persistence (.npz)
    # ------------------------------------------------------------------ #
    def _save_features(self, path: Path, records: Dict[str, list], ctx: VideoContext) -> None:
        """One row per keypoint; `owner` maps it back to its frame. No pickles."""
        kp_arrays: List[np.ndarray] = records["kp_arrays"]
        desc_arrays: List[np.ndarray] = records["desc_arrays"]
        counts = np.array(records["counts"], dtype=np.int32)
        owner = np.repeat(np.arange(len(kp_arrays), dtype=np.int32), counts)
        keypoints = np.concatenate(kp_arrays, axis=0) if counts.sum() else np.zeros((0, 6), np.float32)
        descriptors = np.concatenate(desc_arrays, axis=0) if counts.sum() else np.zeros((0, 32), np.uint8)
        np.savez_compressed(
            path,
            counts=counts,
            owner=owner,
            keypoints=keypoints,      # (K, 6): x, y, size, angle, response, octave
            descriptors=descriptors,  # (K, 32) uint8
            tier=np.array(records["tiers"]),
            reliable=np.array(records["reliable"], dtype=bool),
            fps=np.array(ctx.target_fps),
            min_keypoints=np.array(self.config.min_keypoints),
            nfeatures=np.array(self.config.nfeatures),
        )

    # ------------------------------------------------------------------ #
    # visualization
    # ------------------------------------------------------------------ #
    def _save_preview(self, video_path: Path, records: Dict[str, list]) -> Path:
        """Side-by-side: first frame vs weakest frame, for eyeball verification."""
        path = sibling_output(video_path, "_orb_preview.jpg")
        left = _overlay_keypoints(records["first_bgr"], records["kp_arrays"][0],
                                  f"first frame ({records['counts'][0]} kp)")
        right = _overlay_keypoints(records["weakest_bgr"], records["weakest_kp"],
                                   f"weakest frame ({records['weakest_count']} kp)")
        cv2.imwrite(str(path), np.hstack([left, right]))
        return path

    def _save_vis_video(self, video_path: Path,
                        records: Dict[str, list], ctx: VideoContext) -> Path:
        """Re-read the video and write it back with keypoints + counts overlaid."""
        path = sibling_output(video_path, "_orb_vis.mp4")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not re-open video for ORB visualization: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 ctx.target_fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise IOError(f"Could not open writer for ORB visualization: {path}")
        try:
            for index, kp_arr in enumerate(records["kp_arrays"]):
                ok, bgr = cap.read()
                if not ok:
                    raise IOError(f"Video ended early during ORB visualization at frame {index}")
                writer.write(_overlay_keypoints(
                    bgr, kp_arr,
                    f"frame {index} ({len(kp_arr)} kp, {records['tiers'][index]})"))
                if (index + 1) % self.config.progress_every == 0:
                    print(f"    ... visualized {index + 1}/{len(records['kp_arrays'])} frames", end="\r")
        finally:
            cap.release()
            writer.release()
        print()
        _require_usable(path)
        return path

    def _save_descriptor_mosaic(self, video_path: Path, records: Dict[str, list]) -> Path:
        """Tile raw descriptor bytes (8x4 each) so a degenerate set is visible.

        Uses the first frame that actually has descriptors; each byte is one
        grey pixel, so identical rows/columns would show up as flat bands.
        """
        path = sibling_output(video_path, "_orb_desc.png")
        frame_idx = next((i for i, c in enumerate(records["counts"]) if c > 0), None)
        canvas = np.full((80, 640, 3), 30, dtype=np.uint8)
        if frame_idx is None:
            cv2.putText(canvas, "no descriptors in any frame", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
            cv2.imwrite(str(path), canvas)
            return path

        descriptors: np.ndarray = records["desc_arrays"][frame_idx]
        take = min(64, len(descriptors))
        sample = descriptors[np.linspace(0, len(descriptors) - 1, take).astype(int)]
        tiles = [cv2.resize(d.reshape(4, 8), (64, 32), interpolation=cv2.INTER_NEAREST)
                 for d in sample]
        cols, pad = 8, 6
        rows = (take + cols - 1) // cols
        canvas = np.full((80 + rows * (32 + pad), cols * (64 + pad) + pad, 3), 30, dtype=np.uint8)
        cv2.putText(canvas, f"frame {frame_idx}: {len(descriptors)} descriptors "
                            f"(showing {take}, 1 px = 1 byte)", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
        for pos, tile in enumerate(tiles):
            r, c = divmod(pos, cols)
            y, x = 80 + r * (32 + pad), pad + c * (64 + pad)
            canvas[y:y + 32, x:x + 64] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(path), canvas)
        return path

    def _save_timeline(self, video_path: Path, records: Dict[str, list]) -> Path:
        """Keypoint count per frame, with the reliability threshold marked."""
        path = sibling_output(video_path, "_orb_timeline.png")
        counts = np.array(records["counts"], dtype=np.int32)
        unreliable = [i for i, ok in enumerate(records["reliable"]) if not ok]
        save_line_chart(
            path,
            f"ORB keypoints per frame ({len(counts)} frames, {len(unreliable)} unreliable)",
            counts,
            threshold=self.config.min_keypoints,
            threshold_label=f"min={self.config.min_keypoints}",
            bad_indices=unreliable,
        )
        return path

    # ------------------------------------------------------------------ #
    # terminal report
    # ------------------------------------------------------------------ #
    def _report(self, ctx: VideoContext, features_path: Path,
                visuals: Dict[str, str], records: Dict[str, list]) -> None:
        counts = np.array(records["counts"], dtype=np.int32)
        unreliable = [i for i, ok in enumerate(records["reliable"]) if not ok]
        tier_use = {t: records["tiers"].count(t) for t in (TIER_PLAIN, TIER_CLAHE, TIER_SENSITIVE)}
        stats = {
            "frames": len(counts),
            "min": int(counts.min()),
            "mean": round(float(counts.mean()), 1),
            "max": int(counts.max()),
            "reliable_frames": int(np.sum(records["reliable"])),
            "unreliable_frames": unreliable,
            "tier_use": tier_use,
        }
        ctx.shared["orb"] = {
            "counts": counts.tolist(),
            "tiers": records["tiers"],
            "reliable": records["reliable"],
            "unreliable_frames": unreliable,
            "stats": stats,
            "features_path": str(features_path),
            **visuals,
        }
        line = "-" * 52
        print(f"\n{line}\nORB FEATURES (per-frame keypoints + descriptors)\n{line}")
        print(f"  Frames          : {stats['frames']}")
        print(f"  Keypoints min/mean/max : {stats['min']} / {stats['mean']} / {stats['max']}")
        print(f"  Reliable frames : {stats['reliable_frames']}/{stats['frames']} "
              f"(min_keypoints={self.config.min_keypoints})")
        print(f"  Tier use        : {tier_use}")
        if unreliable:
            print(f"  !! Unreliable frames (no tier reached the minimum): {unreliable}")
        else:
            print("  Every frame has reliable features.")
        print(f"  Features saved  : {features_path}")
        for label, vis_path in visuals.items():
            print(f"  {label:<15}: {vis_path}")
        print(line)


def _require_usable(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise IOError(f"Visualization produced no output: {path}")
