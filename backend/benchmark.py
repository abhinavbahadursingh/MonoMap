"""SLAM benchmark: wall-time the full flow, verdict vs realtime, baseline compare.

Usage:
    python backend/benchmark.py [--input video.mp4]
    python backend/benchmark.py --save-as output/benchmark_baseline.json
    python backend/benchmark.py --compare output/benchmark_baseline.json [--save-as ...]

Every run executes the complete flow via pipeline.run_slam, prints video
stats + per-phase times + TOTAL + the realtime verdict
(total prospection vs video duration), and writes output/benchmark_latest.json.
--compare additionally prints a BEFORE (baseline file) vs AFTER (this run)
table with per-phase deltas and speedup — the required workflow is:
measure baseline -> change ONE thing -> benchmark --compare -> keep or revert.

The realtime bar is total SLAM wall time <= video duration (factor >= 1).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR / "src"))
sys.path.insert(0, str(BACKEND_DIR))

from pipeline import run_slam  # noqa: E402
from pipeline import SLAM_PHASES  # noqa: E402  (record tuning for comparability)
from main import default_output  # noqa: E402  (same output naming as real runs)
from video_pipeline.context import VideoContext  # noqa: E402
from video_pipeline.stages import probe_video  # noqa: E402

OUTPUT_DIR = BACKEND_DIR / "output"
LATEST_PATH = OUTPUT_DIR / "benchmark_latest.json"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark the full SLAM pipeline")
    p.add_argument("--input", "-i", default=str(BACKEND_DIR / "video1.mp4"),
                   help="Input video (default: backend/video1.mp4)")
    p.add_argument("--save-as", default=None,
                   help="Also snapshot this run as a named baseline JSON")
    p.add_argument("--compare", default=None, metavar="BASELINE_JSON",
                   help="Print BEFORE(baseline) vs AFTER(this run) comparison")
    return p.parse_args(argv)


def run_benchmark(input_path: Path) -> dict:
    meta = probe_video(input_path)
    # Identical output naming to real runs: benchmark measures THE pipeline,
    # not a parallel artifact universe.
    default_out = default_output(input_path, 10.0, 640, 360)
    ctx = VideoContext(input_path=input_path, output_path=default_out,
                       target_fps=10.0, target_width=640, target_height=360)
    wall_start = time.perf_counter()
    result = run_slam(ctx)
    total = time.perf_counter() - wall_start
    n_processed = len(result["context"].shared.get("orb", {}).get("counts", []))
    n_keyframes = len(result["context"].shared.get("keyframes", {}).get("ids", []))
    n_landmarks = result["context"].shared.get("local_map", {}).get("n_landmarks", 0)
    orb_budget = next((s.config.nfeatures for n, s in SLAM_PHASES if n == "features"),
                      None)
    # Lowe ratio read back from the run's own output (single source of truth).
    _matches_path = ctx.output_path.with_name(ctx.output_path.stem + "_orb_matches.npz")
    try:
        with np.load(str(_matches_path), allow_pickle=False) as _z:
            lowe_ratio = float(_z["ratio_thresh"])
    except Exception:
        lowe_ratio = None
    report = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "video": {"path": str(input_path), "duration_sec": round(meta.duration_sec, 3),
                  "frames": meta.frame_count, "fps": round(meta.fps, 3),
                  "width": meta.width, "height": meta.height},
        "processed_frames": n_processed,
        "keyframes": n_keyframes,
        "landmarks": n_landmarks,
        "phase_seconds": {k: round(v, 3) for k, v in result["phase_times"].items()},
        "total_seconds": round(total, 3),
        "realtime_factor": round(meta.duration_sec / total, 3) if total > 0 else 0.0,
        "realtime_ok": bool(total <= meta.duration_sec),
        "config": {"target_fps": 10.0, "target_width": 640, "target_height": 360,
                   "orb_nfeatures": orb_budget, "lowe_ratio": lowe_ratio},
    }
    return report


def print_report(tag: str, rep: dict) -> None:
    v = rep["video"]
    line = "=" * 60
    print(f"\n{line}\nBENCHMARK [{tag}] {rep['timestamp']}\n{line}")
    print(f"  Video           : {Path(v['path']).name} "
          f"({v['duration_sec']}s, {v['frames']} frames, {v['width']}x{v['height']} @ {v['fps']}fps)")
    print(f"  Processed       : {rep['processed_frames']} frames, "
          f"{rep['keyframes']} keyframes, {rep['landmarks']} landmarks")
    for name, seconds in rep["phase_seconds"].items():
        print(f"    {name:<14}: {seconds:7.3f}s")
    print(f"  TOTAL           : {rep['total_seconds']:.3f}s vs video "
          f"{v['duration_sec']:.3f}s (x{rep['realtime_factor']:.2f})")
    print(f"  VERDICT         : {'PASS (realtime)' if rep['realtime_ok'] else 'FAIL (slower than realtime)'}")
    print(line)


def print_comparison(before: dict, after: dict) -> None:
    line = "=" * 60
    print(f"\n{line}\nBEFORE vs AFTER  ({before['timestamp']} -> {after['timestamp']})\n{line}")
    print(f"  {'phase':<14} {'BEFORE':>9} {'AFTER':>9} {'delta':>9}")
    phases = sorted(set(before["phase_seconds"]) | set(after["phase_seconds"]))
    for name in phases:
        b = before["phase_seconds"].get(name, 0.0)
        a = after["phase_seconds"].get(name, 0.0)
        print(f"  {name:<14} {b:>9.3f} {a:>9.3f} {a - b:>+9.3f}")
    b, a = before["total_seconds"], after["total_seconds"]
    print(f"  {'TOTAL':<14} {b:>9.3f} {a:>9.3f} {a - b:>+9.3f}   (x{b / a:.2f} speedup)")
    print(f"  realtime BEFORE : x{before['realtime_factor']:.2f} "
          f"({'PASS' if before['realtime_ok'] else 'FAIL'})")
    print(f"  realtime AFTER  : x{after['realtime_factor']:.2f} "
          f"({'PASS' if after['realtime_ok'] else 'FAIL'})")
    print(line)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = run_benchmark(Path(args.input))
    print_report("current", report)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(report, indent=2))
    print(f"Latest written to: {LATEST_PATH}")
    if args.save_as:
        Path(args.save_as).write_text(json.dumps(report, indent=2))
        print(f"Baseline snapshot: {args.save_as}")
    if args.compare:
        baseline = json.loads(Path(args.compare).read_text())
        print_comparison(baseline, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
