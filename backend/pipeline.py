"""SLAM orchestration layer: the canonical end-to-end flow in one place.

Complete flow:
    Video -> Features -> Matching -> Motion -> Keyframes -> Triangulation
      -> Local Map -> Optimization -> Final Map
    (probe + normalize + trajectory + filtering run as supporting phases)

`run_slam(ctx)` executes every phase with per-phase timing and names the
failing phase on error, then returns timings plus an artifact manifest —
everything a caller (CLI today, API/frontend tomorrow) needs without
reaching into stage internals. `build_slam_pipeline()` exposes the same
sequence as a chainable VideoPipeline for programmatic use and tests.

LOOP-CLOSURE INTEGRATION POINT (not implemented, by request): a future
loop-closure stage plugs into SLAM_PHASES between "local_map" and
"optimization" as ("loop_closure", LoopClosureStage()) — no other change
needed. It must only APPEND constraints/observations for the optimizer
(see optimization.py), never rewrite earlier artifacts in place.
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from video_pipeline.camera_motion import CameraMotionStage  # noqa: E402
from video_pipeline.context import VideoContext  # noqa: E402
from video_pipeline.keyframes import KeyframeSelectionStage  # noqa: E402
from video_pipeline.local_map import LocalMappingStage  # noqa: E402
from video_pipeline.optimization import PoseOptimizationStage  # noqa: E402
from video_pipeline.orb_features import OrbConfig, OrbFeatureStage  # noqa: E402
from video_pipeline.orb_matching import DescriptorMatchingStage  # noqa: E402
from video_pipeline.pipeline import VideoPipeline  # noqa: E402
from video_pipeline.point_filter import PointFilterStage  # noqa: E402
from video_pipeline.stages import FrameSamplingStage, ProbeStage, Stage  # noqa: E402
from video_pipeline.trajectory import TrajectoryStage  # noqa: E402
from video_pipeline.triangulation import TriangulationStage  # noqa: E402
from video_pipeline.utils import format_summary  # noqa: E402

#: Canonical phase sequence. Names are stable identifiers used in timing
#: reports and failure attribution — do not rename lightly.
#: Performance note (Phase 13, measured): the ORB budget is set here, at the
#: orchestration layer — NOT by changing OrbConfig's default — because it is
#: a deployment speed/quality tradeoff: 1000 -> 500 keeps everyday frames
#: (typical yield ~600, dense frames cap) fully covered while cutting
#: describe/match/triangulate/SOR work. Revert to default if maps thin out.
SLAM_PHASES: List[Tuple[str, Stage]] = [
    ("probe", ProbeStage()),
    ("normalize", FrameSamplingStage()),
    ("features", OrbFeatureStage(OrbConfig(nfeatures=500))),
    ("matching", DescriptorMatchingStage()),
    ("motion", CameraMotionStage()),
    ("trajectory", TrajectoryStage()),
    ("triangulation", TriangulationStage()),
    ("filter", PointFilterStage()),
    ("keyframes", KeyframeSelectionStage()),
    ("local_map", LocalMappingStage()),
    ("optimization", PoseOptimizationStage()),
]


def build_slam_pipeline() -> VideoPipeline:
    """The canonical sequence as a chainable pipeline (tests, custom flows)."""
    pipeline = VideoPipeline()
    for _, stage in SLAM_PHASES:
        pipeline.add_stage(stage)
    return pipeline


def _collect_artifacts(ctx: VideoContext) -> List[str]:
    """Every file path recorded across ctx.shared (+ the output video)."""
    found: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, str) and Path(node).is_file():
            found.append(node)

    walk(ctx.shared)
    if ctx.output_path.is_file():
        found.append(str(ctx.output_path))
    return sorted(set(found))


def run_slam(ctx: VideoContext, skip: Collection[str] = (),
             progress_callback: Optional[Callable[[str, int, int], None]] = None,
             phase_log_callback: Optional[Callable[[str, str, float, bool], None]] = None) -> Dict[str, Any]:
    """Run the full SLAM flow; return context, timings and artifact manifest.

    Args:
        skip: Phase names to skip (e.g. {"optimization"}). Skipped phases
            neither run nor time; their outputs are simply not produced by
            this run (stale files from earlier runs are left alone — check
            timestamps if in doubt).
        progress_callback: Optional hook called as
            ``progress_callback(phase_name, index_1_based, total_active)``
            before each phase runs. Used by the API server to expose live
            stage info for polling; the CLI passes nothing (terminal
            prints remain the progress channel there).
        phase_log_callback: Optional hook called as
            ``phase_log_callback(phase_name, log_text, elapsed_sec, ok)``
            after each phase finishes, with that phase's full terminal
            output. The API server uses it to stream per-phase parameters
            to the frontend; the CLI passes nothing (it already prints
            live, so no capture is done there).

    Raises whatever the failing stage raised, after printing WHICH phase
    failed and how far the run got — failure attribution is the reason
    this loop exists instead of a bare pipeline.run().
    """
    skip = set(skip)
    unknown = skip - {name for name, _ in SLAM_PHASES}
    if unknown:
        raise ValueError(
            f"Unknown phase(s) to skip: {sorted(unknown)}. "
            f"Valid names: {[name for name, _ in SLAM_PHASES]}")
    active = [(n, s) for n, s in SLAM_PHASES if n not in skip]
    print(f"\n########## SLAM PIPELINE ({len(active)}/{len(SLAM_PHASES)} phases"
          f"{', skipped: ' + ', '.join(sorted(skip)) if skip else ''}) ##########")
    phase_times: Dict[str, float] = {}
    phase_logs: Dict[str, str] = {}
    total_start = time.perf_counter()
    for index, (name, stage) in enumerate(active, 1):
        print(f"\n===== phase {index}/{len(active)}: {name} "
              f"({stage.name}) =====")
        if progress_callback is not None:
            try:
                progress_callback(name, index, len(active))
            except Exception:
                pass  # progress reporting must never break the SLAM run
        if phase_log_callback is None:
            # CLI path: live terminal printing, no capture.
            start = time.perf_counter()
            try:
                ctx = stage(ctx)
            except Exception as exc:
                print(f"\n[PHASE FAILED] '{name}' ({stage.name}) after "
                      f"{time.perf_counter() - start:.1f}s: {exc}")
                raise
            elapsed = time.perf_counter() - start
        else:
            # API path: capture this phase's stdout for live streaming.
            buf = io.StringIO()
            start = time.perf_counter()
            try:
                with contextlib.redirect_stdout(buf):
                    ctx = stage(ctx)
            except Exception as exc:
                elapsed = time.perf_counter() - start
                log = buf.getvalue()
                phase_logs[name] = log
                try:
                    phase_log_callback(name, log, elapsed, False)
                except Exception:
                    pass  # log streaming must never break the SLAM run
                print(log, end="")  # keep the server console complete
                print(f"\n[PHASE FAILED] '{name}' ({stage.name}) after "
                      f"{elapsed:.1f}s: {exc}")
                raise
            elapsed = time.perf_counter() - start
            log = buf.getvalue()
            phase_logs[name] = log
            try:
                phase_log_callback(name, log, elapsed, True)
            except Exception:
                pass  # log streaming must never break the SLAM run
            print(log, end="")  # keep the server console complete
        phase_times[name] = elapsed
        print(f"----- phase '{name}' done in {elapsed:.1f}s -----")

    total = time.perf_counter() - total_start
    if ctx.source is not None and ctx.result is not None:
        print(format_summary(ctx.source, ctx.result, ctx.target_fps,
                             (ctx.target_width, ctx.target_height)))
    artifacts = _collect_artifacts(ctx)
    print("\n########## SLAM TIMING ##########")
    for name, seconds in phase_times.items():
        print(f"  {name:<14}: {seconds:7.1f}s")
    print(f"  {'TOTAL':<14}: {total:7.1f}s")
    print(f"  Artifacts      : {len(artifacts)} files")
    print("#################################\n")
    return {"context": ctx, "phase_times": phase_times,
            "phase_logs": phase_logs,
            "total_seconds": total, "artifacts": artifacts}
