"""Pipeline stages.

Each stage is a small unit with one job: it receives a VideoContext,
does its work, records the outcome on the context, and returns it.

To extend the pipeline, subclass `Stage` and implement `process()`:

    class MyDetector(Stage):
        name = "my_detector"

        def process(self, ctx: VideoContext) -> VideoContext:
            ctx.shared["detections"] = detect(ctx.output_path)
            return ctx

    pipeline.add_stage(MyDetector())

Existing stages never need to change.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from fractions import Fraction
from pathlib import Path

import cv2

from video_pipeline.context import VideoContext, VideoMetadata
from video_pipeline.utils import ensure_parent_dir, format_metadata_block, validate_input


class Stage(ABC):
    """One unit of work in the pipeline."""

    name: str = "stage"

    @abstractmethod
    def process(self, ctx: VideoContext) -> VideoContext:
        """Run the stage, record results on ctx, and return it."""
        raise NotImplementedError

    def __call__(self, ctx: VideoContext) -> VideoContext:
        print(f"\n>>> [Stage] {self.name} ...")
        ctx = self.process(ctx)
        print(f"<<< [Stage] {self.name} done.")
        return ctx


def probe_video(path: Path) -> VideoMetadata:
    """Read FPS / size / frame-count / duration using only OpenCV.

    Some containers store no frame count; for those we decode once and
    count, which is slower but always correct.
    """
    validate_input(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise ValueError(f"Video reports no usable frame rate: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            frame_count = _count_frames_by_decoding(cap)
        return VideoMetadata(
            path=path,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_sec=frame_count / fps,
        )
    finally:
        cap.release()


def _count_frames_by_decoding(cap: cv2.VideoCapture) -> int:
    """Decode-and-count fallback for headers without a frame count."""
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            return count
        count += 1


def _resample_ratio(source_fps: float, target_fps: float) -> Fraction:
    """Exact output-frames-per-input-frame ratio.

    Frame rates are decimal quantities (25, 30, 29.97), so building the
    fraction from their decimal text keeps it exact. Every sampling
    decision below is integer math on this ratio — no float timestamps,
    no epsilon corrections, and therefore no drift to compensate for.
    """
    return Fraction(str(target_fps)) / Fraction(str(round(source_fps, 3)))


def _frames_to_emit(input_index: int, ratio: Fraction) -> int:
    """How many output frames input frame `input_index` contributes.

    Output slot k starts at input-time k / ratio, and input frame i spans
    input-time [i, i + 1), so frame i emits one output frame per slot
    starting inside it: ceil((i + 1) * ratio) - ceil(i * ratio).
    Downsampling yields 0 or 1, upsampling duplicates the frame, and the
    total over a stream is exactly ceil(frames * ratio) — by construction,
    not by a padding pass afterwards.
    """
    slot_start = ratio * input_index
    return math.ceil(slot_start + ratio) - math.ceil(slot_start)


def _require_usable_output(path: Path) -> None:
    """Fail loudly if the writer produced nothing usable."""
    if not path.exists() or path.stat().st_size == 0:
        raise IOError(f"Writer produced no output: {path}")


class ProbeStage(Stage):
    """Stage 1 — extract input video metadata and show it."""

    name = "probe_input"

    def process(self, ctx: VideoContext) -> VideoContext:
        ctx.source = probe_video(ctx.input_path)
        print(format_metadata_block("INPUT VIDEO INFO", ctx.source))
        return ctx


class FrameSamplingStage(Stage):
    """Stage 2 — resample to target FPS and resize to target resolution.

    Sampling uses exact rational slot assignment (see `_frames_to_emit`),
    so any input rate maps to any target rate and the output count is
    exact. Frames stream through one at a time, so memory use stays flat
    regardless of video length.
    """

    name = "frame_sampling(normalize)"

    def process(self, ctx: VideoContext) -> VideoContext:
        if ctx.source is None:
            # ProbeStage normally provides this; probing here keeps the
            # stage usable standalone.
            ctx.source = probe_video(ctx.input_path)
        source = ctx.source
        ratio = _resample_ratio(source.fps, ctx.target_fps)
        target_size = (ctx.target_width, ctx.target_height)

        ensure_parent_dir(ctx.output_path)
        cap = cv2.VideoCapture(str(ctx.input_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {ctx.input_path}")
        writer = cv2.VideoWriter(
            str(ctx.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            ctx.target_fps,
            target_size,
        )
        if not writer.isOpened():
            cap.release()
            raise IOError(f"Could not open writer for: {ctx.output_path}")

        frames_read = 0
        frames_written = 0
        try:
            # Driven by the decoder, not the header count: the header is
            # metadata and can be wrong, the decoded stream is fact.
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                emits = _frames_to_emit(frames_read, ratio)
                if emits:
                    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                    for _ in range(emits):
                        writer.write(resized)
                        frames_written += 1
                frames_read += 1
                if frames_read == 1 or frames_read % 100 == 0:
                    print(f"    ... {frames_read} frames in, {frames_written} out", end="\r")
        finally:
            cap.release()
            writer.release()

        if frames_read == 0:
            raise IOError(f"Video contained no decodable frames: {ctx.input_path}")
        if frames_written != math.ceil(ratio * frames_read):
            # Unreachable by construction (see `_frames_to_emit`); a loud
            # failure beats silently writing a short video.
            raise RuntimeError(
                f"Sampling invariant broken: {frames_read} in -> {frames_written} out"
            )
        _require_usable_output(ctx.output_path)

        print(f"\n    Sampling: {frames_read} input frames -> {frames_written} output frames")
        ctx.shared["frames_read"] = frames_read
        ctx.shared["frames_written"] = frames_written
        # Built from what we actually wrote. The container header stores
        # FPS only approximately, so re-probing it here would report a
        # slightly wrong rate — our write parameters are the truth.
        ctx.result = VideoMetadata(
            path=ctx.output_path,
            fps=float(ctx.target_fps),
            width=ctx.target_width,
            height=ctx.target_height,
            frame_count=frames_written,
            duration_sec=frames_written / ctx.target_fps,
        )
        print(format_metadata_block("OUTPUT VIDEO INFO (normalized)", ctx.result))
        return ctx
