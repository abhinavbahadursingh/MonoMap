"""Chainable, extensible pipeline runner."""
from __future__ import annotations

from typing import Iterable, List, Optional

from video_pipeline.context import VideoContext
from video_pipeline.stages import Stage
from video_pipeline.utils import format_summary


class VideoPipeline:
    """Ordered list of stages applied to a VideoContext.

    Example:
        pipeline = VideoPipeline([ProbeStage(), FrameSamplingStage()])
        # later, attach new work with zero changes to existing code:
        pipeline.add_stage(MyNewStage())
        ctx = pipeline.run(ctx)
    """

    def __init__(self, stages: Optional[Iterable[Stage]] = None):
        self._stages: List[Stage] = list(stages) if stages else []

    def add_stage(self, stage: Stage) -> "VideoPipeline":
        """Append a stage; returns self so calls can be chained."""
        self._stages.append(stage)
        return self

    def insert_stage(self, index: int, stage: Stage) -> "VideoPipeline":
        self._stages.insert(index, stage)
        return self

    @property
    def stages(self) -> List[Stage]:
        return list(self._stages)

    def run(self, ctx: VideoContext) -> VideoContext:
        print(f"\n===== Running pipeline ({len(self._stages)} stages) =====")
        for i, stage in enumerate(self._stages, 1):
            print(f"--- step {i}/{len(self._stages)}: {stage.name} ---")
            ctx = stage(ctx)
        if ctx.source is not None and ctx.result is not None:
            print(format_summary(
                ctx.source, ctx.result, ctx.target_fps,
                (ctx.target_width, ctx.target_height),
            ))
        print("===== Pipeline finished =====\n")
        return ctx
