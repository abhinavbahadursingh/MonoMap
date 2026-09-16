"""Shared data objects passed between pipeline stages.

Every stage receives a VideoContext, reads what it needs, and writes
its results back onto it. Future stages (detection, keyframes, embeddings,
...) only need to add new fields to `VideoContext.shared` — no signature
changes required anywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class VideoMetadata:
    """Technical properties of one video file."""

    path: Path
    fps: float
    width: int
    height: int
    frame_count: int
    duration_sec: float

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "fps": round(self.fps, 3),
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "frame_count": self.frame_count,
            "duration_sec": round(self.duration_sec, 3),
        }


@dataclass
class VideoContext:
    """Mutable state threaded through every stage of the pipeline.

    Attributes:
        input_path: Original video file.
        output_path: Where the normalized video will be written.
        target_fps: Desired output frame rate (default 10).
        target_width / target_height: Desired output resolution (default 640x360).
        source: Metadata of the input video (filled by ProbeStage).
        result: Metadata of the output video (filled by FrameSamplingStage).
        shared: Free-form dict for future stages to exchange data
                (e.g. shared["frames"], shared["detections"]) without
                changing any function signatures.
    """

    input_path: Path
    output_path: Path
    target_fps: float = 10.0
    target_width: int = 640
    target_height: int = 360
    source: Optional[VideoMetadata] = None
    result: Optional[VideoMetadata] = None
    shared: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fail fast at the boundary so every stage can assume sane targets
        # instead of failing obscurely halfway through a video.
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {self.target_fps}")
        if self.target_width <= 0 or self.target_height <= 0:
            raise ValueError(
                "target resolution must be positive, got "
                f"{self.target_width}x{self.target_height}"
            )
