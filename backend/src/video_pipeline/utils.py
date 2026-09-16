"""Terminal / filesystem / chart helpers (OpenCV + numpy only)."""
from __future__ import annotations

from pathlib import Path
from typing import Collection, Optional, Sequence

import cv2
import numpy as np

from video_pipeline.context import VideoMetadata

_CHART_GREEN = (0, 255, 0)
_CHART_RED = (0, 0, 255)
_CHART_GRAY = (180, 180, 180)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sibling_output(video_path: Path, tag: str) -> Path:
    """`<stem><tag>` next to a video, e.g. `clip_orb.npz` for tag `"_orb.npz"`."""
    return video_path.with_name(video_path.stem + tag)


def validate_input(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Input video not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")
    return path


def format_metadata_block(title: str, meta: VideoMetadata) -> str:
    line = "-" * 52
    return (
        f"\n{line}\n{title}\n{line}\n"
        f"  File        : {meta.path}\n"
        f"  FPS         : {meta.fps:.3f}\n"
        f"  Resolution  : {meta.width} x {meta.height}\n"
        f"  Frame count : {meta.frame_count}\n"
        f"  Duration    : {meta.duration_sec:.3f} s\n"
        f"{line}"
    )


def format_summary(source: VideoMetadata, result: VideoMetadata,
                   target_fps: float, target_size: tuple[int, int]) -> str:
    tw, th = target_size
    return (
        "\n================ PIPELINE SUMMARY ================\n"
        f"  Input          : {source.path.name} "
        f"({source.fps:.2f} FPS, {source.width}x{source.height}, "
        f"{source.frame_count} frames, {source.duration_sec:.2f}s)\n"
        f"  Target         : {target_fps:g} FPS, {tw}x{th}\n"
        f"  Output         : {result.path.name} "
        f"({result.fps:.2f} FPS, {result.width}x{result.height}, "
        f"{result.frame_count} frames, {result.duration_sec:.2f}s)\n"
        "==================================================\n"
    )


def save_line_chart(path: Path, title: str, values: Sequence[int],
                    threshold: Optional[float] = None,
                    threshold_label: str = "",
                    bad_indices: Collection[int] = (),
                    width: int = 1280, height: int = 360) -> None:
    """Dark line chart: value per index, red threshold line, red bad markers.

    Shared by every stage that plots a per-frame/per-pair series, so all
    timelines look and read the same.
    """
    values = np.asarray(list(values), dtype=np.int32)
    bad = set(bad_indices)
    W, H, L, R, T, B = width, height, 70, 25, 40, 45
    img = np.full((H, W, 3), 24, dtype=np.uint8)
    n = len(values)
    vmax = float(max(values.max(initial=0), threshold or 0) * 1.1 or 1.0)

    def x(i: int) -> int:
        return L if n == 1 else L + round(i * (W - L - R) / (n - 1))

    def y(v: float) -> int:
        return H - B - round(v * (H - T - B) / vmax)

    for g in range(5):  # gridlines with values
        v = vmax * g / 4
        cv2.line(img, (L, y(v)), (W - R, y(v)), (60, 60, 60), 1)
        cv2.putText(img, f"{v:.0f}", (10, y(v) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _CHART_GRAY, 1)
    if threshold is not None:
        cv2.line(img, (L, y(threshold)), (W - R, y(threshold)), _CHART_RED, 1)
        cv2.putText(img, threshold_label or f"{threshold:g}", (W - R - 130, y(threshold) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _CHART_RED, 2)
    if n:
        pts = np.array([[x(i), y(v)] for i, v in enumerate(values)], dtype=np.int32)
        cv2.polylines(img, [pts], False, _CHART_GREEN, 2)
    for i in bad:  # bad indices marked on the axis
        cv2.circle(img, (x(i), H - B), 6, _CHART_RED, -1)
    cv2.putText(img, title, (L, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _CHART_GREEN, 2)
    cv2.imwrite(str(path), img)


def render_colored_triview(path: Path, points_xyz: np.ndarray, colors_rgb: np.ndarray,
                           title: str, panel: int = 400, max_drawn: int = 20000) -> None:
    """Orthographic X-Y / X-Z / Y-Z tri-view of a colored cloud (OpenCV only).

    Shared renderer for stages that visualize point clouds, so every map
    preview looks and reads the same. Axes are auto-scaled per view with
    ranges labeled — read the labels, not the pixels, for proportions.
    """
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    bgr = np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)[:, ::-1].copy()
    if len(pts) != len(bgr):
        raise ValueError(
            f"points ({len(pts)}) and colors ({len(bgr)}) disagree in "
            f"render_colored_triview")
    order = np.argsort(pts[:, 2])  # far points first, near drawn over
    pts, bgr = pts[order], bgr[order]
    stride = max(1, len(pts) // max_drawn)

    views = (("X", "Y", (0, 1)), ("X", "Z", (0, 2)), ("Y", "Z", (1, 2)))
    header = 46
    canvas = np.full((header + panel, 3 * panel + 40, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, title, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _CHART_GREEN, 2)
    if len(pts) == 0:
        # An empty cloud is a legitimate result (e.g. textureless clip) —
        # render the fact instead of crashing on it.
        cv2.putText(canvas, "no points to display", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _CHART_GRAY, 2)
        cv2.imwrite(str(path), canvas)
        return
    for v, (xlabel, ylabel, axes) in enumerate(views):
        ox, oy = 10 + v * panel, header
        ax, ay = pts[::stride, axes[0]], pts[::stride, axes[1]]
        lo_x, hi_x = float(ax.min()), float(ax.max())
        lo_y, hi_y = float(ay.min()), float(ay.max())
        span = max(hi_x - lo_x, hi_y - lo_y) or 1.0

        def loc(px: float, py: float) -> tuple[int, int]:
            return (round(ox + (px - lo_x) / span * (panel - 20)),
                    round(oy + panel - 10 - (py - lo_y) / span * (panel - 20)))

        for (px, py), color in zip(zip(ax, ay), bgr[::stride]):
            cv2.circle(canvas, loc(px, py), 2, tuple(int(c) for c in color), -1)
        cv2.putText(canvas, f"{xlabel}-{ylabel}", (ox + 8, oy + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _CHART_GRAY, 1)
        cv2.putText(canvas, f"[{lo_x:+.1f},{hi_x:+.1f}]", (ox + 110, oy + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CHART_GRAY, 1)
    cv2.imwrite(str(path), canvas)
