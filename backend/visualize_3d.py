"""Standalone Open3D visualization of the final SLAM result (NOT a pipeline stage).

Loads `<stem>_slam_optimized.npz` (+ K embedded in it) and builds an
Open3D scene: colored landmark PointCloud, per-keyframe camera frustums
as one merged LineSet, world axes. Then, depending on the platform:

* headless-safe path (default): hidden legacy-Visualizer window +
  `capture_screen_image` -> `<stem>_o3d_front.png`, `<stem>_o3d_top.png`.
  (The Filament OffscreenRenderer needs EGL, which this platform lacks —
  evidenced, not assumed — so the legacy GL path is used deliberately.)
* `--show`: interactive `draw_geometries` window for local machines with
  a display (blocks until closed; skipped by default so servers work).

Always exports (viewer-openable anywhere, no display needed):
`<stem>_cloud.ply` (points + RGB) and `<stem>_cameras.ply` (frustum
wires; JSON fallback if PLY lines ever fail on a platform).

Deliberately imports NOTHING from video_pipeline: visualization stays
decoupled from core SLAM logic; the .npz files are the only contract.

Usage:
    python backend/visualize_3d.py [--stem video1_10fps_640x360] [--show]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BACKEND_DIR / "output"


def load_final(stem: str) -> dict:
    path = OUTPUT_DIR / f"{stem}_slam_optimized.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Optimized map missing (run the SLAM pipeline first): {path}")
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def build_geometries(saved: dict):
    """PointCloud + frustum LineSet + axes. Open3D imported lazily (heavy)."""
    import open3d as o3d

    points = np.array(saved["landmark_pos"], dtype=np.float64)
    colors_rgb = np.array(saved["landmark_colors"], dtype=np.float64) / 255.0
    kf_R = [np.array(m, dtype=np.float64) for m in saved["opt_R"]]
    kf_C = [np.array(p, dtype=np.float64) for p in saved["opt_pos"]]
    K = np.array(saved["K"], dtype=np.float64)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors_rgb)

    # Size camera glyphs from the ROBUST extent: max lets a few far-flung
    # points inflate every frustum past the cameras themselves.
    center = points.mean(axis=0) if len(points) else np.zeros(3)
    spread = np.linalg.norm(points - center, axis=1)
    robust_radius = float(np.percentile(spread, 95)) or 1.0
    dist = robust_radius * 0.05

    # Frustum size relative to the robust scene extent (monocular units
    # are arbitrary; this only sets a readable glyph scale).
    half_w = dist * (K[0, 2] * 2) / (2 * K[0, 0])  # true aspect from K
    half_h = dist * (K[1, 2] * 2) / (2 * K[1, 1])
    corners_cam = np.array([[0, 0, 0],
                            [-half_w, -half_h, dist], [half_w, -half_h, dist],
                            [half_w, half_h, dist], [-half_w, half_h, dist]])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    all_pts, all_lines, offset = [], [], 0
    for R_wc, C in zip(kf_R, kf_C):
        # Row-vectors: X_world = X_cam @ R_wc + C (since R_wc maps world->cam).
        world = (corners_cam @ R_wc) + C
        all_pts.append(world)
        all_lines.extend((a + offset, b + offset) for a, b in edges)
        offset += len(world)
    frustums = o3d.geometry.LineSet()
    frustums.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    frustums.lines = o3d.utility.Vector2iVector(np.array(all_lines, dtype=np.int32))
    frustums.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 1.0, 1.0], (len(all_lines), 1)))

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=dist)
    radius = float(spread.max(initial=1.0)) or 1.0
    # Robust framing radius: max lets a few far-flung points dictate the zoom.
    return cloud, frustums, axes, center, radius, robust_radius


def export_files(stem: str, cloud, frustums) -> dict:
    """Viewer-openable exports (work everywhere, display or not)."""
    import open3d as o3d

    cloud_path = OUTPUT_DIR / f"{stem}_cloud.ply"
    o3d.io.write_point_cloud(str(cloud_path), cloud)
    frustum_path = OUTPUT_DIR / f"{stem}_cameras.ply"
    try:
        ok = o3d.io.write_line_set(str(frustum_path), frustums)
        if not ok:
            raise IOError("write_line_set returned False")
    except Exception:
        frustum_path = frustum_path.with_suffix(".json")  # documented LineSet support
        o3d.io.write_line_set(str(frustum_path), frustums)
    return {"cloud_ply": str(cloud_path), "cameras_path": str(frustum_path)}


def capture_views(stem: str, geometries, center, radius) -> dict:
    """Hidden-window captures (front + top). Skipped only if GL is unavailable."""
    import open3d as o3d

    paths = {}
    views = {"front": ([0.35, 0.25, 1.0], [0, 1, 0]),
             "top": ([0.0, 1.0, 0.02], [0, 0, -1])}
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(width=1280, height=800, visible=False):
        print("[viz] hidden GL window unavailable on this platform; "
              "PNG capture skipped (PLY exports above still stand).")
        return paths
    try:
        for g in geometries:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.09, 0.09, 0.09])
        opt.point_size = 3.0
        opt.line_width = 2.0
        vc = vis.get_view_control()
        zoom = max(0.05, min(0.7, 3.0 / max(radius, 1e-6)))
        for name, (front, up) in views.items():
            vc.set_lookat(center.tolist())
            vc.set_front(list(front))
            vc.set_up(list(up))
            vc.set_zoom(zoom)
            vis.poll_events()
            vis.update_renderer()
            out = OUTPUT_DIR / f"{stem}_o3d_{name}.png"
            vis.capture_screen_image(str(out), do_render=True)
            paths[f"o3d_{name}_png"] = str(out)
    finally:
        vis.destroy_window()
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Open3D visualization of the final SLAM map")
    parser.add_argument("--stem", default="video1_10fps_640x360",
                        help="Output stem in backend/output/ (default: %(default)s)")
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive Open3D window (needs a display)")
    args = parser.parse_args(argv)

    saved = load_final(args.stem)
    print(f"Loaded {len(saved['landmark_pos'])} landmarks, "
          f"{len(saved['keyframe_ids'])} keyframes from {args.stem}_slam_optimized.npz")
    cloud, frustums, axes, center, _, robust_radius = build_geometries(saved)
    exports = export_files(args.stem, cloud, frustums)
    for label, path in exports.items():
        print(f"  {label:<12}: {path}")

    if args.show:
        import open3d as o3d
        o3d.visualization.draw_geometries([cloud, frustums, axes])
    else:
        for label, path in capture_views(args.stem, [cloud, frustums, axes],
                                         center, robust_radius).items():
            print(f"  {label:<12}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
