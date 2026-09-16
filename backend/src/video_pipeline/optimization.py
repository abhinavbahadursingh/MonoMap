"""Pose-optimization stage: Huber-robust motion-only bundle adjustment.

Refines keyframe poses to minimize keyframe-landmark reprojection error
via `scipy.optimize.least_squares`. Landmark positions stay fixed (joint
optimization is a later phase); only poses move.

Design decisions, each for a stated reason:

* Fixed data association: each mapped landmark is associated once, up
  front, to its nearest keyframe keypoint within `assoc_gate_px` (using
  the current poses), and the association is frozen during optimization.
  Re-associating mid-solve would make the objective a moving target.
* Huber loss: a gated nearest neighbor can still mismatch in repetitive
  texture; Huber downweights large residuals instead of letting one bad
  association drag a pose.
* Gauge fixed by freezing keyframe ordinal 0: with landmarks fixed, an
  unconstrained rigid shift of all poses leaves every residual unchanged,
  so one pose must be pinned or the solver wanders a flat valley.
* Rotation as Rodrigues vector + camera center (6 params/keyframe):
  minimal, singularity-free for the small refinements expected here.
* Observations in keyframe 0 are excluded from the residual set: with
  both its pose and the landmarks fixed they are constants — they would
  add arithmetic, not information. Before/after compare the same set.

LOOP-CLOSURE EXTENSION POINT (not implemented, by request): a future
loop-closure phase inserts loop edges by APPENDING rows to the
observation arrays built in `build_observations()` (extra (kf, landmark,
uv) triples with their own weights) before `optimize_poses()` runs. No
other change needed — residuals, parameters and gauge handling are
agnostic to where observations came from.

UNITS — arbitrary, never meters. Optimization changes pose VALUES, never
what the units mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from video_pipeline.context import VideoContext
from video_pipeline.stages import Stage
from video_pipeline.utils import render_colored_triview, sibling_output

#: Stamped on terminal output, the .npz, and shared context.
OPT_SCALE_NOTE = (
    "Optimized poses remain in ARBITRARY UNITS, not meters: least-squares "
    "refines values within the monocular gauge, it cannot observe scale."
)


@dataclass
class OptimizationConfig:
    """Tuning knobs for pose optimization.

    Attributes:
        assoc_gate_px: Association gate — a landmark claims its nearest
            keyframe keypoint only within this radius (pixels).
        huber_f_scale: Huber knee in pixels; residuals beyond this grow
            linearly instead of quadratically.
        visualize: Master switch for the trajectory-overlay + map plots
            (the optimized .npz is always saved).
    """

    assoc_gate_px: float = 3.0
    huber_f_scale: float = 1.0
    visualize: bool = True


def project_points(X: np.ndarray, R: np.ndarray, C: np.ndarray,
                   K: np.ndarray) -> np.ndarray:
    """Project (N,3) world points to pixels: uv = K * R(X - C) / z."""
    cam = (X - C) @ R.T
    z = np.where(cam[:, 2] == 0, np.nan, cam[:, 2])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([fx * cam[:, 0] / z + cx, fy * cam[:, 1] / z + cy], axis=1)


def _rvec_to_R(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.astype(np.float64))
    return R


def _R_to_rvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.array(R, dtype=np.float64))
    return rvec.reshape(3)


def build_observations(kf_ids: List[int], kf_R: List[np.ndarray],
                       kf_C: List[np.ndarray], kf_keypoints: List[np.ndarray],
                       landmark_pos: np.ndarray, landmark_obs: List[List[int]],
                       K: np.ndarray, gate_px: float) -> Dict[str, np.ndarray]:
    """Associate landmarks to keypoints by projection-gated nearest neighbor.

    For every (landmark, observing-keyframe) edge except keyframe ordinal 0
    (gauge-fixed; see module docstring), project with the current pose and
    claim the nearest keypoint within the gate. Returns stacked arrays:
    kf_ord (O,), landmark (O,), uv (O, 2).
    """
    obs_kf, obs_lm, obs_uv = [], [], []
    for kf_ord in range(1, len(kf_ids)):
        edges = [m for m in range(len(landmark_pos)) if kf_ord in landmark_obs[m]]
        if not edges or len(kf_keypoints[kf_ord]) == 0:
            continue
        proj = project_points(landmark_pos[edges], kf_R[kf_ord], kf_C[kf_ord],
                              K)
        kpxy = np.array(kf_keypoints[kf_ord])[:, :2]
        finite = np.isfinite(proj).all(axis=1)
        dist2 = ((proj[finite, None, :] - kpxy[None, :, :]) ** 2).sum(-1)
        best = np.argmin(dist2, axis=1)
        ok = dist2[np.arange(len(best)), best] <= gate_px ** 2
        edge_idx = np.array(edges)[finite][ok]
        obs_kf.extend([kf_ord] * len(edge_idx))
        obs_lm.extend(edge_idx.tolist())
        obs_uv.extend(kpxy[best[ok]].tolist())
    return {"kf_ord": np.array(obs_kf, dtype=np.int32),
            "landmark": np.array(obs_lm, dtype=np.int32),
            "uv": np.array(obs_uv, dtype=np.float64).reshape(-1, 2)}


def pack_params(rvecs: List[np.ndarray], centers: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.concatenate([r, c]) for r, c in zip(rvecs, centers)])


def unpack_params(params: np.ndarray, n_kf: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    blocks = params.reshape(n_kf, 6)
    return ([_rvec_to_R(blocks[i, :3]) for i in range(n_kf)],
            [blocks[i, 3:6].copy() for i in range(n_kf)])


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _so3_right_jacobian(rvec: np.ndarray) -> np.ndarray:
    """Right Jacobian of Exp: d(Rot(r) Y)/dr = -R [Y]x Jr. Series-safe at 0."""
    theta = float(np.linalg.norm(rvec))
    Kmat = _skew(rvec)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * Kmat + (1.0 / 6.0) * (Kmat @ Kmat)
    K1 = Kmat / theta
    return (np.eye(3) - ((1.0 - np.cos(theta)) / theta) * K1
            + ((theta - np.sin(theta)) / theta) * (K1 @ K1))


def analytic_jacobian(params: np.ndarray, opt_ordinals: List[int],
                      rvec_fixed: np.ndarray, C_fixed: np.ndarray,
                      obs: Dict[str, np.ndarray], landmark_pos: np.ndarray,
                      K: np.ndarray):
    """Exact sparse Jacobian of `reprojection_residuals` (same signature).

    Per observation (kf k, world X, observed uv*), with Y = X - C_k and
    U = R_k Y = (Ux, Uy, Uz) and e = uv* - [fx Ux/Uz + cx, fy Uy/Uz + cy]:

        dU/dC = -R,                dU/dr = -R [Y]x Jr(r)
        de/dU = -[[fx/Uz, 0, -fx Ux/Uz^2], [0, fy/Uz, -fy Uy/Uz^2]]
        J_C = de/dU @ dU/dC,       J_r = de/dU @ dU/dr

    Same minimum as finite differences, ~15x fewer residual evaluations.
    """
    from scipy import sparse
    R_list, C_list = unpack_params(params, len(opt_ordinals))
    slot_of = {k: i for i, k in enumerate(opt_ordinals)}
    slot = np.array([slot_of[int(k)] for k in obs["kf_ord"]])
    X = landmark_pos[obs["landmark"]]
    R_all = np.array([R_list[i] for i in slot])
    C_all = np.array([C_list[i] for i in slot])
    Y = X - C_all
    U = np.einsum("nij,nj->ni", R_all, Y)
    Uz = np.where(U[:, 2] == 0, np.nan, U[:, 2])
    fx, fy = K[0, 0], K[1, 1]

    dU = np.zeros((len(X), 2, 3))  # de/dU per observation
    dU[:, 0, 0] = -fx / Uz
    dU[:, 0, 2] = fx * U[:, 0] / Uz ** 2
    dU[:, 1, 1] = -fy / Uz
    dU[:, 1, 2] = fy * U[:, 1] / Uz ** 2
    J_C = dU @ (-R_all)

    Yx = np.zeros((len(X), 3, 3))  # batched [Y]x
    Yx[:, 0, 1], Yx[:, 0, 2] = -Y[:, 2], Y[:, 1]
    Yx[:, 1, 0], Yx[:, 1, 2] = Y[:, 2], -Y[:, 0]
    Yx[:, 2, 0], Yx[:, 2, 1] = -Y[:, 1], Y[:, 0]
    rvecs = np.asarray(params, dtype=np.float64).reshape(len(opt_ordinals), 6)[:, :3]
    Jr_kf = np.array([_so3_right_jacobian(rv) for rv in rvecs])
    J_r = dU @ (-R_all @ Yx @ Jr_kf[slot])

    J_dense = np.concatenate([J_r, J_C], axis=2)  # (O, 2, 6)
    n_obs, n_params = len(X), 6 * len(opt_ordinals)
    rows = np.repeat((np.arange(n_obs)[:, None] * 2 + np.array([0, 1])).reshape(-1), 6)
    cols = (slot[:, None, None] * 6 + np.arange(6)[None, None, :])
    cols = np.broadcast_to(cols, (n_obs, 2, 6)).reshape(-1)
    return sparse.csr_matrix((J_dense.reshape(-1), (rows, cols)),
                             shape=(2 * n_obs, n_params))


def reprojection_residuals(params: np.ndarray, opt_ordinals: List[int],
                           rvec_fixed: np.ndarray, C_fixed: np.ndarray,
                           obs: Dict[str, np.ndarray], landmark_pos: np.ndarray,
                           K: np.ndarray) -> np.ndarray:
    """Stacked (observed - projected) pixel residuals for the observation set."""
    R_list, C_list = unpack_params(params, len(opt_ordinals))
    R_all = {0: _rvec_to_R(rvec_fixed)}
    C_all = {0: C_fixed.copy()}
    R_all.update(dict(zip(opt_ordinals, R_list)))
    C_all.update(dict(zip(opt_ordinals, C_list)))
    # Gather per-observation pose (all vectorized below; no per-residual Python).
    all_R = np.array([R_all[int(k)] for k in obs["kf_ord"]])
    all_C = np.array([C_all[int(k)] for k in obs["kf_ord"]])
    X = landmark_pos[obs["landmark"]]
    cam = np.einsum("nij,nj->ni", all_R, X - all_C)
    z = np.where(cam[:, 2] == 0, np.nan, cam[:, 2])
    proj = np.stack([K[0, 0] * cam[:, 0] / z + K[0, 2],
                     K[1, 1] * cam[:, 1] / z + K[1, 2]], axis=1)
    return (obs["uv"] - proj).reshape(-1)


def rms(errors: np.ndarray) -> float:
    flat = np.asarray(errors, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(flat ** 2))) if len(flat) else float("nan")


class PoseOptimizationStage(Stage):
    """Stage 11 — Huber-robust pose-only bundle adjustment + final map.

    Loads `_local_map.npz` (poses, keypoints, landmark links) and `_motion.npz`
    (the shared K) — read-only. Records on `ctx.shared["optimization"]`:

        shared["optimization"] = {
            "rms_before": ..., "rms_after": ..., "n_obs": ...,
            "n_params": ..., "scipy": {...}, "stats": {...},
            "scale_note": ..., "optimized_path": ...,
            "traj_plot_path": ..., "map_plot_path": ...,
        }
    """

    name = "pose_optimization"

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()

    # ------------------------------------------------------------------ #
    # stage entry point
    # ------------------------------------------------------------------ #
    def process(self, ctx: VideoContext) -> VideoContext:
        video_path = ctx.output_path
        map_path = sibling_output(video_path, "_local_map.npz")
        motion_path = sibling_output(video_path, "_motion.npz")
        for needed, stage in ((map_path, "local mapping"), (motion_path, "motion")):
            if not needed.exists():
                raise IOError(f"{stage} data missing for optimization: {needed}")
        with np.load(str(map_path), allow_pickle=False) as z:
            saved = {k: z[k] for k in z.files}
        with np.load(str(motion_path), allow_pickle=False) as z:
            K = np.array(z["K"], dtype=np.float64)

        kf_ids = [int(i) for i in saved["keyframe_ids"]]
        kf_R = [np.array(m, dtype=np.float64) for m in saved["keyframe_R"]]
        kf_C = [np.array(p, dtype=np.float64) for p in saved["keyframe_pos"]]
        feat_off = saved["kf_feat_offsets"].astype(int)
        kf_kp = [np.array(saved["kf_keypoints"][feat_off[k]:feat_off[k + 1]])
                 for k in range(len(kf_ids))]
        lm_off, lm_ids = saved["lm_kf_offsets"].astype(int), saved["lm_kf_ids"]
        landmark_obs = [lm_ids[lm_off[m]:lm_off[m + 1]].tolist()
                        for m in range(len(saved["landmark_pos"]))]
        landmark_pos = np.array(saved["landmark_pos"], dtype=np.float64)

        obs = build_observations(kf_ids, kf_R, kf_C, kf_kp, landmark_pos,
                                 landmark_obs, K, self.config.assoc_gate_px)
        opt_ordinals = sorted(set(int(k) for k in obs["kf_ord"]))
        fitted = self._optimize(obs, opt_ordinals, kf_R, kf_C, landmark_pos, K)

        optimized_path = sibling_output(video_path, "_slam_optimized.npz")
        self._save_optimized(optimized_path, saved, kf_ids, fitted, K, obs,
                             landmark_pos)
        visuals: Dict[str, str] = {}
        if self.config.visualize:
            visuals["traj_plot_path"] = str(self._save_traj_plot(
                video_path, kf_ids, fitted, kf_C))
            visuals["map_plot_path"] = str(self._save_map_plot(
                video_path, saved, obs))

        self._report(ctx, optimized_path, visuals, kf_ids, fitted, obs)
        return ctx

    # ------------------------------------------------------------------ #
    # optimization
    # ------------------------------------------------------------------ #
    def _optimize(self, obs: Dict[str, np.ndarray], opt_ordinals: List[int],
                  kf_R: List[np.ndarray], kf_C: List[np.ndarray],
                  landmark_pos: np.ndarray, K: np.ndarray) -> Dict:
        n_obs = len(obs["kf_ord"])
        rvec_fixed, C_fixed = _R_to_rvec(kf_R[0]), kf_C[0].copy()
        if not n_obs or not opt_ordinals:
            return {"opt_R": dict(enumerate(kf_R)), "opt_C": dict(enumerate(kf_C)),
                    "opt_ordinals": [], "rms_before": float("nan"),
                    "rms_after": float("nan"), "scipy": {"status": "skipped: no observations"},
                    "n_obs": 0}
        x0 = pack_params([_R_to_rvec(kf_R[k]) for k in opt_ordinals],
                         [kf_C[k] for k in opt_ordinals])
        args = (opt_ordinals, rvec_fixed, C_fixed, obs, landmark_pos, K)
        rms_before = rms(reprojection_residuals(x0, *args))
        result = least_squares(
            reprojection_residuals, x0, jac=analytic_jacobian, args=args,
            method="trf", loss="huber", f_scale=self.config.huber_f_scale,
            verbose=0)
        rms_after = rms(reprojection_residuals(result.x, *args))
        R_opt, C_opt = unpack_params(result.x, len(opt_ordinals))
        opt_R = dict(enumerate(kf_R))
        opt_C = dict(enumerate(kf_C))
        opt_R.update(dict(zip(opt_ordinals, R_opt)))
        opt_C.update(dict(zip(opt_ordinals, C_opt)))
        return {"opt_R": opt_R, "opt_C": opt_C, "opt_ordinals": opt_ordinals,
                "rms_before": rms_before, "rms_after": rms_after,
                "scipy": {"cost": float(result.cost), "nfev": int(result.nfev),
                          "optimality": float(result.optimality),
                          "status": int(result.status),
                          "message": str(result.message).strip(),
                          "success": bool(result.success),
                          "jacobian": "analytic-sparse"},
                "n_obs": n_obs}

    # ------------------------------------------------------------------ #
    # persistence (.npz — optimized poses + final map)
    # ------------------------------------------------------------------ #
    def _save_optimized(self, path: Path, saved: Dict, kf_ids: List[int],
                        fitted: Dict, K: np.ndarray,
                        obs: Dict[str, np.ndarray],
                        landmark_pos: np.ndarray) -> None:
        n_kf = len(kf_ids)
        status = []
        for k in range(n_kf):
            if k == 0:
                status.append("fixed_gauge")
            elif k in fitted["opt_ordinals"]:
                status.append("optimized")
            else:
                status.append("no_observations")
        obs_per_lm = np.bincount(obs["landmark"], minlength=len(landmark_pos))
        np.savez_compressed(
            path,
            keyframe_ids=np.array(kf_ids, dtype=np.int32),
            opt_R=np.array([fitted["opt_R"][k] for k in range(n_kf)]),
            opt_pos=np.array([fitted["opt_C"][k] for k in range(n_kf)]),
            kf_status=np.array(status),
            landmark_pos=np.array(saved["landmark_pos"], dtype=np.float64),
            landmark_colors=np.array(saved["landmark_colors"], dtype=np.uint8),
            landmark_pair=np.array(saved["landmark_pair"], dtype=np.int32),
            landmark_segment=np.array(saved["landmark_segment"], dtype=np.int32),
            obs_per_landmark=obs_per_lm.astype(np.int32),
            K=K,
            rms_before=np.array(fitted["rms_before"]),
            rms_after=np.array(fitted["rms_after"]),
            n_obs=np.array(fitted["n_obs"]),
            convention=np.array(
                "opt_R maps world to camera frame; opt_pos are optimized camera "
                "centers; keyframe ordinal 0 is gauge-fixed; landmarks unchanged "
                "(pose-only optimization)"),
            scale_note=np.array(OPT_SCALE_NOTE),
        )

    # ------------------------------------------------------------------ #
    # visualization
    # ------------------------------------------------------------------ #
    def _save_traj_plot(self, video_path: Path, kf_ids: List[int],
                        fitted: Dict, orig_C: List[np.ndarray]) -> Path:
        """X-Z: original keyframe path (gray) vs optimized (green)."""
        path = sibling_output(video_path, "_slam_traj_plot.png")
        W, H, M = 1200, 700, 70
        img = np.full((H, W, 3), 24, dtype=np.uint8)
        orig = np.array(orig_C, dtype=np.float64)
        opt = np.array([fitted["opt_C"][k] for k in range(len(kf_ids))])
        span = float(np.abs(np.vstack([orig, opt])[:, [0, 2]]).max(initial=0.0)) or 1.0
        scale = min(W - 2 * M, H - 2 * M) / (2 * span * 1.1)
        cx, cy = W // 2, H // 2

        def located(p: np.ndarray) -> tuple[int, int]:
            return (round(cx + p[0] * scale), round(cy - p[2] * scale))

        for cloud, color, width in ((orig, (120, 120, 120), 1), (opt, (0, 255, 0), 2)):
            pts = np.array([located(p) for p in cloud], dtype=np.int32)
            cv2.polylines(img, [pts], False, color, width)
        for k, frame in enumerate(kf_ids):
            cv2.circle(img, located(opt[k]), 6, (0, 255, 255), 2)
            if k % max(1, len(kf_ids) // 20) == 0:
                cv2.putText(img, str(frame), (located(opt[k])[0] + 9, located(opt[k])[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(img,
                    f"Optimized keyframe trajectory X-Z: gray=before, green=after "
                    f"(RMS {fitted['rms_before']:.2f}->{fitted['rms_after']:.2f}px, "
                    f"ARBITRARY UNITS - NOT meters)", (M, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imwrite(str(path), img)
        return path

    def _save_map_plot(self, video_path: Path, saved: Dict,
                       obs: Dict[str, np.ndarray]) -> Path:
        """Tri-view of the final map, colored by observation count."""
        path = sibling_output(video_path, "_slam_map_plot.png")
        pts = np.array(saved["landmark_pos"], dtype=np.float64)
        counts = np.bincount(obs["landmark"], minlength=len(pts))

        def color_for(n: int, nmax: int) -> tuple[int, int, int]:
            if n == 0:
                return (128, 128, 128)
            f = n / max(nmax, 1)
            return (int(255 * f), int(255 * (1 - f * 0.5)), 0)

        nmax = int(counts.max(initial=0))
        colors = np.array([color_for(int(n), nmax) for n in counts], dtype=np.uint8)
        render_colored_triview(
            path, pts, colors,
            f"Final map: {len(pts)} landmarks by observation count "
            f"(gray=unobserved, ARBITRARY UNITS - NOT meters)")
        return path

    # ------------------------------------------------------------------ #
    # terminal report
    # ------------------------------------------------------------------ #
    def _report(self, ctx: VideoContext, optimized_path: Path,
                visuals: Dict[str, str], kf_ids: List[int],
                fitted: Dict, obs: Dict[str, np.ndarray]) -> None:
        stats = {
            "n_keyframes": len(kf_ids),
            "n_optimized": len(fitted["opt_ordinals"]),
            "n_obs": fitted["n_obs"],
            "n_params": 6 * len(fitted["opt_ordinals"]),
            "rms_before": fitted["rms_before"],
            "rms_after": fitted["rms_after"],
        }
        ctx.shared["optimization"] = {
            **stats,
            "scipy": fitted["scipy"],
            "scale_note": OPT_SCALE_NOTE,
            "optimized_path": str(optimized_path),
            **visuals,
        }
        line = "-" * 52
        print(f"\n{line}\nPOSE OPTIMIZATION (Huber-robust bundle adjustment)\n{line}")
        print(f"  Observations    : {stats['n_obs']} "
              f"(keyframe-landmark edges, kf0 excluded)")
        print(f"  Parameters      : {stats['n_params']} "
              f"({stats['n_optimized']}/{stats['n_keyframes']} keyframes; kf0 gauge-fixed)")
        print(f"  RMS reproj before: {stats['rms_before']:.3f} px")
        print(f"  RMS reproj after : {stats['rms_after']:.3f} px")
        print(f"  SciPy           : {fitted['scipy']}")
        print(f"  NOTE: {OPT_SCALE_NOTE}")
        print(f"  Optimized saved : {optimized_path}")
        for label, vis_path in visuals.items():
            print(f"  {label:<15}: {vis_path}")
        print(line)
