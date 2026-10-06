"""ChArUco touch-off TCP calibration (math only, no ROS).

The camera sees a ChArUco board lying on the table. PnP on the color image plus
the hand-eye TF gives every ChArUco corner (by ID) in the robot base frame. The
operator then puts the wire tip on at least four corners with the torch held
perpendicular to the board and captures the flange pose each time.

Each touch alone determines the flange->tip offset, because the touched point is
known in the base frame:

    R_i t + q_i = P_i   ->   t_i = R_iᵀ (P_i - q_i)

Several touches on different corners average out hand-eye/PnP errors, and their
spread is the honest uncertainty. With the torch perpendicular, the tool axis in
the flange frame is a_i = R_iᵀ (-n) (n = board normal pointing up, toward the
camera), averaged the same way.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

MIN_POINTS = 4


def make_board(squares_x, squares_y, square_m, marker_m, dictionary):
    import cv2
    name = dictionary if str(dictionary).startswith("DICT_") else f"DICT_{dictionary}"
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    return cv2.aruco.CharucoBoard((int(squares_x), int(squares_y)), float(square_m), float(marker_m), dic)


def board_pose_from_images(board, grays, K, D, min_corners=8):
    """Average ChArUco PnP over frames. Returns (T_cam_board 4x4, info) or raises."""
    import cv2
    det = cv2.aruco.CharucoDetector(board)
    obj_all = board.getChessboardCorners()
    rots, trans, seen, reproj = [], [], set(), []
    for gray in grays:
        cc, ci, _, _ = det.detectBoard(gray)
        if ci is None or len(ci) < min_corners:
            continue
        obj = obj_all[ci.ravel()]
        ok, rv, tv = cv2.solvePnP(obj, cc, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        rv, tv = cv2.solvePnPRefineLM(obj, cc, K, D, rv, tv)
        proj = cv2.projectPoints(obj, rv, tv, K, D)[0].reshape(-1, 2)
        reproj.append(float(np.linalg.norm(proj - cc.reshape(-1, 2), axis=1).mean()))
        rots.append(Rot.from_rotvec(rv.ravel()))
        trans.append(tv.ravel())
        seen.update(int(i) for i in ci.ravel())
    if not rots:
        raise ValueError("ChArUco board not detected (need at least "
                         f"{min_corners} corners in one frame)")
    R = Rot.concatenate(rots).mean().as_matrix()
    t = np.mean(trans, axis=0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    spread_mm = float(np.max(np.linalg.norm(np.asarray(trans) - t, axis=1)) * 1e3)
    return T, dict(frames=len(rots), corner_ids_seen=sorted(seen), reprojection_px=float(np.mean(reproj)),
                   translation_spread_mm=spread_mm)


def board_targets(board, T_base_board):
    """Every ChArUco corner in the base frame plus the board normal (toward the camera side).

    Returns dict(ids, points_base [N,3], points_board_mm [N,2], normal_base [3]).
    """
    obj = np.asarray(board.getChessboardCorners(), float)
    pts = obj @ T_base_board[:3, :3].T + T_base_board[:3, 3]
    n = T_base_board[:3, 2].copy()
    return dict(ids=list(range(len(obj))), points_base=pts, points_board_mm=obj[:, :2] * 1e3, normal_base=n)


def orient_normal_up(normal_base, camera_position_base, board_origin_base):
    """Board normal pointing to the side the camera looks from (the free side)."""
    n = np.asarray(normal_base, float)
    n = n / np.linalg.norm(n)
    return n if n @ (np.asarray(camera_position_base) - np.asarray(board_origin_base)) > 0 else -n


def solve(samples, normal_up=None):
    """samples: list of dict(pose=[tx,ty,tz,qx,qy,qz,qw] base<-flange, target=[x,y,z] base, id=int).

    Returns translation (mean of per-touch solutions), residuals, LOO and, when
    ``normal_up`` is given, the tool axis in the flange frame (torch perpendicular).
    """
    if len(samples) < 1:
        raise ValueError("No touches captured")
    Rs = [Rot.from_quat(s["pose"][3:]).as_matrix() for s in samples]
    qs = [np.asarray(s["pose"][:3], float) for s in samples]
    Ps = [np.asarray(s["target"], float) for s in samples]
    per = np.array([R.T @ (P - q) for R, q, P in zip(Rs, qs, Ps)])
    t = per.mean(axis=0)
    resid = np.array([np.linalg.norm(R @ t + q - P) for R, q, P in zip(Rs, qs, Ps)])
    loo = []
    if len(samples) > 1:
        for i in range(len(samples)):
            loo.append(float(np.linalg.norm(np.delete(per, i, axis=0).mean(axis=0) - t)))
    out = dict(
        tcp_translation=[float(v) for v in t],
        per_touch_translation=[[float(v) for v in p] for p in per],
        per_touch_spread_m=float(np.max(np.linalg.norm(per - t, axis=1))),
        residuals_m=[float(v) for v in resid],
        rms_residual_m=float(np.sqrt(np.mean(resid ** 2))),
        max_residual_m=float(np.max(resid)),
        loo_shifts_m=loo,
        max_loo_shift_m=float(max(loo)) if loo else None,
        ids=[int(s["id"]) for s in samples],
        distinct_ids=len({int(s["id"]) for s in samples}),
        target_span_m=float(np.max([np.linalg.norm(a - b) for a in Ps for b in Ps])) if len(Ps) > 1 else 0.0,
    )
    if normal_up is not None:
        n = np.asarray(normal_up, float)
        n = n / np.linalg.norm(n)
        dirs = np.array([R.T @ (-n) for R in Rs])
        a = dirs.mean(axis=0)
        a /= np.linalg.norm(a)
        ang = np.degrees(np.arccos(np.clip(dirs @ a, -1, 1)))
        out.update(axis_dir=[float(v) for v in a], axis_spread_deg=float(np.max(ang)),
                   per_touch_axis_deg=[float(v) for v in ang])
    return out


DEFAULT_LIMITS = dict(min_points=MIN_POINTS, max_rms_mm=1.5, max_residual_mm=3.0,
                      max_loo_shift_mm=2.0, max_axis_spread_deg=3.0, min_span_mm=40.0)


def acceptance(result, limits=None):
    lim = {**DEFAULT_LIMITS, **(limits or {})}
    checks = []

    def check(name, ok, detail):
        checks.append(dict(name=name, passed=bool(ok), detail=detail))

    check("points", result["distinct_ids"] >= lim["min_points"],
          f"{result['distinct_ids']} distinct IDs (need {lim['min_points']})")
    check("span", result["target_span_m"] * 1e3 >= lim["min_span_mm"],
          f"touched corners span {result['target_span_m'] * 1e3:.0f} mm (need {lim['min_span_mm']:.0f})")
    check("rms", result["rms_residual_m"] * 1e3 <= lim["max_rms_mm"],
          f"RMS {result['rms_residual_m'] * 1e3:.2f} mm (max {lim['max_rms_mm']})")
    check("max_residual", result["max_residual_m"] * 1e3 <= lim["max_residual_mm"],
          f"max {result['max_residual_m'] * 1e3:.2f} mm (max {lim['max_residual_mm']})")
    if result.get("max_loo_shift_m") is not None:
        check("loo", result["max_loo_shift_m"] * 1e3 <= lim["max_loo_shift_mm"],
              f"leave-one-out shift {result['max_loo_shift_m'] * 1e3:.2f} mm (max {lim['max_loo_shift_mm']})")
    if result.get("axis_spread_deg") is not None:
        check("axis", result["axis_spread_deg"] <= lim["max_axis_spread_deg"],
              f"axis spread {result['axis_spread_deg']:.2f}° (max {lim['max_axis_spread_deg']})")
    failed = [c for c in checks if not c["passed"]]
    return dict(passed=not failed, checks=checks, limits=lim,
                summary="all checks passed" if not failed else "; ".join(c["detail"] for c in failed))
