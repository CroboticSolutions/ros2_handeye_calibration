"""Board stability check of an applied hand-eye calibration.

The board lies still while the robot looks at it from several poses. Each view
gives the board pose in the robot base through the calibration under test
(camera TF x ChArUco pose). With a correct hand-eye every view puts the board
in the same place; the spread of the board corners and of the board rotation
across views is the validation result. It needs no tool and no TCP, so it
isolates the hand-eye (plus robot kinematics and camera) from the TCP.

Limits are informative only; they never gate an Apply.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Informative targets (validation research 2026-10-09): sub-mm and sub-degree
# board scatter for a small arm; max about twice the RMS.
POSITION_RMS_LIMIT_M = 0.001
POSITION_MAX_LIMIT_M = 0.002
ROTATION_MAX_LIMIT_DEG = 0.5
MIN_VIEWS = 3
MIN_COMMON_CORNERS = 4


def _mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """Chordal L2 mean of rotation matrices (SVD projection of the average)."""
    u, _, vt = np.linalg.svd(np.mean(rotations, axis=0))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def _angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def board_consistency(views: list[dict[str, Any]]) -> dict[str, Any]:
    """Spread of the board in the robot base across views.

    Each view: ``board_base`` (4x4 board pose in the base) and ``corners``
    ({id: board-frame xyz}). Corners seen in every view are compared, so an
    extrapolated corner outside the image never enters the metric.
    """
    result: dict[str, Any] = {"views": len(views), "min_views": MIN_VIEWS}
    if len(views) < MIN_VIEWS:
        result["message"] = f"Need at least {MIN_VIEWS} views ({len(views)} so far)."
        return result
    common = set(views[0]["corners"])
    for view in views[1:]:
        common &= set(view["corners"])
    ids = sorted(common)
    if len(ids) < MIN_COMMON_CORNERS:
        result["message"] = (f"Only {len(ids)} corners are seen in every view (need {MIN_COMMON_CORNERS}); "
                             "keep the whole board in view.")
        return result
    local = np.array([np.r_[np.asarray(views[0]["corners"][i], dtype=float), 1.0] for i in ids])
    points = np.array([(np.asarray(v["board_base"], dtype=float) @ local.T).T[:, :3] for v in views])
    mean_points = points.mean(axis=0)
    deviation = np.linalg.norm(points - mean_points, axis=2)  # views x corners
    per_view_mean = deviation.mean(axis=1)
    rotations = [np.asarray(v["board_base"], dtype=float)[:3, :3] for v in views]
    mean_r = _mean_rotation(rotations)
    per_view_rot = [_angle_deg(mean_r.T @ r) for r in rotations]
    centroid_offsets = points.mean(axis=1) - mean_points.mean(axis=0)
    rms = float(np.sqrt(np.mean(deviation ** 2)))
    worst = float(deviation.max())
    rot_max = float(max(per_view_rot))
    result.update(
        common_corners=len(ids),
        position_rms_m=rms,
        position_max_m=worst,
        rotation_max_deg=rot_max,
        rotation_rms_deg=float(np.sqrt(np.mean(np.square(per_view_rot)))),
        # Spread of the board centre per base axis: a single dominant axis
        # (often Z) points at a depth/scale or camera-rotation error.
        centroid_std_m=[float(v) for v in centroid_offsets.std(axis=0)],
        per_view=[{"mean_deviation_m": float(d), "rotation_deg": float(a),
                   "centroid_offset_m": [float(x) for x in c]}
                  for d, a, c in zip(per_view_mean, per_view_rot, centroid_offsets)],
        limits={"position_rms_m": POSITION_RMS_LIMIT_M, "position_max_m": POSITION_MAX_LIMIT_M,
                "rotation_max_deg": ROTATION_MAX_LIMIT_DEG},
        passed=bool(rms <= POSITION_RMS_LIMIT_M and worst <= POSITION_MAX_LIMIT_M
                    and rot_max <= ROTATION_MAX_LIMIT_DEG),
    )
    return result


def summarize(views_by_source: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Status payload: one consistency result per calibration source."""
    return {
        "views": max((len(v) for v in views_by_source.values()), default=0),
        "sources": {name: board_consistency(views) for name, views in views_by_source.items() if views},
    }
