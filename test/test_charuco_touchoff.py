"""ChArUco touch-off TCP math."""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from hand_eye_calibration.charuco_touchoff import acceptance, board_targets, make_board, solve

T_TRUE = np.array([0.0377, -0.0015, 0.0985])
AXIS_TRUE = np.array([0.35, 0.0, 0.937]) / np.linalg.norm([0.35, 0.0, 0.937])


def _touch(target, yaw_deg, normal_up=np.array([0, 0, 1.0]), noise=0.0, rng=None):
    """Flange pose that puts the tip on target with the tool axis along -normal."""
    # rotation taking AXIS_TRUE (flange) to -normal (base), then spin about the normal
    a, b = AXIS_TRUE, -normal_up
    v = np.cross(a, b)
    R0 = Rot.from_rotvec(v / np.linalg.norm(v) * np.arcsin(np.linalg.norm(v)) if np.dot(a, b) > 0
                         else v / np.linalg.norm(v) * (np.pi - np.arcsin(np.linalg.norm(v))))
    R = Rot.from_rotvec(np.radians(yaw_deg) * normal_up) * R0
    q = np.asarray(target) - R.apply(T_TRUE)
    if rng is not None:
        q = q + rng.normal(0, noise, 3)
    return dict(pose=list(q) + list(R.as_quat()), target=list(target))


def test_board_targets_ids_and_spacing() -> None:
    board = make_board(6, 8, 0.025, 0.018, "DICT_4X4_100")
    T = np.eye(4); T[:3, 3] = [0.4, -0.05, 0.0]
    tg = board_targets(board, T)
    assert len(tg["ids"]) == 35
    d = np.linalg.norm(tg["points_base"][1] - tg["points_base"][0])
    assert abs(d - 0.025) < 1e-9


def test_exact_touches_recover_tcp_and_axis() -> None:
    targets = [[0.40, -0.05, 0.0], [0.45, -0.05, 0.0], [0.40, 0.02, 0.0], [0.47, 0.03, 0.0]]
    samples = [dict(_touch(p, yaw), id=i) for i, (p, yaw) in enumerate(zip(targets, (0, 40, -35, 80)))]
    r = solve(samples, normal_up=[0, 0, 1])
    assert np.allclose(r["tcp_translation"], T_TRUE, atol=1e-9)
    assert r["rms_residual_m"] < 1e-9
    assert np.allclose(r["axis_dir"], AXIS_TRUE, atol=1e-9)
    assert acceptance(r)["passed"]


def test_noise_shows_in_residuals_and_gate() -> None:
    rng = np.random.default_rng(0)
    targets = [[0.40, -0.05, 0.0], [0.45, -0.05, 0.0], [0.40, 0.02, 0.0], [0.47, 0.03, 0.0], [0.43, 0.0, 0.0]]
    samples = [dict(_touch(p, 20 * i, noise=0.0005, rng=rng), id=i) for i, p in enumerate(targets)]
    r = solve(samples, normal_up=[0, 0, 1])
    assert np.linalg.norm(np.array(r["tcp_translation"]) - T_TRUE) < 0.001
    assert 0.0002 < r["rms_residual_m"] < 0.0015
    bad = [dict(s) for s in samples]
    bad[0] = dict(_touch([0.40, -0.05, 0.0], 0), id=0); bad[0]["target"] = [0.41, -0.05, 0.0]  # wrong corner
    gate = acceptance(solve(bad, normal_up=[0, 0, 1]))
    assert not gate["passed"]


def test_gate_needs_four_distinct_ids() -> None:
    samples = [dict(_touch([0.40, -0.05, 0.0], y), id=3) for y in (0, 30, 60, 90)]
    gate = acceptance(solve(samples, normal_up=[0, 0, 1]))
    assert not gate["passed"] and "distinct" in gate["summary"]
