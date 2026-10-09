import numpy as np
from scipy.spatial.transform import Rotation as Rot

from hand_eye_calibration import board_check


def _corners():
    return {i: [0.025 * (i % 5), 0.025 * (i // 5), 0.0] for i in range(35)}


def _view(t, rotvec=(0.0, 0.0, 0.0), corners=None):
    m = np.eye(4)
    m[:3, :3] = Rot.from_rotvec(rotvec).as_matrix()
    m[:3, 3] = t
    return {"board_base": m, "corners": corners or _corners(), "stamp_ns": 0}


def test_identical_views_pass_with_zero_spread():
    res = board_check.board_consistency([_view([0.4, 0.0, 0.0])] * 4)
    assert res["passed"] is True
    assert res["position_rms_m"] < 1e-12
    assert res["rotation_max_deg"] < 1e-6
    assert res["common_corners"] == 35


def test_translation_spread_is_reported_and_fails_limit():
    views = [_view([0.4, 0.0, 0.0]), _view([0.4, 0.0, 0.003]), _view([0.4, 0.0, -0.003])]
    res = board_check.board_consistency(views)
    assert abs(res["position_max_m"] - 0.003) < 1e-9
    assert res["centroid_std_m"][2] > res["centroid_std_m"][0]
    assert res["passed"] is False


def test_rotation_spread_in_degrees():
    views = [_view([0.4, 0, 0]), _view([0.4, 0, 0], (0, 0, np.radians(1.0))), _view([0.4, 0, 0])]
    res = board_check.board_consistency(views)
    assert 0.6 < res["rotation_max_deg"] < 0.7  # 1° view is 2/3° from the mean
    assert res["passed"] is False


def test_only_corners_seen_in_every_view_count():
    partial = {i: c for i, c in _corners().items() if i < 10}
    res = board_check.board_consistency([_view([0.4, 0, 0]), _view([0.4, 0, 0], corners=partial), _view([0.4, 0, 0])])
    assert res["common_corners"] == 10


def test_needs_three_views_and_four_common_corners():
    assert "message" in board_check.board_consistency([_view([0, 0, 0])] * 2)
    few = {i: c for i, c in _corners().items() if i < 3}
    assert "message" in board_check.board_consistency([_view([0, 0, 0], corners=few)] * 3)


def test_summarize_per_source():
    out = board_check.summarize({"applied_tf": [_view([0.4, 0, 0])] * 3, "candidate": []})
    assert out["views"] == 3
    assert list(out["sources"]) == ["applied_tf"]
