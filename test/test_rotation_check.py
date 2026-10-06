"""Rotation check of the applied TCP: targets and deviation."""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from hand_eye_calibration.rotation_check_node import rotation_targets, tip_deviation

TCP = np.array([0.0377, -0.0015, 0.0985])


def test_targets_rotate_about_tip_and_return() -> None:
    ref = Rot.from_euler("xyz", [180, 10, 30], degrees=True).as_quat()
    targets, labels = rotation_targets(ref, 20, 45)
    assert len(targets) == 7 and labels[-1] == "back to start"
    angles = [np.degrees((Rot.from_quat(q) * Rot.from_quat(ref).inv()).magnitude()) for q in targets]
    assert np.allclose(angles[:4], 20, atol=1e-6) and np.allclose(angles[4:6], 45, atol=1e-6)
    assert angles[6] < 1e-6


def test_wrong_applied_tcp_shows_as_deviation_after_rotation() -> None:
    """True tip touched to the point in every pose; deviation uses the applied TCP."""
    pivot = np.array([0.45, -0.03, 0.002])
    ref = Rot.from_euler("xyz", [180, 10, 30], degrees=True).as_quat()
    targets, _ = rotation_targets(ref, 20, 45)

    def touch(q):  # flange pose with the TRUE tip on the pivot
        return np.r_[pivot - Rot.from_quat(q).apply(TCP), q]

    # correct applied TCP: zero everywhere
    assert max(tip_deviation(touch(q), TCP, pivot)[0] for q in targets) < 1e-9
    # applied TCP 3 mm off: start point is where the wrong tip was, rotations reveal it
    wrong = TCP + np.array([0.0, 0.0, 0.003])
    start = touch(ref)
    start_tip = start[:3] + Rot.from_quat(ref).apply(wrong)
    devs = [tip_deviation(touch(q), wrong, start_tip)[0] for q in targets]
    assert max(devs) > 0.0009          # ~2*3 mm*sin(20°/2) for the tilts
    assert devs[-1] < 1e-9             # back at the start orientation
