import numpy as np
from scipy.spatial.transform import Rotation

from hand_eye_calibration import depth_sweep as ds


def fk(q):
    """Free-flying camera: q = position + rotation vector."""
    m = np.eye(4)
    m[:3, :3] = Rotation.from_rotvec(q[3:]).as_matrix()
    m[:3, 3] = q[:3]
    return m


def board():
    m = np.eye(4)
    m[:3, 3] = [0.5, 0.0, 0.0]
    return m  # board on the table, normal +z


CENTER = np.array([0.075, 0.1, 0.0])
CAM_NOW = fk(np.r_[0.575, 0.1, 0.45, np.pi, 0, 0])  # straight above, looking down


def test_targets_have_requested_incidence_and_distance():
    targets = ds.look_at_targets(board(), CENTER, CAM_NOW, incidences=(30.0,), azimuths=(0.0, 90.0), distances=(0.4,))
    c = (board() @ np.r_[CENTER, 1])[:3]
    for t in targets:
        v = t["position"] - c
        assert abs(np.linalg.norm(v) - 0.4) < 1e-9
        assert abs(np.degrees(np.arccos(v[2] / np.linalg.norm(v))) - 30.0) < 1e-6


def test_ik_aims_camera_at_board():
    t = ds.look_at_targets(board(), CENTER, CAM_NOW, incidences=(35.0,), azimuths=(90.0,), distances=(0.4,))[0]
    q0 = np.r_[0.575, 0.1, 0.45, np.pi, 0, 0]
    sol = ds.solve_camera_ik(fk, q0, t["position"], t["look_at"], np.full(6, -4.0), np.full(6, 4.0))
    assert sol is not None
    q, perr, aerr = sol
    assert perr < 1e-3 and aerr < 0.5


def test_select_spreads_incidence_then_azimuth():
    cands = [{"incidence_deg": i, "azimuth_deg": a, "distance_m": d, "cost": 0.1}
             for i in (15, 35, 55) for a in (0, 90, 180) for d in (0.3, 0.4)]
    chosen = ds.select_views(cands, 6)
    assert sorted(c["incidence_deg"] for c in chosen) == [15, 15, 35, 35, 55, 55]
    for inc in (15, 35, 55):
        az = [c["azimuth_deg"] for c in chosen if c["incidence_deg"] == inc]
        assert len(set(az)) == 2


def test_order_by_travel_starts_nearest():
    views = [{"q": np.array([2.0])}, {"q": np.array([0.5])}, {"q": np.array([1.0])}]
    assert [v["q"][0] for v in ds.order_by_travel(views, np.array([0.0]))] == [0.5, 1.0, 2.0]
