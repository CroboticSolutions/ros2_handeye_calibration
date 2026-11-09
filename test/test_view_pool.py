"""View pool for automatic hand-eye acquisition: ssik on PiPER and CRX, shared limits."""
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip('ssik')
from hand_eye_calibration.view_pool import (  # noqa: E402
    DISTANCE_SCALE, JOINT_LIMIT_MARGIN_RAD, MIN_ELEVATION_DEG, WRIST_CLEAR_RAD, AnalyticIKViews, ViewLimits,
    joint_window_bounds, orbit_camera_poses, select_view, shortest_route, tilt_and_roll, weakest_observability)
from hand_eye_calibration.visibility import BoardFraming, CameraKinematics  # noqa: E402

URDF = (Path(__file__).resolve().parents[3] / 'robots/piper_ros/src/robot_description/'
        'piper_description/urdf/piper_description.urdf')
NAMES = [f'joint{i}' for i in range(1, 7)]
# Scene of the D435 run 20261013_095104: board in base_link, link6 -> optical (HE-v9).
BOARD = np.array([[0.9988, 0.0294, 0.0382, 0.3449], [0.03, -0.9994, -0.0153, 0.0955],
                  [0.0377, 0.0164, -0.9992, 0.0124], [0, 0, 0, 1.]])
MOUNT = np.array([[0.00405, 0.938, 0.34662, -0.09158], [-0.99993, -4e-05, 0.01178, 0.03203],
                  [0.01106, -0.34664, 0.93793, 0.02248], [0, 0, 0, 1.]])
INITIAL = np.array([-0.1088, 1.0473, -1.0635, 0.0546, 0.9561, -0.168])
CAMERA = NS(k=[922.785, 0, 639.465, 0, 923.031, 376.727, 0, 0, 1], d=[0.] * 5, width=1280, height=720,
            distortion_model='plumb_bob')
SPEC = {'squares_x': 6, 'squares_y': 8, 'square_length_m': 0.025}
LIMITS = ViewLimits()


def gram(rotations):
    g = np.zeros((3, 3))
    for a, b in itertools.combinations(rotations, 2):
        m = a.T @ b - np.eye(3)
        g += m.T @ m
    return g


def chain_limits(root, names):
    limits = [root.find(f"joint[@name='{n}']/limit") for n in names]
    return (np.array([float(x.get('lower')) for x in limits]), np.array([float(x.get('upper')) for x in limits]))


@pytest.fixture(scope='module')
def scene():
    urdf = URDF.read_text()
    root = ET.fromstring(urdf)
    fk = CameraKinematics(root, 'base_link', 'link6', NAMES, np.eye(4))
    lower, upper = chain_limits(root, NAMES)
    planner = AnalyticIKViews.create(urdf, 'base_link', 'link6', NAMES, lower, upper, fk.camera, LIMITS)
    framing = BoardFraming(SPEC, CAMERA)
    center = (BOARD @ np.r_[framing.center, 1])[:3]
    reference = fk.camera(INITIAL) @ MOUNT
    pool, sampled = planner.pool(BOARD, center, MOUNT, framing, reference, 0.30, INITIAL,
                                 np.random.default_rng(3), 60)
    return NS(planner=planner, fk=fk, lower=lower, upper=upper, framing=framing, reference=reference,
              center=center, pool=pool, sampled=sampled)


def test_create_rejects_chains_it_cannot_solve(scene):
    assert scene.planner is not None
    urdf = URDF.read_text()
    assert AnalyticIKViews.create(urdf, 'base_link', 'link5', NAMES[:5], scene.lower[:5], scene.upper[:5],
                                  scene.fk.camera) is None      # camera before the last joint
    assert AnalyticIKViews.create(None, 'base_link', 'link6', NAMES, scene.lower, scene.upper,
                                  scene.fk.camera) is None
    assert AnalyticIKViews.create(urdf, 'base_link', 'link6', NAMES, scene.lower, scene.upper,
                                  lambda q: np.eye(4)) is None  # FK disagrees with the runner's model


def test_pool_views_are_exact_framed_and_inside_every_limit(scene):
    assert len(scene.pool) >= 60
    lo, hi = joint_window_bounds(INITIAL, scene.lower + JOINT_LIMIT_MARGIN_RAD,
                                 scene.upper - JOINT_LIMIT_MARGIN_RAD, LIMITS)
    distance = np.linalg.norm(scene.reference[:3, 3] - scene.center)
    normal = -BOARD[:3, 2]   # board z points down here
    for q, effector in scene.pool:
        assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)
        assert np.all(np.abs(q - INITIAL)[:3] <= np.deg2rad(LIMITS.max_arm_joint_change_deg) + 1e-9)
        assert np.all(np.abs(q - INITIAL)[3:] <= np.deg2rad(LIMITS.max_wrist_joint_change_deg) + 1e-9)
        assert np.sign(q[4]) == np.sign(INITIAL[4]) and abs(q[4]) >= WRIST_CLEAR_RAD - 1e-9
        actual = scene.fk.camera(q)
        assert np.linalg.norm(actual[:3, 3] - effector[:3, 3]) < 1e-6
        assert Rotation.from_matrix(actual[:3, :3].T @ effector[:3, :3]).magnitude() < 1e-5
        camera = effector @ MOUNT
        assert scene.framing.contains(np.linalg.inv(camera) @ BOARD, margin=.13)
        assert np.linalg.norm(camera[:3, 3] - scene.reference[:3, 3]) <= 0.30
        offset = camera[:3, 3] - scene.center
        assert DISTANCE_SCALE[0] * distance - 1e-9 <= np.linalg.norm(offset) <= DISTANCE_SCALE[1] * distance + 1e-9
        assert offset @ normal / np.linalg.norm(offset) >= np.sin(np.deg2rad(MIN_ELEVATION_DEG)) - 1e-9
        tilt, roll = tilt_and_roll(scene.reference, camera[None])
        assert tilt[0] <= np.deg2rad(LIMITS.max_tilt_deg) + 1e-9
        assert abs(roll[0]) <= np.deg2rad(LIMITS.max_roll_deg) + 1e-9


def test_greedy_views_raise_the_weakest_translation_direction(scene):
    pool = list(scene.pool)
    chosen = [scene.fk.camera(INITIAL)[:3, :3]]
    for _ in range(20):
        chosen.append(pool.pop(select_view(pool, chosen))[1][:3, :3])
    rng = np.random.default_rng(0)
    random_sets = [[scene.pool[i][1][:3, :3] for i in rng.choice(len(scene.pool), 20, replace=False)]
                   for _ in range(20)]
    greedy = weakest_observability(gram(chosen))
    assert greedy > max(weakest_observability(gram([chosen[0]] + s)) for s in random_sets)
    relative = [np.degrees(Rotation.from_matrix(a.T @ b).magnitude()) for a, b in itertools.combinations(chosen, 2)]
    assert np.median(relative) > 20   # local sampler: 13.7 deg


def test_selection_tolerance_prefers_the_smaller_move():
    rotations = [np.eye(3), Rotation.from_euler('y', 40, degrees=True).as_matrix()]
    big = Rotation.from_euler('x', 40, degrees=True).as_matrix()     # weakest 0.698
    near = Rotation.from_euler('x', 37, degrees=True).as_matrix()    # weakest 0.669 (-4 %)
    pose = lambda rotation: np.block([[rotation, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
    pool = [(np.full(6, 1.0), pose(big)), (np.full(6, 0.2), pose(near))]
    assert select_view(pool, rotations) == 0                                  # pure information
    assert select_view(pool, rotations, np.zeros(6), 0.10) == 1               # within 10 %: smaller move
    assert select_view(pool, rotations, np.zeros(6), 0.01) == 0               # outside the tolerance
    assert select_view([], rotations) is None


def test_tilt_and_roll_of_known_rotations():
    reference = np.eye(4)
    reference[:3, :3] = Rotation.from_euler('zyx', [30, -20, 110], degrees=True).as_matrix()
    poses = []
    for tilt, roll in ((0, 0), (25, 0), (0, 40), (30, -35)):
        p = reference.copy()
        swing = Rotation.from_rotvec(np.deg2rad(tilt) * np.array([0.6, 0.8, 0.]))
        p[:3, :3] = reference[:3, :3] @ (swing * Rotation.from_euler('z', roll, degrees=True)).as_matrix()
        poses.append(p)
    tilt, roll = tilt_and_roll(reference, np.array(poses))
    np.testing.assert_allclose(np.degrees(tilt), [0, 25, 0, 30], atol=1e-4)
    np.testing.assert_allclose(np.degrees(roll), [0, 0, 40, -35], atol=1e-6)


def test_orbit_poses_keep_the_board_centre_on_the_line_of_sight():
    reference = np.eye(4)
    reference[:3, 3] = [0.3, 0.1, 0.7]
    center = np.array([0.35, 0.05, 0.0])
    z = center - reference[:3, 3]
    z /= np.linalg.norm(z)
    x = np.cross([0, 1., 0], z)
    x /= np.linalg.norm(x)
    reference[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    poses = orbit_camera_poses(reference, center, np.random.default_rng(1), 500, LIMITS)
    tilt, roll = tilt_and_roll(reference, poses)
    assert np.degrees(tilt).max() <= LIMITS.max_tilt_deg + 1e-6
    assert np.degrees(np.abs(roll)).max() <= LIMITS.max_roll_deg + 1e-6
    assert np.degrees(tilt).max() > 0.8 * LIMITS.max_tilt_deg   # the range is used
    to_center = center - poses[:, :3, 3]
    to_center /= np.linalg.norm(to_center, axis=1)[:, None]
    np.testing.assert_allclose(np.einsum('ni,ni->n', to_center, poses[:, :3, 2]), 1.0, atol=1e-9)


def test_sample_main_uses_pool_then_falls_back(monkeypatch):
    from hand_eye_calibration import intrinsic_acquisition as acquisition
    s = acquisition.IntrinsicSession.__new__(acquisition.IntrinsicSession)
    s.views, s.validation, s.rejections, s.pool_left = [], [], {}, {}
    s.initial, s.estimate, s.board, s.rng = np.zeros(6), np.eye(4), np.eye(4), np.random.default_rng(0)
    s.underactuated, s.n, s.limits = False, NS(), LIMITS
    s.fk = NS(camera=lambda q: np.eye(4))
    updates, fields = [], {}
    s.r = NS(check=lambda: None, joint_positions=lambda: np.zeros(6), lower=np.full(6, -3.), upper=np.full(6, 3.),
             update=lambda *a, **kw: updates.append(a[1] if len(a) > 1 else '') or fields.update(kw),
             framing=NS(center=np.zeros(3), contains=lambda *a, **kw: True), max_camera_excursion=.3)
    poses = [np.eye(4) for _ in range(2)]
    for i, p in enumerate(poses):
        p[:3, :3] = Rotation.from_euler('x', 20 * (i + 1), degrees=True).as_matrix()
    s.view_pool = NS(pool=lambda *a, **kw: ([(np.full(6, .1 * i), p) for i, p in enumerate(poses)], 5000))
    collected = []

    def collect(target, phase):
        collected.append((tuple(target), phase))
        s.views.append(np.eye(4) if len(collected) > 2 else poses[int(round(target[0] * 10))])
        return True
    s.collect = collect
    fallback = []
    monkeypatch.setattr(acquisition, 'random_camera_pose', lambda *a: fallback.append(1) or np.eye(4))
    local = iter([np.full(6, 2.0), np.full(6, .3)])   # the first is outside the joint window
    s.ik = lambda pose, nearest=False: next(local)
    s.duplicate = lambda actual: False
    s.sample_main(3, 'refinement')
    assert sorted(c[0][0] for c in collected[:2]) == pytest.approx([0., .1])   # both pool views first
    assert len(collected) == 3 and collected[2][0][0] == .3 and len(fallback) == 2  # then the local sampler
    assert s.rejections['joint_window'] == 1
    assert any('View pool exhausted at 2/3' in u for u in updates)
    assert 'View pool exhausted at 2/3' in fields['view_sampler_warning']   # GUI alert
    assert fields['view_sampler'] == 'local random'


def _session_runner(monkeypatch):
    from hand_eye_calibration import intrinsic_acquisition as acquisition
    from test_automatic_calibration import synthetic_sequence
    r, n, _ = synthetic_sequence(monkeypatch)
    r.names = n.get_parameter('auto_joint_names').value
    chain = [(np.eye(4), np.array([0., 1., 0.]), 'revolute', name) for name in r.names]
    monkeypatch.setattr(acquisition, 'CameraKinematics',
                        lambda *a: NS(chain=chain, camera=r.fake_kinematics(*a).camera))
    r.lower, r.upper = np.full(6, -3.), np.full(6, 3.)
    r.robot_description = '<robot/>'
    return acquisition, r


def test_session_uses_analytic_views_with_the_runner_limits(monkeypatch):
    acquisition, r = _session_runner(monkeypatch)
    seen = {}
    monkeypatch.setattr(AnalyticIKViews, 'create',
                        classmethod(lambda cls, *a, **kw: seen.setdefault('limits', a[-1]) and 'pool'))
    r.view_limits = ViewLimits(max_tilt_deg=20.)
    assert acquisition.IntrinsicSession(r, None).view_pool == 'pool'
    assert seen['limits'].max_tilt_deg == 20.


def test_session_refuses_to_move_without_ssik(monkeypatch):
    """No silent local sampler for a six-joint chain (CRX 21.10.: sigma 2.7-6.7 mm)."""
    acquisition, r = _session_runner(monkeypatch)
    monkeypatch.setattr(AnalyticIKViews, 'create', classmethod(lambda cls, *a, **kw: None))
    with pytest.raises(ValueError, match='ssik'):
        acquisition.IntrinsicSession(r, None)


def test_views_are_added_while_sigma_is_above_its_limit():
    from hand_eye_calibration import intrinsic_acquisition as acquisition
    s = acquisition.IntrinsicSession.__new__(acquisition.IntrinsicSession)
    s.main_count, s.max_main_count, s.initial_count = 20, 26, 8
    updates, sampled = [], []
    s.r = NS(max_position_sigma=.002, validation_views=5, update=lambda *a, **kw: updates.append(kw))
    s.sample_main = lambda count, phase, reuse=False: sampled.append((count, phase, reuse))
    sigmas = iter([.003, .0015])
    s.fit = lambda full=False: {'uncertainty': {'worst_direction_sigma_m': next(sigmas)}}
    report = s.add_views_for_sigma({'uncertainty': {'worst_direction_sigma_m': .004}})
    assert sampled == [(2, 'refinement', True), (2, 'refinement', True)]
    assert report['uncertainty']['worst_direction_sigma_m'] == .0015
    assert updates[-1]['max_training_samples'] == 34
    # Never beyond auto_max_training_samples main views.
    sampled.clear()
    s.fit = lambda full=False: {'uncertainty': {'worst_direction_sigma_m': .005}}
    report = s.add_views_for_sigma({'uncertainty': {'worst_direction_sigma_m': .005}})
    assert sum(count for count, *_ in sampled) == 6
    # Already within the limit: nothing is added.
    sampled.clear()
    s.add_views_for_sigma({'uncertainty': {'worst_direction_sigma_m': .001}})
    assert sampled == []


def test_shortest_route_keeps_views_and_shortens_travel():
    rng = np.random.default_rng(4)
    entries = [(rng.uniform(-1, 1, 6), np.eye(4)) for _ in range(20)]
    start = np.zeros(6)
    route = shortest_route(start, entries)
    assert sorted(id(e) for e in route) == sorted(id(e) for e in entries)   # same views, reordered

    def travel(order):
        points = [start] + [q for q, _ in order]
        return sum(np.max(np.abs(b - a)) for a, b in zip(points, points[1:]))
    assert travel(route) < 0.8 * travel(entries)   # random 6-D views: ~29 % shorter
    # A line of views is visited in order from the nearer end.
    line = [(np.full(6, x), np.eye(4)) for x in (.3, .1, .4, .2)]
    assert [q[0] for q, _ in shortest_route(np.zeros(6), line)] == [.1, .2, .3, .4]
    assert shortest_route(start, entries[:1]) == entries[:1]


def test_selected_views_are_visited_along_the_shortest_route():
    from hand_eye_calibration import intrinsic_acquisition as acquisition
    s = acquisition.IntrinsicSession.__new__(acquisition.IntrinsicSession)
    s.views, s.validation, s.rejections, s.pool_left = [], [], {}, {}
    s.initial, s.estimate, s.board, s.rng = np.zeros(6), np.eye(4), np.eye(4), np.random.default_rng(0)
    s.n, s.limits = NS(), LIMITS
    s.r = NS(check=lambda: None, joint_positions=lambda: np.zeros(6), update=lambda *a, **kw: None,
             framing=NS(center=np.zeros(3)), max_camera_excursion=.3)
    poses = []
    for axis, angle in (('x', 40), ('y', 35), ('z', 30)):
        p = np.eye(4)
        p[:3, :3] = Rotation.from_euler(axis, angle, degrees=True).as_matrix()
        poses.append(p)
    # Joint distance from zero grows with the index; information order differs.
    pool = [(np.full(6, .1 * (i + 1)), p) for i, p in enumerate(poses)]
    s.view_pool = NS(pool=lambda *a, **kw: (list(pool), 100))
    visited = []
    s.duplicate = lambda actual: False
    s.collected_or_failed = lambda target, phase, failures: (visited.append(target[0]) or True, 0)
    assert s.sample_pool(3, 'refinement', np.eye(4), np.zeros(3)) == 3
    assert visited == pytest.approx([.1, .2, .3])


def test_joint_window_keeps_the_wrist_side_and_the_limit_margin():
    lower, upper = np.full(6, -3.), np.full(6, 3.)
    window = LIMITS.joint_window(6)
    np.testing.assert_allclose(np.degrees(window), [45, 45, 45, 60, 60, 60])
    lo, hi = joint_window_bounds(np.array([0., 0., 0., 0., -0.8, 2.95]), lower, upper, LIMITS)
    assert hi[4] == pytest.approx(-WRIST_CLEAR_RAD)            # J5 stays negative
    assert lo[4] == pytest.approx(-0.8 - window[4])
    assert hi[5] == pytest.approx(3.)                          # the (margined) limit wins
    lo, hi = joint_window_bounds(np.array([0., 0., 0., 0., 0.05, 0.]), lower, upper, LIMITS)
    assert lo[4] < 0 < hi[4]                                    # near 0: no side imposed
    lo, hi = joint_window_bounds(np.array([0., 0., 0., 0., 0.8, 0.]), lower, upper, LIMITS)
    assert lo[4] == pytest.approx(WRIST_CLEAR_RAD)
    active = np.array([True] * 5 + [False])
    lo, hi = joint_window_bounds(np.full(6, .5), lower, upper, LIMITS, active)
    assert lo[5] == hi[5] == .5                                 # a joint after the camera stays put


CRX_URDF = Path(__file__).resolve().parents[3] / 'perception/hand_teleop/tests/data/crx10ia_l.urdf'
CRX_NAMES = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']


def test_analytic_views_on_the_crx_give_exact_in_window_solutions():
    urdf = CRX_URDF.read_text()
    root = ET.fromstring(urdf)
    fk = CameraKinematics(root, 'base_link', 'flange', CRX_NAMES, np.eye(4))
    lower, upper = chain_limits(root, CRX_NAMES)
    views = AnalyticIKViews.create(urdf, 'base_link', 'flange', CRX_NAMES, lower, upper, fk.camera)
    assert views is not None
    q0 = np.array([0., .3, -.4, 0., -.9, 0.])
    lo, hi = joint_window_bounds(q0, views.lower, views.upper, views.limits)
    step = np.r_[.2, -.1, .1, .3, .2, -.4]
    target = fk.camera(q0 + step)
    solutions = views.analytic.ik(target)
    assert solutions
    for q in solutions:
        np.testing.assert_allclose(fk.camera(q), target, atol=1e-6)
    inside = [q for q in solutions if np.all(q >= lo) and np.all(q <= hi)]
    assert len(inside) == 1                              # the window holds one branch here
    np.testing.assert_allclose(inside[0], q0 + step, atol=1e-6)
    # Another chain (or a different effector FK) is refused.
    assert AnalyticIKViews.create(urdf, 'base_link', 'flange', CRX_NAMES, lower, upper,
                                  lambda q: np.eye(4)) is None
