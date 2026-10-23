"""Closed-form IK view pool for intrinsic acquisition (PiPER URDF, D435 1280x720)."""
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip('hand_teleop.closed_form_ik')
from hand_eye_calibration.closed_form_views import (  # noqa: E402
    JOINT_LIMIT_MARGIN_RAD, NUMERIC_DISTANCE_SCALE, NUMERIC_JOINT_WINDOW_RAD, WRIST_CLEAR_RAD,
    AnalyticIKViews, ClosedFormViews, NumericIKViews, shortest_route, weakest_observability)
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


def gram(rotations):
    g = np.zeros((3, 3))
    for a, b in itertools.combinations(rotations, 2):
        m = a.T @ b - np.eye(3)
        g += m.T @ m
    return g


@pytest.fixture(scope='module')
def scene():
    urdf = URDF.read_text()
    root = ET.fromstring(urdf)
    fk = CameraKinematics(root, 'base_link', 'link6', NAMES, np.eye(4))
    limits = [root.find(f"joint[@name='{n}']/limit") for n in NAMES]
    lower = np.array([float(x.get('lower')) for x in limits])
    upper = np.array([float(x.get('upper')) for x in limits])
    planner = ClosedFormViews.create(urdf, 'base_link', 'link6', NAMES, lower, upper, fk.camera)
    framing = BoardFraming(SPEC, CAMERA)
    center = (BOARD @ np.r_[framing.center, 1])[:3]
    reference = fk.camera(INITIAL) @ MOUNT
    pool, sampled = planner.pool(BOARD, center, MOUNT, framing, reference, 0.30, INITIAL,
                                 np.random.default_rng(3), 60)
    return NS(planner=planner, fk=fk, lower=lower, upper=upper, framing=framing, reference=reference,
              pool=pool, sampled=sampled)


def test_create_rejects_chains_without_matching_closed_form(scene):
    assert scene.planner is not None
    urdf = URDF.read_text()
    assert ClosedFormViews.create(urdf, 'base_link', 'link5', NAMES[:5], scene.lower[:5], scene.upper[:5],
                                  scene.fk.camera) is None      # camera before the last joint
    assert ClosedFormViews.create(None, 'base_link', 'link6', NAMES, scene.lower, scene.upper,
                                  scene.fk.camera) is None
    assert ClosedFormViews.create(urdf, 'base_link', 'link6', NAMES, scene.lower, scene.upper,
                                  lambda q: np.eye(4)) is None  # FK disagrees with the runner's model


def test_pool_views_are_framed_reachable_and_clear_of_limits(scene):
    assert len(scene.pool) >= 60
    for q, effector in scene.pool:
        assert np.all(q >= scene.lower + JOINT_LIMIT_MARGIN_RAD - 1e-9)
        assert np.all(q <= scene.upper - JOINT_LIMIT_MARGIN_RAD + 1e-9)
        actual = scene.fk.camera(q)   # closed-form IK: < 1 um, ~1e-5 rad
        assert np.linalg.norm(actual[:3, 3] - effector[:3, 3]) < 1e-6
        assert Rotation.from_matrix(actual[:3, :3].T @ effector[:3, :3]).magnitude() < 1e-4
        camera = effector @ MOUNT
        assert scene.framing.contains(np.linalg.inv(camera) @ BOARD, margin=.13)
        assert np.linalg.norm(camera[:3, 3] - scene.reference[:3, 3]) <= 0.30


def test_greedy_views_raise_the_weakest_translation_direction(scene):
    pool = list(scene.pool)
    chosen = [scene.fk.camera(INITIAL)[:3, :3]]
    for _ in range(20):
        chosen.append(pool.pop(ClosedFormViews.best(pool, chosen))[1][:3, :3])
    rng = np.random.default_rng(0)
    random_sets = [[scene.pool[i][1][:3, :3] for i in rng.choice(len(scene.pool), 20, replace=False)]
                   for _ in range(20)]
    greedy = weakest_observability(gram(chosen))
    assert greedy > max(weakest_observability(gram([chosen[0]] + s)) for s in random_sets)
    relative = [np.degrees(Rotation.from_matrix(a.T @ b).magnitude()) for a, b in itertools.combinations(chosen, 2)]
    assert np.median(relative) > 20


def test_sample_main_uses_pool_then_falls_back(monkeypatch):
    from hand_eye_calibration import intrinsic_acquisition as acquisition
    s = acquisition.IntrinsicSession.__new__(acquisition.IntrinsicSession)
    s.views, s.validation, s.rejections, s.pool_left = [], [], {}, {}
    s.initial, s.estimate, s.board, s.rng = np.zeros(6), np.eye(4), np.eye(4), np.random.default_rng(0)
    s.underactuated, s.n = False, NS()
    s.fk = NS(camera=lambda q: np.eye(4))
    updates, fields = [], {}
    s.r = NS(check=lambda: None, joint_positions=lambda: np.zeros(6),
             update=lambda *a, **kw: updates.append(a[1] if len(a) > 1 else '') or fields.update(kw),
             framing=NS(center=np.zeros(3), contains=lambda *a, **kw: True), max_camera_excursion=.3)
    poses = [np.eye(4) for _ in range(2)]
    for i, p in enumerate(poses):
        p[:3, :3] = Rotation.from_euler('x', 20 * (i + 1), degrees=True).as_matrix()
    s.closed_form = NS(pool=lambda *a, **kw: ([(np.full(6, i), p) for i, p in enumerate(poses)], 5000))
    collected = []

    def collect(target, phase):
        collected.append((tuple(target), phase))
        s.views.append(np.eye(4) if len(collected) > 2 else poses[int(target[0])])
        return True
    s.collect = collect
    fallback = []
    monkeypatch.setattr(acquisition, 'random_camera_pose', lambda *a: fallback.append(1) or np.eye(4))
    s.ik = lambda pose, nearest=False: np.full(6, 9.)
    s.duplicate = lambda actual: False
    s.sample_main(3, 'refinement')
    assert [c[0][0] for c in collected[:2]] in ([0., 1.], [1., 0.])   # both pool views first
    assert len(collected) == 3 and collected[2][0][0] == 9. and fallback  # then the local sampler
    assert any('View pool exhausted at 2/3' in u for u in updates)
    assert 'View pool exhausted at 2/3' in fields['view_sampler_warning']   # GUI alert
    assert fields['view_sampler'] == 'local random'


def test_numeric_pool_matches_closed_form_quality(scene):
    """Numeric IK (any 6-joint URDF chain, CRX) on the PiPER scene: views are framed,
    reached exactly, clear of the limits, on the nearest branch, and as informative."""
    numeric = NumericIKViews(scene.fk.camera, scene.lower, scene.upper)
    center = (BOARD @ np.r_[scene.framing.center, 1])[:3]
    pool, sampled = numeric.pool(BOARD, center, MOUNT, scene.framing, scene.reference, 0.30, INITIAL,
                                 np.random.default_rng(3), 60)
    # One IK branch (nearest the return pose) and PiPER's tight limits: fewer than closed form.
    assert len(pool) >= 25 and sampled <= 5000
    distance = np.linalg.norm(scene.reference[:3, 3] - center)
    for q, effector in pool:
        assert np.all(q >= scene.lower + JOINT_LIMIT_MARGIN_RAD - 1e-9)
        assert np.all(q <= scene.upper - JOINT_LIMIT_MARGIN_RAD + 1e-9)
        actual = scene.fk.camera(q)   # IK tolerance: 1 mm, 0.5 deg
        assert np.linalg.norm(actual[:3, 3] - effector[:3, 3]) <= 1e-3 + 1e-9
        assert Rotation.from_matrix(effector[:3, :3].T @ actual[:3, :3]).magnitude() <= np.deg2rad(.5) + 1e-9
        camera = effector @ MOUNT
        assert scene.framing.contains(np.linalg.inv(camera) @ BOARD, margin=.13)
        assert np.linalg.norm(camera[:3, 3] - scene.reference[:3, 3]) <= 0.30 + 1e-9
        r = np.linalg.norm(camera[:3, 3] - center)
        assert NUMERIC_DISTANCE_SCALE[0] * distance - .05 <= r <= NUMERIC_DISTANCE_SCALE[1] * distance + .05
    chosen = [scene.fk.camera(INITIAL)[:3, :3]]
    for _ in range(20):
        chosen.append(pool.pop(ClosedFormViews.best(pool, chosen))[1][:3, :3])
    relative = [np.degrees(Rotation.from_matrix(a.T @ b).magnitude()) for a, b in itertools.combinations(chosen, 2)]
    assert np.median(relative) > 20   # local sampler: 13.7 deg


def test_session_uses_numeric_views_without_closed_form(monkeypatch):
    """CRX: closed_form_ik needs z joint axes, so the session must not fall back to local views."""
    from hand_eye_calibration import intrinsic_acquisition as acquisition
    from test_automatic_calibration import synthetic_sequence
    from hand_eye_calibration import bootstrap_calibration
    r, n, _ = synthetic_sequence(monkeypatch)
    r.names = n.get_parameter('auto_joint_names').value
    # A six-revolute URDF chain (visibility.CameraKinematics) without a closed form.
    chain = [(np.eye(4), np.array([0., 1., 0.]), 'revolute', name) for name in r.names]
    monkeypatch.setattr(acquisition, 'CameraKinematics',
                        lambda *a: NS(chain=chain, camera=bootstrap_calibration.CameraKinematics(*a).camera))
    monkeypatch.setattr(ClosedFormViews, 'create', classmethod(lambda cls, *a, **kw: None))
    r.lower, r.upper = np.full(6, -3.), np.full(6, 3.)
    r.robot_description = '<robot/>'
    s = acquisition.IntrinsicSession(r, None)
    assert isinstance(s.closed_form, NumericIKViews)


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
    s.n = NS()
    s.r = NS(check=lambda: None, joint_positions=lambda: np.zeros(6), update=lambda *a, **kw: None,
             framing=NS(center=np.zeros(3)), max_camera_excursion=.3)
    poses = []
    for axis, angle in (('x', 40), ('y', 35), ('z', 30)):
        p = np.eye(4)
        p[:3, :3] = Rotation.from_euler(axis, angle, degrees=True).as_matrix()
        poses.append(p)
    # Joint distance from zero grows with the index; information order differs.
    pool = [(np.full(6, .1 * (i + 1)), p) for i, p in enumerate(poses)]
    s.closed_form = NS(pool=lambda *a, **kw: (list(pool), 100))
    visited = []
    s.duplicate = lambda actual: False
    s.collected_or_failed = lambda target, phase, failures: (visited.append(target[0]) or True, 0)
    assert s.sample_closed_form(3, 'refinement', np.eye(4), np.zeros(3)) == 3
    assert visited == pytest.approx([.1, .2, .3])


def test_numeric_views_stay_in_the_joint_window_without_a_wrist_flip(scene):
    """The bounded solver alone wandered up to 185 deg on J6 and flipped J5 (CRX, 21.10.)."""
    numeric = NumericIKViews(scene.fk.camera, scene.lower, scene.upper)
    center = (BOARD @ np.r_[scene.framing.center, 1])[:3]
    pool, _ = numeric.pool(BOARD, center, MOUNT, scene.framing, scene.reference, 0.30, INITIAL,
                           np.random.default_rng(3), 60)
    assert pool
    for q, _ in pool:
        assert np.all(np.abs(q - INITIAL) <= NUMERIC_JOINT_WINDOW_RAD + 1e-9)
        assert np.sign(q[4]) == np.sign(INITIAL[4]) and abs(q[4]) >= WRIST_CLEAR_RAD - 1e-9


def test_window_bounds_keep_the_wrist_side_and_the_limit_margin():
    lower, upper = np.full(6, -3.), np.full(6, 3.)
    views = NumericIKViews(lambda q: np.eye(4), lower, upper)
    lo, hi = views.window_bounds(np.array([0., 0., 0., 0., -0.8, 2.95]))
    assert hi[4] == pytest.approx(-WRIST_CLEAR_RAD)            # J5 stays negative
    assert lo[4] == pytest.approx(-0.8 - NUMERIC_JOINT_WINDOW_RAD[4])
    assert hi[5] == pytest.approx(3. - JOINT_LIMIT_MARGIN_RAD)  # limit margin wins
    lo, hi = views.window_bounds(np.array([0., 0., 0., 0., 0.05, 0.]))
    assert lo[4] < 0 < hi[4]                                    # near 0: no side imposed
    lo, hi = views.window_bounds(np.array([0., 0., 0., 0., 0.8, 0.]))
    assert lo[4] == pytest.approx(WRIST_CLEAR_RAD)


CRX_URDF = Path(__file__).resolve().parents[3] / 'perception/hand_teleop/tests/data/crx10ia_l.urdf'
CRX_NAMES = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']


def test_analytic_views_on_the_crx_give_exact_in_window_solutions():
    pytest.importorskip('ssik')
    urdf = CRX_URDF.read_text()
    root = ET.fromstring(urdf)
    fk = CameraKinematics(root, 'base_link', 'flange', CRX_NAMES, np.eye(4))
    limits = [root.find(f"joint[@name='{n}']/limit") for n in CRX_NAMES]
    lower = np.array([float(x.get('lower')) for x in limits])
    upper = np.array([float(x.get('upper')) for x in limits])
    assert ClosedFormViews.create(urdf, 'base_link', 'flange', CRX_NAMES, lower, upper, fk.camera) is None
    views = AnalyticIKViews.create(urdf, 'base_link', 'flange', CRX_NAMES, lower, upper, fk.camera)
    assert views is not None
    q0 = np.array([0., .3, -.4, 0., -.9, 0.])
    views.lower, views.upper = views.window_bounds(q0)
    target = fk.camera(q0 + np.r_[.2, -.1, .1, .3, .2, -.4])
    solutions = views.solutions(target, q0)
    assert solutions
    for q in solutions:
        np.testing.assert_allclose(fk.camera(q), target, atol=1e-6)
    inside = [q for q in solutions if np.all(q >= views.lower) and np.all(q <= views.upper)]
    assert len(inside) == 1                              # the window holds one branch here
    np.testing.assert_allclose(inside[0], q0 + np.r_[.2, -.1, .1, .3, .2, -.4], atol=1e-6)
    # Another chain (or a different effector FK) is refused.
    assert AnalyticIKViews.create(urdf, 'base_link', 'flange', CRX_NAMES, lower, upper,
                                  lambda q: np.eye(4)) is None


def test_session_prefers_analytic_views_over_numeric(monkeypatch):
    from hand_eye_calibration import intrinsic_acquisition as acquisition
    from test_automatic_calibration import synthetic_sequence
    from hand_eye_calibration import bootstrap_calibration
    r, n, _ = synthetic_sequence(monkeypatch)
    r.names = n.get_parameter('auto_joint_names').value
    chain = [(np.eye(4), np.array([0., 1., 0.]), 'revolute', name) for name in r.names]
    monkeypatch.setattr(acquisition, 'CameraKinematics',
                        lambda *a: NS(chain=chain, camera=bootstrap_calibration.CameraKinematics(*a).camera))
    monkeypatch.setattr(ClosedFormViews, 'create', classmethod(lambda cls, *a, **kw: None))
    sentinel = object()
    monkeypatch.setattr(AnalyticIKViews, 'create', classmethod(lambda cls, *a, **kw: sentinel))
    r.lower, r.upper = np.full(6, -3.), np.full(6, 3.)
    r.robot_description = '<robot/>'
    assert acquisition.IntrinsicSession(r, None).closed_form is sentinel
