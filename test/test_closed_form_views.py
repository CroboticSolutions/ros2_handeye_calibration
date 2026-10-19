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
    JOINT_LIMIT_MARGIN_RAD, ClosedFormViews, weakest_observability)
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
    s.views, s.validation, s.rejections = [], [], {}
    s.initial, s.estimate, s.board, s.rng = np.zeros(6), np.eye(4), np.eye(4), np.random.default_rng(0)
    s.underactuated, s.n = False, NS()
    s.fk = NS(camera=lambda q: np.eye(4))
    updates = []
    s.r = NS(check=lambda: None, update=lambda *a, **kw: updates.append(a[1] if len(a) > 1 else ''),
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
