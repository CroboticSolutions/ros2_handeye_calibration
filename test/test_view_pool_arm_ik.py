"""Analytic view pool on every arm_ik kind (arm_ik test fixtures): UR10 eaik, CRX ssik,
PiPER closed_form. Real camera mounts from the URDFs."""
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip('arm_ik')
from hand_eye_calibration.view_pool import (  # noqa: E402
    JOINT_LIMIT_MARGIN_RAD, AnalyticIKViews, ViewLimits, ViewPoolUnavailable, joint_window_bounds,
    normalize_ik_backend, tilt_and_roll)
from hand_eye_calibration.visibility import BoardFraming, CameraKinematics  # noqa: E402

DATA = Path(__file__).resolve().parents[3] / 'robots/arm_ik/test/data'
# kind: (urdf, effector (IK link), camera optical frame, joints)
ROBOTS = {
    'eaik': ('ur10_weld.urdf', 'tool0', 'camera_color_optical_frame',
             ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
              'wrist_3_joint']),
    'ssik': ('crx10ia_l.urdf', 'flange', 'oak_rgb_camera_optical_frame', [f'J{i}' for i in range(1, 7)]),
    'closed_form': ('piper_welding_gun.urdf', 'link6', 'camera_color_optical_frame',
                    [f'joint{i}' for i in range(1, 7)]),
}
CAMERA = NS(k=[600., 0, 320., 0, 600., 240., 0, 0, 1], d=[0.] * 5, width=640, height=480,
            distortion_model='plumb_bob')
SPEC = {'squares_x': 6, 'squares_y': 8, 'square_length_m': 0.025}
LIMITS = ViewLimits()
WANTED = 24


def chain_limits(root, names):
    limits = [root.find(f"joint[@name='{n}']/limit") for n in names]
    return np.array([float(x.get('lower')) for x in limits]), np.array([float(x.get('upper')) for x in limits])


def make_scene(kind):
    name, effector, optical, names = ROBOTS[kind]
    urdf = (DATA / name).read_text()
    root = ET.fromstring(urdf)
    fk = CameraKinematics(root, 'base_link', effector, names, np.eye(4))
    camera_fk = CameraKinematics(root, 'base_link', optical, names, np.eye(4))
    lower, upper = chain_limits(root, names)
    views = AnalyticIKViews.build(kind, urdf, 'base_link', effector, names, lower, upper, fk.camera, LIMITS)
    rng = np.random.default_rng(7)
    lo, hi = np.maximum(lower, -np.pi) + .3, np.minimum(upper, np.pi) - .3
    for _ in range(20000):   # a start with the camera looking down and the wrist bent
        q = rng.uniform(lo, hi)
        camera = camera_fk.camera(q)
        if camera[2, 2] < -.85 and abs(q[4]) > .4 and camera[2, 3] > .3:
            break
    else:
        pytest.fail('no start pose')
    mount = np.linalg.inv(fk.camera(q)) @ camera
    framing = BoardFraming(SPEC, CAMERA)
    board = camera.copy()                       # board facing the camera 0.45 m ahead, centred
    board[:3, 3] = camera[:3, 3] + 0.45 * camera[:3, 2] - camera[:3, :3] @ np.asarray(framing.center)
    center = (board @ np.r_[framing.center, 1])[:3]
    pool, sampled = views.pool(board, center, mount, framing, camera, 0.30, q, np.random.default_rng(3), WANTED)
    return NS(views=views, fk=fk, lower=lower, upper=upper, initial=q, mount=mount, board=board,
              framing=framing, reference=camera, pool=pool, sampled=sampled, urdf=urdf, names=names,
              effector=effector)


@pytest.fixture(scope='module', params=sorted(ROBOTS))
def scene(request):
    return make_scene(request.param)


def test_pool_uses_the_requested_solver_kind(scene):
    assert scene.views.kind == scene.views.solver.kind
    assert scene.views.label == f'analytic IK ({scene.views.kind})'
    assert scene.views.kind in scene.views.describe() and 'model tolerance' in scene.views.describe()


def test_pool_candidate_count_and_fk_cross_check(scene):
    assert len(scene.pool) >= WANTED and scene.sampled > 0
    tolerance = scene.views.solver.model_tolerance_m
    for q, effector in scene.pool:
        np.testing.assert_allclose(scene.fk.camera(q), effector, atol=1e-9)    # runner FK = solver FK
        camera = effector @ scene.mount
        assert scene.framing.contains(np.linalg.inv(camera) @ scene.board, margin=.12)
        tilt, roll = tilt_and_roll(scene.reference, camera[None])
        assert tilt[0] <= np.deg2rad(LIMITS.max_tilt_deg) + 1e-3 + tolerance
        assert abs(roll[0]) <= np.deg2rad(LIMITS.max_roll_deg) + 1e-3 + tolerance


def test_pool_respects_the_joint_window(scene):
    lo, hi = joint_window_bounds(scene.initial, scene.lower + JOINT_LIMIT_MARGIN_RAD,
                                 scene.upper - JOINT_LIMIT_MARGIN_RAD, LIMITS)
    for q, _ in scene.pool:
        assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)
        assert np.sign(q[4]) == np.sign(scene.initial[4])               # no wrist flip


def test_solve_takes_the_in_window_branch_nearest_the_reference(scene):
    views, q0 = scene.views, scene.initial
    lo, hi = joint_window_bounds(q0, views.lower, views.upper, views.limits)
    steps = np.random.default_rng(1).uniform(-.15, .15, (30, 6))
    targets = np.array([scene.fk.camera(q0 + d) for d in steps])
    far = scene.fk.camera(q0 + np.r_[0, 0, 0, 0, 0, 2.5])            # J6 outside the 60 deg window
    solutions = views.solve(np.concatenate([targets, far[None]]), q0, lo, hi)
    assert solutions[-1] is None
    tolerance = views.solver.model_tolerance_m
    for q, d, target in zip(solutions, steps, targets):
        assert q is not None and np.all(q >= lo) and np.all(q <= hi)
        assert np.linalg.norm(scene.fk.camera(q)[:3, 3] - target[:3, 3]) <= tolerance + 1e-9
        if tolerance <= 1e-6:                                         # exact solvers: the sampled q
            np.testing.assert_allclose(q, q0 + d, atol=1e-6)


def test_build_refuses_what_it_cannot_solve():
    name, effector, _, names = ROBOTS['ssik']
    urdf = (DATA / name).read_text()
    root = ET.fromstring(urdf)
    fk = CameraKinematics(root, 'base_link', effector, names, np.eye(4))
    lower, upper = chain_limits(root, names)
    with pytest.raises(ViewPoolUnavailable, match='eaik does not support'):
        AnalyticIKViews.build('eaik', urdf, 'base_link', effector, names, lower, upper, fk.camera)   # CRX
    with pytest.raises(ViewPoolUnavailable, match="'service'"):
        AnalyticIKViews.build('service', urdf, 'base_link', effector, names, lower, upper, fk.camera)
    with pytest.raises(ViewPoolUnavailable, match='FK disagrees'):
        AnalyticIKViews.build('ssik', urdf, 'base_link', effector, names, lower, upper, lambda q: np.eye(4))
    with pytest.raises(ViewPoolUnavailable, match='differ'):
        AnalyticIKViews.build('ssik', urdf, 'base_link', effector, names[::-1], lower, upper, fk.camera)
    p_name, _, _, p_names = ROBOTS['closed_form']
    p_urdf = (DATA / p_name).read_text()
    p_root = ET.fromstring(p_urdf)
    p_fk = CameraKinematics(p_root, 'base_link', 'link5', p_names[:5], np.eye(4))
    p_lower, p_upper = chain_limits(p_root, p_names[:5])
    with pytest.raises(ViewPoolUnavailable, match='closed_form does not support'):   # camera before J6
        AnalyticIKViews.build('closed_form', p_urdf, 'base_link', 'link5', p_names[:5], p_lower, p_upper, p_fk.camera)
    assert AnalyticIKViews.create(p_urdf, 'base_link', 'link5', p_names[:5], p_lower, p_upper, p_fk.camera,
                                  kind='closed_form') is None


def test_ik_backend_values():
    assert [normalize_ik_backend(v) for v in ('eaik', 'SSIK', ' closed_form ', 'analytic', 'service', '', None)] == \
        ['eaik', 'ssik', 'closed_form', 'ssik', 'service', 'ssik', 'ssik']
    with pytest.raises(ValueError):
        normalize_ik_backend('kdl')
