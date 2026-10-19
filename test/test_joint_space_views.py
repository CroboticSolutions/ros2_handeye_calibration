"""Joint-space view pool for a camera on PiPER link5 (Femto run 20261014_090523)."""
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hand_eye_calibration.closed_form_views import DISTANCE_RANGE_M, ClosedFormViews, weakest_observability
from hand_eye_calibration.joint_space_views import JointSpaceViews
from hand_eye_calibration.motion_planning import SAMPLER_JOINT_MARGIN_RAD
from hand_eye_calibration.visibility import BoardFraming, CameraKinematics

URDF = (Path(__file__).resolve().parents[3] / 'robots/piper_ros/src/robot_description/'
        'piper_description/urdf/piper_description.urdf')
NAMES = [f'joint{i}' for i in range(1, 7)]
# Offline solve of run 20261014_090523: board in base_link, link5 -> optical.
BOARD = np.array([[0.99972, 0.01183, 0.02051, 0.34904], [0.01203, -0.99988, -0.00931, 0.101],
                  [0.0204, 0.00956, -0.99975, -0.01374], [0, 0, 0, 1.]])
MOUNT = np.array([[0.00147, 0.99999, -0.00303, -0.06338], [0.00396, -0.00304, -0.99999, -0.09507],
                  [-0.99999, 0.00146, -0.00396, -0.02572], [0, 0, 0, 1.]])
INITIAL = np.array([-0.0095, 1.1723, -1.1282, 0.1169, 1.1863, 0.0])
CAMERA = NS(k=[747.752, 0, 637.843, 0, 747.179, 353.929, 0, 0, 1], d=[0.] * 8, width=1280, height=720,
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
    root = ET.fromstring(URDF.read_text())
    fk = CameraKinematics(root, 'base_link', 'link5', NAMES, np.eye(4))
    limits = [root.find(f"joint[@name='{n}']/limit") for n in NAMES]
    lower = np.array([float(x.get('lower')) for x in limits])
    upper = np.array([float(x.get('upper')) for x in limits])
    planner = JointSpaceViews(fk, NAMES, lower, upper)
    framing = BoardFraming(SPEC, CAMERA)
    center = (BOARD @ np.r_[framing.center, 1])[:3]
    reference = fk.camera(INITIAL) @ MOUNT
    pool, sampled = planner.pool(BOARD, center, MOUNT, framing, reference, 0.30, INITIAL,
                                 np.random.default_rng(3), 60)
    return NS(root=root, planner=planner, fk=fk, lower=lower, upper=upper, framing=framing, center=center,
              reference=reference, pool=pool, sampled=sampled)


def test_batched_fk_matches_camera_kinematics(scene):
    q = np.random.default_rng(0).uniform(scene.lower, scene.upper, (20, 6))
    for pose, joints in zip(scene.planner.effectors(q), q):
        np.testing.assert_allclose(pose, scene.fk.camera(joints), atol=1e-12)
    assert scene.planner.active.tolist() == [True] * 5 + [False]   # joint6 does not move a link5 camera


def test_pool_views_are_framed_and_clear_of_limits(scene):
    assert len(scene.pool) >= 60
    for q, effector in scene.pool:
        assert np.all(q[:5] >= scene.lower[:5] + SAMPLER_JOINT_MARGIN_RAD - 1e-9)
        assert np.all(q[:5] <= scene.upper[:5] - SAMPLER_JOINT_MARGIN_RAD + 1e-9)
        assert q[5] == INITIAL[5]
        np.testing.assert_allclose(effector, scene.fk.camera(q), atol=1e-12)
        camera = effector @ MOUNT
        assert scene.framing.contains(np.linalg.inv(camera) @ BOARD, margin=.13)
        assert np.linalg.norm(camera[:3, 3] - scene.reference[:3, 3]) <= 0.30
        assert DISTANCE_RANGE_M[0] <= np.linalg.norm(camera[:3, 3] - scene.center) <= DISTANCE_RANGE_M[1]


def test_greedy_views_are_more_diverse_than_the_captured_run(scene):
    # The local sampler's 20 views of the run: weakest direction 1.98, median relative rotation 11.7 deg.
    pool = list(scene.pool)
    chosen = [scene.fk.camera(INITIAL)[:3, :3]]
    for _ in range(19):
        chosen.append(pool.pop(ClosedFormViews.best(pool, chosen))[1][:3, :3])
    assert weakest_observability(gram(chosen)) > 3.
    relative = [np.degrees(Rotation.from_matrix(a.T @ b).magnitude()) for a, b in itertools.combinations(chosen, 2)]
    assert np.median(relative) > 20


def test_link5_session_uses_the_joint_space_pool(scene):
    from hand_eye_calibration.intrinsic_acquisition import IntrinsicSession
    node = NS(tracking_base_frame='camera_color_optical_frame', robot_base_frame='base_link',
              robot_effector_frame='link5')
    runner = NS(node=node, camera_info=NS(header=NS(frame_id='camera_color_optical_frame'), **vars(CAMERA)),
                board_spec=SPEC, names=NAMES, lower=scene.lower, upper=scene.upper,
                robot_description=URDF.read_text(), joint_positions=lambda: INITIAL.copy(),
                min_training_samples=20, validation_views=5)
    session = IntrinsicSession(runner, scene.root)
    assert session.underactuated
    assert isinstance(session.closed_form, JointSpaceViews)
