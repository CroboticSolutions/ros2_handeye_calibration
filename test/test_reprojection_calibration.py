import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hand_eye_calibration.reprojection_calibration import (
    Camera, Dataset, JointOffsetModel, calibrate, merge_frames, pose_list, pose_matrix)
from hand_eye_calibration.acceptance import evaluate


K = [900., 0., 640., 0., 900., 360., 0., 0., 1.]
BOARD = np.array([[x * .015, y * .015, 0.] for y in range(1, 9) for x in range(1, 13)])


def transform(rotvec, t):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    m[:3, 3] = t
    return m


X_TRUE = transform([.2, -.3, 1.4], [-.11, -.03, .01])
B_TRUE = transform([math.pi, 0, .3], [.45, .05, .02])


def look_poses(n, rng):
    """Camera poses around the board centre, returned as base->effector."""
    centre = (B_TRUE @ np.r_[BOARD.mean(axis=0), 1])[:3]
    poses = []
    while len(poses) < n:
        tilt = rng.uniform(-.5, .5, 2)
        roll = rng.uniform(-.8, .8)
        dist = rng.uniform(.25, .4)
        cam = np.eye(4)
        # Optical axis points down onto the board (board z is flipped).
        cam[:3, :3] = B_TRUE[:3, :3] @ Rotation.from_rotvec([tilt[0], tilt[1], roll]).as_matrix()
        cam[:3, 3] = centre - cam[:3, :3] @ [0, 0, dist]
        poses.append(cam @ np.linalg.inv(X_TRUE))
    return poses


def project(robot, noise_px, rng, x=X_TRUE, b=B_TRUE):
    T = np.linalg.inv(x) @ np.linalg.inv(robot) @ b
    p = BOARD @ T[:3, :3].T + T[:3, 3]
    k = np.asarray(K).reshape(3, 3)
    uv = np.column_stack((k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]))
    return uv + rng.normal(0, noise_px, uv.shape), T


def make_samples(n=15, noise_px=.3, robot_noise_m=0., seed=1, frames=3):
    rng = np.random.default_rng(seed)
    samples = []
    for robot in look_poses(n, rng):
        measured = robot.copy()
        measured[:3, 3] += rng.normal(0, robot_noise_m, 3)
        frame_list = []
        for _ in range(frames):
            uv, T = project(robot, noise_px, rng)
            frame_list.append({'ids': list(range(len(BOARD))), 'image_points': uv.tolist(),
                               'object_points': BOARD.tolist()})
        samples.append({'robot': pose_list(measured), 'tracking': pose_list(T), 'frames': frame_list})
    return samples


def rotation_deg(a, b):
    return math.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())


def test_merge_frames_uses_median_and_drops_rare_corners():
    frames = [{'ids': [0, 1, 2, 3, 4, 5, 6], 'image_points': [[i, 0]] * 7, 'object_points': [[0, 0, 0]] * 7}
              for i in (1., 2., 100.)]
    frames[0]['ids'] = [0, 1, 2, 3, 4, 5, 9]
    ids, image, _ = merge_frames(frames)
    assert 9 not in ids.tolist()
    assert image[0, 0] == 2.


def test_recovers_mount_from_perturbed_initial_guess():
    samples = make_samples()
    dataset = Dataset(samples, Camera(K, [0] * 5))
    start = X_TRUE @ transform([.02, -.01, .03], [.006, -.004, .008])
    report = calibrate(dataset, start, bootstrap_samples=0)
    X = pose_matrix(report['transform'])
    assert np.linalg.norm(X[:3, 3] - X_TRUE[:3, 3]) < 5e-4
    assert rotation_deg(X, X_TRUE) < .05
    assert report['reprojection_rms_px'] < .6
    assert report['leave_one_out']['position_rms_m'] < .002


def test_board_spread_and_loo_expose_robot_pose_error():
    clean = calibrate(Dataset(make_samples(), Camera(K, [0] * 5)), X_TRUE, bootstrap_samples=0)
    noisy = calibrate(Dataset(make_samples(robot_noise_m=.004), Camera(K, [0] * 5)), X_TRUE, bootstrap_samples=0)
    assert noisy['board_spread']['position_rms_m'] > 3 * clean['board_spread']['position_rms_m']
    assert noisy['leave_one_out']['position_rms_m'] > .002
    assert not evaluate(noisy, 15)['passed']


def test_validation_views_are_not_used_for_training():
    samples = make_samples(n=18)
    report = calibrate(Dataset(samples, Camera(K, [0] * 5)), X_TRUE, validation_indices=[15, 16, 17],
                       bootstrap_samples=0, leave_one_out=False)
    assert report['training_views'] == 15
    assert report['validation_views']['views'] == 3
    assert report['validation_views']['position_max_m'] < .003


def test_bootstrap_reports_uncertainty_and_clean_data_passes_gate():
    report = calibrate(Dataset(make_samples(), Camera(K, [0] * 5)), X_TRUE, bootstrap_samples=12)
    assert report['uncertainty']['n_bootstrap'] >= 8
    assert report['uncertainty']['worst_direction_sigma_m'] < .002
    result = evaluate(report, 15)
    assert result['passed'], result['summary']
    assert not evaluate(report, 8)['passed']


def test_intrinsics_refinement_recovers_focal_error():
    samples = make_samples(n=18, noise_px=.2)
    wrong = list(K)
    wrong[0] *= 1.01
    wrong[4] *= 1.01
    report = calibrate(Dataset(samples, Camera(wrong, [0] * 5)), X_TRUE, estimate_intrinsics=True,
                       bootstrap_samples=0, leave_one_out=False)
    assert abs(report['intrinsics']['fx'] - 900.) < 3.
    assert np.linalg.norm(pose_matrix(report['transform'])[:3, 3] - X_TRUE[:3, 3]) < .002


def test_outlier_view_is_rejected():
    samples = make_samples(n=15)
    bad = pose_matrix(samples[4]['robot'])
    bad[:3, 3] += [.02, 0, 0]
    samples[4]['robot'] = pose_list(bad)
    report = calibrate(Dataset(samples, Camera(K, [0] * 5)), X_TRUE, bootstrap_samples=0, leave_one_out=False)
    assert report['rejected_views'] == [4]
    assert np.linalg.norm(pose_matrix(report['transform'])[:3, 3] - X_TRUE[:3, 3]) < 1e-3


class PlanarArm:
    """Toy 4-joint chain with an offset on joint 2 to test offset estimation."""
    names = ['j1', 'j2', 'j3', 'j4']

    def camera(self, q):
        m = transform([0, 0, q[0]], [0, 0, .1])
        m = m @ transform([0, q[1], 0], [0, 0, .1]) @ transform([0, q[2], 0], [.25, 0, 0])
        return m @ transform([q[3], 0, 0], [.2, 0, 0])


def test_joint_offset_is_estimated():
    rng = np.random.default_rng(3)
    arm = PlanarArm()
    true_offset = math.radians(1.0)
    x = transform([0, math.pi, 0], [.03, 0, -.02])
    samples = []
    for _ in range(2000):
        if len(samples) == 20:
            break
        q = np.array([rng.uniform(-.4, .4), rng.uniform(-.3, .3), rng.uniform(-.6, .1), rng.uniform(-.6, .6)])
        actual = q.copy(); actual[1] += true_offset
        robot = arm.camera(actual)
        cam = robot @ x
        board = transform([0, 0, 0], [.35, -.07, -.25])
        T = np.linalg.inv(cam) @ board
        if T[2, 3] < .1:
            continue
        p = BOARD @ T[:3, :3].T + T[:3, 3]
        if np.any(p[:, 2] < .05):
            continue
        uv = np.column_stack((900 * p[:, 0] / p[:, 2] + 640, 900 * p[:, 1] / p[:, 2] + 360))
        if np.any(uv < 0) or np.any(uv[:, 0] > 1280) or np.any(uv[:, 1] > 720):
            continue
        samples.append({'robot': pose_list(arm.camera(q)), 'tracking': pose_list(T),
                        'joints': dict(zip(arm.names, q)),
                        'frames': [{'ids': list(range(len(BOARD))), 'image_points': uv.tolist(),
                                    'object_points': BOARD.tolist()}]})
    assert len(samples) == 20
    dataset = Dataset(samples, Camera(K, [0] * 5))
    model = JointOffsetModel(arm, arm.names, prior_sigma_rad=math.radians(3))
    report = calibrate(dataset, x, joint_model=model, bootstrap_samples=0, leave_one_out=False)
    assert report['joint_offsets']['j2'] == pytest.approx(true_offset, abs=math.radians(.15))
