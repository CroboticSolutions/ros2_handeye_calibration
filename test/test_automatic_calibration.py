"""Geometry and cancellation regression tests; no robot motion or ROS discovery."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hand_eye_calibration.automatic_geometry import camera_targets, validation_error
from hand_eye_calibration.automatic_calibration import AutomaticCalibration, Stopped, sample_matrix
from hand_eye_calibration.calibration_backend import CalibrationBackend


def transform(t, r):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler('xyz', r).as_matrix()
    m[:3, 3] = t
    return m


def as_sample(m):
    return list(m[:3, 3]) + list(Rotation.from_matrix(m[:3, :3]).as_quat())


def test_orbits_preserve_board_bearing_and_recover_camera_transform():
    initial = transform([0.4, 0.05, 0.3], [0.4, 2.8, -0.3])
    observation = transform([0.025, -0.015, 0.32], [0.1, 0.2, 0])
    mount = transform([0.025, 0.03, 0.04], [0.05, 0.1, -0.1])
    board = initial @ observation
    poses = camera_targets(initial, observation)
    assert len(poses) == 24
    robots, observations = [], []
    for camera in poses:
        obs = np.linalg.inv(camera) @ board
        np.testing.assert_allclose(obs[:3, 3] / obs[2, 3], observation[:3, 3] / observation[2, 3], atol=1e-10)
        robots.append(camera @ np.linalg.inv(mount))
        observations.append(obs)
    result = CalibrationBackend.compute_calibration(
        list(map(as_sample, robots[:-3])), list(map(as_sample, observations[:-3])))
    np.testing.assert_allclose(sample_matrix(result), mount, atol=1e-6)
    check = validation_error(robots[:-3], observations[:-3], sample_matrix(result), list(zip(robots[-3:], observations[-3:])))
    assert check['max_translation_m'] < 1e-6
    wrong = mount.copy(); wrong[0, 3] += 0.1
    check = validation_error(robots[:-3], observations[:-3], wrong, list(zip(robots[-3:], observations[-3:])))
    assert check['max_translation_m'] > 0.01


@pytest.mark.parametrize('depth', [0, -0.2, float('nan')])
def test_invalid_initial_board_is_rejected(depth):
    board = np.eye(4); board[2, 3] = depth
    with pytest.raises(ValueError):
        camera_targets(np.eye(4), board)


def bare_runner():
    runner = AutomaticCalibration.__new__(AutomaticCalibration)
    runner.lock = threading.RLock()
    runner.stop_event = threading.Event()
    runner.goal = None
    runner.thread = None
    runner.visible = True
    runner.visible_at = time.monotonic()
    runner.status = {'active': False}
    runner.node = SimpleNamespace(get_parameter=lambda _: SimpleNamespace(value=True), calibration_type='eye-in-hand')
    return runner


def test_stale_detection_cannot_start():
    runner = bare_runner(); runner.visible_at -= 10
    resp = runner.start(None, SimpleNamespace(success=False, message=''))
    assert not resp.success
    assert runner.thread is None


def test_duplicate_start_cannot_spawn_worker():
    runner = bare_runner(); runner.status['active'] = True
    resp = runner.start(None, SimpleNamespace(success=False, message=''))
    assert not resp.success
    assert runner.thread is None


def test_stop_cancels_current_trajectory_and_interrupts_wait():
    runner = bare_runner(); runner.goal = Mock()
    response = runner.stop(None, SimpleNamespace())
    assert response.success
    runner.goal.cancel_goal_async.assert_called_once()
    with pytest.raises(Stopped):
        runner.wait(SimpleNamespace(done=lambda: False), 30)


def test_stop_while_goal_acceptance_is_pending_cancels_late_goal(monkeypatch):
    runner = bare_runner()
    runner.names = ['joint1']
    runner.check = lambda: None
    runner.joint_positions = lambda: np.array([0.0])
    callbacks = []
    future = SimpleNamespace(add_done_callback=callbacks.append)
    runner.action = SimpleNamespace(send_goal_async=lambda _: future)
    def stop_wait(*args):
        runner.stop_event.set()
        raise Stopped()
    runner.wait = stop_wait
    with pytest.raises(Stopped):
        runner.move(np.array([0.1]))
    handle = Mock(accepted=True)
    callbacks[0](SimpleNamespace(result=lambda: handle))
    handle.cancel_goal_async.assert_called_once()


def synthetic_sequence(monkeypatch, *, bad_validation=False, reachable=True):
    """Exercise the complete worker with a deterministic virtual controller/camera."""
    runner = bare_runner()
    runner.status['active'] = True
    initial_camera = transform([0.4, 0, 0.3], [0.3, 2.9, 0])
    observation = transform([0.01, 0.02, 0.3], [0.1, 0.1, 0])
    mount = transform([0.02, 0.03, 0.04], [0.1, -0.1, 0.05])
    board = initial_camera @ observation
    current = [initial_camera]
    params = {'auto_joint_names': [f'joint{i}' for i in range(1, 7)], 'auto_ik_link': 'arm_tcp'}
    node = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value=params[name]),
        robot_base_frame='base', tracking_base_frame='camera', robot_effector_frame='wrist',
        robot_samples=[], tracking_samples=[], sample_metrics=[], _publish_status=Mock(),
        _last_uncertainty=None, _last_calibration_detail=None, get_logger=lambda: Mock())
    runner.node = node
    urdf = '<robot>' + ''.join(f'<joint name="joint{i}"><limit lower="-3" upper="3"/></joint>' for i in range(1, 7)) + '</robot>'
    runner.description = SimpleNamespace(wait_for_service=lambda **_: True, call_async=lambda _: SimpleNamespace(values=[SimpleNamespace(string_value=urdf)]))
    runner.wait = lambda value, _: value
    runner.check = lambda: None
    runner.ik = SimpleNamespace(wait_for_service=lambda **_: True)
    runner.action = SimpleNamespace(wait_for_server=lambda **_: True)
    runner.joint_positions = lambda: np.zeros(6)
    runner.settle = lambda _: None
    runner.fresh_board = lambda: observation
    monkeypatch.setattr('hand_eye_calibration.automatic_calibration.matrix', lambda value: value)
    runner.tf = lambda parent, child: initial_camera if parent == 'base' else np.eye(4)
    runner.update = lambda state, message, **values: runner.status.update(state=state, message=message, **values)
    poses = []
    def solve(pose, seed):
        poses.append(pose)
        return np.full(6, len(poses)) if reachable else None
    runner.solve = solve
    runner.move = Mock(side_effect=lambda q: current.__setitem__(0, initial_camera if q[0] == 0 else poses[int(q[0])-1]))
    capture_count = [0]
    def capture():
        capture_count[0] += 1
        robot = current[0] @ np.linalg.inv(mount)
        tracking = np.linalg.inv(current[0]) @ board
        if bad_validation and capture_count[0] > 21:
            tracking[0, 3] += 0.05
        node.robot_samples.append(as_sample(robot))
        node.tracking_samples.append(as_sample(tracking))
        node.sample_metrics.append({})
        return True
    runner.prepare_framing = lambda _: None
    runner.fits = lambda _: True
    runner.path_fits = lambda *args: True
    runner.guarded_move = lambda q: runner.move(q)
    runner.capture = capture
    node.get_calibration = lambda: CalibrationBackend.compute_calibration(node.robot_samples, node.tracking_samples)
    node.save_calibration_service_callback = Mock(return_value=SimpleNamespace(success=True, message='Saved'))
    return runner, node, mount


def test_complete_sequence_saves_only_training_samples(monkeypatch):
    runner, node, mount = synthetic_sequence(monkeypatch)
    runner.run()
    assert runner.status['state'] == 'completed', runner.status
    assert not runner.active
    assert len(node.robot_samples) == 21  # all three holdouts excluded
    assert runner.status['validation']['poses'] == 3
    node.save_calibration_service_callback.assert_called_once()
    np.testing.assert_allclose(sample_matrix(node.get_calibration()), mount, atol=1e-6)
    np.testing.assert_array_equal(runner.move.call_args.args[0], np.zeros(6))


def test_failed_validation_preserves_saved_calibration(monkeypatch):
    runner, node, _ = synthetic_sequence(monkeypatch, bad_validation=True)
    runner.run()
    assert runner.status['state'] == 'failed'
    assert 'validation failed' in runner.status['message']
    node.save_calibration_service_callback.assert_not_called()


def test_unreachable_views_do_not_produce_a_saved_calibration(monkeypatch):
    runner, node, _ = synthetic_sequence(monkeypatch, reachable=False)
    runner.run()
    assert runner.status['state'] == 'failed'
    assert 'Too few valid views' in runner.status['message']
    node.save_calibration_service_callback.assert_not_called()


def test_stop_during_motion_does_not_return_or_save(monkeypatch):
    runner, node, _ = synthetic_sequence(monkeypatch)
    runner.move = Mock(side_effect=Stopped('Stopped'))
    runner.run()
    assert runner.status['state'] == 'stopped'
    runner.move.assert_called_once()
    node.save_calibration_service_callback.assert_not_called()
