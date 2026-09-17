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
    runner.check_collisions = False
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


def synthetic_sequence(monkeypatch, *, bad_validation=False, reachable=True, robot_profile=None):
    """Independent virtual robot + camera; no mount TF is available to the worker."""
    from sensor_msgs.msg import CameraInfo
    from hand_eye_calibration import bootstrap_calibration as bootstrap
    runner = bare_runner(); runner.status['active'] = True
    mount = transform([.03, -.02, .04], [.6, -.7, .4])
    current = [np.zeros(6) if robot_profile is None else np.array(robot_profile['joints'])]
    def physical_robot(q):
        return transform(np.array([.3, .1, .5]) + .15*q[:3], q[3:])
    if robot_profile is not None:
        import xml.etree.ElementTree as ET
        from hand_eye_calibration.visibility import CameraKinematics
        physical_robot = CameraKinematics(ET.fromstring(robot_profile['urdf']), 'base_link', robot_profile.get('tip','link6'),
            robot_profile.get('names',[f'joint{i}' for i in range(1,7)]), np.eye(4)).camera
    board = physical_robot(current[0]) @ mount @ transform([-.0975, -.0675, .65], [0, 0, 0])
    def observation():
        return np.linalg.inv(physical_robot(current[0]) @ mount) @ board
    monkeypatch.setattr(bootstrap, 'CameraKinematics', lambda *a: SimpleNamespace(camera=physical_robot))
    original_session = bootstrap.BootstrapSession
    def session(*args):
        result = original_session(*args); runner.session = result; return result
    monkeypatch.setattr(bootstrap, 'BootstrapSession', session)
    params = {'auto_joint_names': robot_profile.get('names',[f'joint{i}' for i in range(1,7)]) if robot_profile else [f'joint{i}' for i in range(1,7)]}
    node = SimpleNamespace(get_parameter=lambda name: SimpleNamespace(value=params[name]),
        robot_base_frame='base', tracking_base_frame='camera', robot_effector_frame='wrist',
        robot_samples=[], tracking_samples=[], sample_metrics=[], _publish_status=Mock(),
        _last_uncertainty=None, _last_calibration_detail=None, get_logger=lambda: Mock())
    runner.node = node
    def configure(root):
        runner.names=params['auto_joint_names']
        limits=[root.find(f"joint[@name='{name}']/limit") for name in runner.names]
        runner.lower=np.array([float(j.get('lower')) for j in limits])
        runner.upper=np.array([float(j.get('upper')) for j in limits])
    runner.configure_robot=configure
    runner.planning=SimpleNamespace(wait_for_service=lambda **_:True)
    runner.plan_joint_path=lambda start,end:[start.copy(),end.copy()] if reachable else None
    urdf = '<robot>' + ''.join(f'<joint name="joint{i}"><limit lower="-3" upper="3"/></joint>' for i in range(1, 7)) + '</robot>'
    if robot_profile is not None:
        urdf = robot_profile['urdf']
        node.robot_base_frame = 'base_link'; node.robot_effector_frame = robot_profile.get('tip','link6')
    runner.description = SimpleNamespace(wait_for_service=lambda **_: True, call_async=lambda _: SimpleNamespace(values=[SimpleNamespace(string_value=urdf)]))
    runner.wait = lambda value, _: value
    runner.check = lambda: None
    runner.ik = Mock(side_effect=AssertionError('Bootstrap must not request camera IK'))
    runner.action = SimpleNamespace(wait_for_server=lambda **_: True)
    runner.joint_positions = lambda: current[0].copy()
    runner.settle = lambda _: None
    runner.observed_board = lambda **_: observation()
    runner.tf = Mock(side_effect=AssertionError('No initial camera transform exists'))
    runner.update = lambda state, message, **values: runner.status.update(state=state, message=message, **values)
    runner.camera_info = CameraInfo(width=640, height=480, k=[600.,0.,320.,0.,600.,240.,0.,0.,1.])
    runner.camera_info.header.frame_id = 'camera'
    runner.board_spec = {'squares_x':13, 'squares_y':9, 'square_length_m':.015}
    runner.collision_free_path = lambda a,b: reachable or np.max(np.abs(a-b)) < .001
    def move(q, **kwargs):
        assert np.max(np.abs(q-current[0])) <= .051
        current[0] = q.copy()
    runner.move = Mock(side_effect=move)
    runner.sample_events = []
    def capture():
        tracking = observation()
        session = runner.session
        runner.sample_events.append((len(node.robot_samples)+1, session.estimate is not None,
            None if session.view_planner is None else float(np.max(np.abs(current[0]-session.view_planner.goal)))))
        if bad_validation and session.estimate is not None and (len(session.views)+1)%3 == 0:
            tracking[0,3] += .05
        node.robot_samples.append(as_sample(physical_robot(current[0])))
        node.tracking_samples.append(as_sample(tracking)); node.sample_metrics.append({})
        return True
    runner.capture = capture
    node.get_calibration = lambda: CalibrationBackend.compute_calibration(node.robot_samples, node.tracking_samples)
    node._compute_uncertainty = lambda cal: {'worst_direction_sigma_m': .001, 'n_bootstrap':40, 'worst_direction_axis':[0.,0.,1.]}
    node.save_calibration_service_callback = Mock(return_value=SimpleNamespace(success=True, message='Saved'))
    return runner, node, mount


def test_complete_sequence_saves_only_training_samples(monkeypatch):
    runner, node, mount = synthetic_sequence(monkeypatch)
    runner.run()
    assert runner.status['state'] == 'completed', runner.status
    assert not runner.active
    assert len(node.robot_samples) == 15  # six initial + nine targeted, all used in the solve
    assert runner.status['initial_samples'] == 6
    assert runner.status['targeted_samples'] == 9
    assert [event[1] for event in runner.sample_events] == [False]*6+[True]*9
    assert all(event[2] is not None and event[2] < .003 for event in runner.sample_events[6:])
    assert all(call.kwargs['speed_scale'] == 1. for call in runner.move.call_args_list)
    runner.tf.assert_not_called()
    assert runner.status['validation'] is None
    node.save_calibration_service_callback.assert_called_once()
    np.testing.assert_allclose(sample_matrix(node.get_calibration()), mount, atol=1e-6)
    assert not np.allclose(runner.move.call_args.args[0], np.zeros(6))  # stays at the final useful view


def test_failed_validation_preserves_saved_calibration(monkeypatch):
    runner, node, _ = synthetic_sequence(monkeypatch, bad_validation=True)
    runner.run()
    assert runner.status['state'] == 'failed'
    assert 'inconsistent' in runner.status['message']
    node.save_calibration_service_callback.assert_not_called()


def test_unreachable_views_do_not_produce_a_saved_calibration(monkeypatch):
    runner, node, _ = synthetic_sequence(monkeypatch, reachable=False)
    runner.run()
    assert runner.status['state'] == 'failed'
    assert 'local search exhausted' in runner.status['message']
    assert runner.status['attempts'] == 8
    runner.move.assert_not_called()
    node.save_calibration_service_callback.assert_not_called()


def test_stop_during_motion_does_not_return_or_save(monkeypatch):
    runner, node, _ = synthetic_sequence(monkeypatch)
    runner.move = Mock(side_effect=Stopped('Stopped'))
    runner.run()
    assert runner.status['state'] == 'stopped'
    runner.move.assert_called_once()
    node.save_calibration_service_callback.assert_not_called()


def test_real_path_rejects_intermediate_collision_before_sending_goal():
    from sensor_msgs.msg import JointState
    runner = bare_runner(); runner.check_collisions = True
    runner.names = ['joint1']; runner.joints = JointState(name=['joint1'], position=[0.0])
    runner.check = lambda: None
    runner.joint_positions = lambda: np.array([0.0])
    requests = []
    def valid(req):
        requests.append(req)
        return SimpleNamespace(valid=not (.04 < req.robot_state.joint_state.position[0] < .06))
    runner.validity = SimpleNamespace(wait_for_service=lambda **_: True, call_async=valid)
    runner.node.get_parameter = lambda _: SimpleNamespace(value='arm')
    runner.wait = lambda value, _: value
    runner.action = Mock()
    with pytest.raises(RuntimeError, match='collision'):
        runner.move(np.array([.1]))
    assert len(requests) > 1 and all(r.robot_state.is_diff for r in requests)
    runner.action.send_goal_async.assert_not_called()


def test_real_path_fails_closed_if_collision_service_is_missing():
    runner = bare_runner(); runner.check_collisions = True
    runner.validity = SimpleNamespace(wait_for_service=lambda **_: False)
    runner.check = lambda: None
    runner.joint_positions = lambda: np.array([0.0])
    runner.action = Mock()
    with pytest.raises(RuntimeError, match='unavailable'):
        runner.move(np.array([.1]))
    runner.action.send_goal_async.assert_not_called()
