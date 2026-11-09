"""Geometry and cancellation regression tests; no robot motion or ROS discovery."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hand_eye_calibration.automatic_calibration import AutomaticCalibration, Stopped, sample_matrix
from hand_eye_calibration.calibration_backend import CalibrationBackend


def transform(t, r):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler('xyz', r).as_matrix()
    m[:3, 3] = t
    return m


def as_sample(m):
    return list(m[:3, 3]) + list(Rotation.from_matrix(m[:3, :3]).as_quat())


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
    # Stand-in for visibility.CameraKinematics: the virtual robot's camera FK.
    runner.fake_kinematics = lambda *a: SimpleNamespace(camera=physical_robot)
    params = {'auto_joint_names': robot_profile.get('names',[f'joint{i}' for i in range(1,7)]) if robot_profile else [f'joint{i}' for i in range(1,7)]}
    node = SimpleNamespace(get_parameter=lambda name: SimpleNamespace(value=params[name]),
        solver_name='intrinsic_pose', robot_base_frame='base', tracking_base_frame='camera', robot_effector_frame='wrist',
        robot_samples=[], tracking_samples=[], sample_metrics=[], sample_roles=[], _publish_status=Mock(),
        _last_uncertainty=None, _last_calibration_detail=None, get_logger=lambda: Mock())
    runner.node = node
    runner.capture_role = 'training'
    runner.min_training_samples = 20; runner.max_training_samples = 30; runner.validation_views = 5
    node.training_indices = lambda: [i for i,role in enumerate(node.sample_roles) if role == 'training']
    node.training_count = lambda: len(node.training_indices())
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
    runner.wait_for_tracking = lambda: None
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
        if bad_validation and runner.capture_role == 'validation':
            tracking[0,3] += .05
        node.robot_samples.append(as_sample(physical_robot(current[0])))
        node.tracking_samples.append(as_sample(tracking)); node.sample_metrics.append({})
        node.sample_roles.append(runner.capture_role)
        return True
    runner.capture = capture
    def calibration(full=False):
        ids = node.training_indices()
        result = CalibrationBackend.compute_calibration([node.robot_samples[i] for i in ids], [node.tracking_samples[i] for i in ids])
        # Acquisition fixture: exact synthetic data and a passing quality report.
        node._last_reprojection = {'board_in_base':as_sample(board),
            'uncertainty':{'worst_direction_sigma_m':.001},
            'board_spread':{'position_rms_m':0.},
            'leave_one_out':{'position_rms_m':0.,'reprojection_rms_px':0.}}
        return result
    node.get_calibration = calibration
    node._compute_uncertainty = lambda cal: {'worst_direction_sigma_m': .001, 'n_bootstrap':40, 'worst_direction_axis':[0.,0.,1.]}
    node.save_calibration_service_callback = Mock(return_value=SimpleNamespace(success=True, message='Saved'))
    return runner, node, mount


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


def test_teach_mode_rejected_before_start():
    runner = bare_runner()
    runner._teaching(SimpleNamespace(data=True))
    response = runner.start(None, SimpleNamespace(success=False, message=''))
    assert not response.success
    assert 'teach mode' in response.message
    assert runner.thread is None


def test_stale_teach_feedback_blocks_motion():
    runner = bare_runner()
    runner._teaching(SimpleNamespace(data=False))
    assert runner.motion_block_reason() is None
    runner.teaching_at -= 2
    assert 'stale' in runner.motion_block_reason()


def test_initial_joint_limits_report_actual_joint_and_do_not_clamp():
    runner = bare_runner()
    runner.names = ['joint1', 'joint5']
    runner.lower = np.array([-2.618, -1.22]); runner.upper = np.array([2.168, 1.22])
    runner.require_joint_limits(np.array([0., 1.21]))
    runner.require_joint_limits(np.array([0., 1.22028]))  # stopped at the limit (run 20261014_090523)
    q = np.array([0., 1.24])
    with pytest.raises(RuntimeError, match=r'joint5=.*allowed'):
        runner.require_joint_limits(q)
    assert q[1] == 1.24
