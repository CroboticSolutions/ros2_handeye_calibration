import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hand_eye_calibration.visibility import BoardFraming, CameraKinematics, visible_joint_path
from hand_eye_calibration.automatic_calibration import AutomaticCalibration, FramingCorrection, TrackingInterrupted


def framing():
    return BoardFraming(dict(squares_x=13, squares_y=9, square_length_m=.015),
        SimpleNamespace(width=1920, height=1080, k=[1386, 0, 960, 0, 1388, 540, 0, 0, 1], d=[], distortion_model='plumb_bob'))


def observation():
    m = np.eye(4); m[:3, 3] = [-.0975, -.0675, .55]
    return m


def test_full_board_required_even_when_origin_visible():
    f = framing(); board = observation()
    assert f.contains(board)
    board[0, 3] = .2
    assert not f.contains(board)
    board[2, 3] = -.5
    assert not f.contains(board)


def test_correction_preserves_tilt_and_fits_entire_board():
    f = framing(); board = observation()
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler('xyz', [.15, .35, .1]).as_matrix()
    camera[0, 3] = .2
    assert not f.contains(np.linalg.inv(camera) @ board)
    corrected = f.corrected(camera, board)
    assert corrected is not None
    np.testing.assert_allclose(corrected[:3, :3], camera[:3, :3])
    assert f.contains(np.linalg.inv(corrected) @ board)


def test_path_rejects_invisible_middle_with_visible_endpoints():
    f = framing(); board = observation()
    def camera(q):
        m = np.eye(4); m[0, 3] = .5*np.sin(np.pi*q[0])
        return m
    assert f.contains(board)
    assert not visible_joint_path(np.array([0.]), np.array([1.]), camera, f, board)


def test_urdf_fk_includes_origin_axis_and_camera_offset():
    root = ET.fromstring('''<robot><joint name="j" type="revolute"><parent link="base"/>
      <child link="tip"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint></robot>''')
    offset = np.eye(4); offset[0, 3] = -.1
    fk = CameraKinematics(root, 'base', 'tip', ['j'], offset)
    np.testing.assert_allclose(fk.camera([np.pi/2])[:3, 3], [1, .1, 0], atol=1e-10)


def runner():
    r = AutomaticCalibration.__new__(AutomaticCalibration)
    r.check_collisions = False
    r.check = lambda: None
    r.update = Mock()
    r.framing = framing()
    r.optical_tracking = np.eye(4)
    r.kinematics = SimpleNamespace(camera=lambda q: np.eye(4))
    r.observed_board = Mock(return_value=observation())
    state = [np.zeros(6)]
    r.joint_positions = lambda: state[0].copy()
    r.move = Mock(side_effect=lambda q, **kwargs: state.__setitem__(0, q.copy()))
    return r










def test_camera_intrinsic_change_interrupts_active_run():
    r = runner(); r.stop_event = threading.Event(); r.status = {'active': True}
    def info(fx):
        return SimpleNamespace(width=1920, height=1080, distortion_model='plumb_bob',
            header=SimpleNamespace(frame_id='camera'), k=[fx, 0, 960, 0, 1388, 540, 0, 0, 1], d=[])
    r.camera_info = info(1386)
    r._camera_info(info(1386))
    assert not r.stop_event.is_set()
    r._camera_info(info(1000))
    assert r.stop_event.is_set()




def test_monitor_loss_cancels_active_goal(monkeypatch):
    from concurrent.futures import Future
    from action_msgs.msg import GoalStatus
    r = runner(); r.names = [f'joint{i}' for i in range(6)]
    r.lock = threading.RLock(); r.stop_event = threading.Event(); r.goal = None
    handle = Mock(accepted=True)
    pending = Future()
    handle.get_result_async.return_value = pending
    accepted = Future(); accepted.set_result(handle)
    r.action = SimpleNamespace(send_goal_async=lambda _: accepted)
    r.wait = lambda future, timeout: future.result()
    r.settle = Mock()
    monitor = Mock(side_effect=RuntimeError('lost'))
    with pytest.raises(RuntimeError, match='lost'):
        AutomaticCalibration.move(r, np.ones(6)*.2, monitor=monitor)
    handle.cancel_goal_async.assert_called_once()
    assert r.stop_event.is_set()
    r.settle.assert_not_called()


@pytest.mark.parametrize("interruption", [FramingCorrection, TrackingInterrupted])
def test_warning_waits_for_terminal_cancellation_before_replanning(interruption):
    from concurrent.futures import Future
    from action_msgs.msg import GoalStatus
    r = runner(); r.names = [f'joint{i}' for i in range(6)]
    r.lock = threading.RLock(); r.stop_event = threading.Event(); r.goal = None
    handle = Mock(accepted=True)
    pending = Future()
    handle.get_result_async.return_value = pending
    def cancel():
        pending.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
        result = Future(); result.set_result(SimpleNamespace()); return result
    handle.cancel_goal_async.side_effect = cancel
    accepted = Future(); accepted.set_result(handle)
    r.action = SimpleNamespace(send_goal_async=lambda _: accepted)
    r.wait = lambda future, timeout: future.result(timeout=.1)
    r.settle = Mock()
    with pytest.raises(interruption):
        AutomaticCalibration.move(r, np.ones(6)*.2, monitor=Mock(side_effect=interruption()))
    handle.cancel_goal_async.assert_called_once()
    r.settle.assert_called_once()
    assert pending.done()
    assert not r.stop_event.is_set()


def test_monitor_rejects_stalled_image_even_with_paused_ros_clock():
    r = monitor_runner(); r.monitor_at -= 1
    with pytest.raises(RuntimeError, match='stale'):
        r.motion_monitor()


def test_monitor_warning_and_hard_edge_are_distinct():
    r = monitor_runner()
    r.framing = SimpleNamespace(contains=Mock(side_effect=[True, False]))
    with pytest.raises(FramingCorrection):
        r.motion_monitor()
    r.framing.contains.side_effect = [False]
    with pytest.raises(RuntimeError, match='edge'):
        r.motion_monitor()


def monitor_runner(age=.1):
    from geometry_msgs.msg import TransformStamped
    from rclpy.time import Time
    from scipy.spatial.transform import Rotation
    r = runner(); r.visible = False  # Latest frame was rejected.
    r.monitor_stamp = 10000000000; r.monitor_at = time.monotonic()
    stamped = TransformStamped(); stamped.header.stamp = Time(seconds=10).to_msg()
    obs = observation(); t = stamped.transform
    t.translation.x, t.translation.y, t.translation.z = map(float, obs[:3, 3])
    t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w = map(float, Rotation.from_matrix(obs[:3, :3]).as_quat())
    r.node = SimpleNamespace(tracking_base_frame='camera', tracking_marker_frame='board',
        tf_buffer=SimpleNamespace(lookup_transform=lambda *a: stamped),
        get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=10+age)))
    return r


def test_rejected_frame_keeps_recent_valid_pose_during_motion():
    r = monitor_runner(.05)
    r.motion_monitor()  # No stop on a single rejected image.


@pytest.mark.parametrize('age', [.36, 1.0, -.2])
def test_rejected_frames_cannot_extend_pose_freshness(age):
    with pytest.raises(RuntimeError, match='stale'):
        monitor_runner(age).motion_monitor()


def test_paused_ros_clock_does_not_allow_stale_motion():
    r = monitor_runner(.05)
    r.motion_monitor(); r.monitor_at -= 1
    with pytest.raises(RuntimeError, match='stale'):
        r.motion_monitor()






def test_stationary_reacquisition_requires_multiple_fresh_frames():
    from geometry_msgs.msg import TransformStamped
    from rclpy.time import Time
    r = runner(); r.stop_event = threading.Event()
    r.fresh_board = Mock(return_value=observation())
    now = [10.]
    def stamped(*args):
        now[0] += .05
        msg = TransformStamped(); msg.header.stamp = Time(seconds=now[0]).to_msg()
        return msg
    r.node = SimpleNamespace(tracking_base_frame='camera', tracking_marker_frame='board',
        tf_buffer=SimpleNamespace(lookup_transform=stamped),
        get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=now[0])))
    r.wait_for_tracking()
    assert r.fresh_board.call_count >= 5


def test_stationary_reacquisition_times_out_on_repeated_old_frame(monkeypatch):
    r = monitor_runner(.05); r.stop_event = Mock()
    r.fresh_board = Mock(return_value=observation())
    ticks = iter(np.arange(0, 20, .5))
    monkeypatch.setattr(time, 'monotonic', lambda: next(ticks))
    with pytest.raises(RuntimeError, match='within 5 seconds'):
        r.wait_for_tracking()
