import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hand_eye_calibration.visibility import BoardFraming, CameraKinematics, visible_joint_path
from hand_eye_calibration.automatic_calibration import AutomaticCalibration, FramingCorrection


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
    r.check = lambda: None
    r.update = Mock()
    r.framing = framing()
    r.optical_tracking = np.eye(4)
    r.kinematics = SimpleNamespace(camera=lambda q: np.eye(4))
    r.observed_board = Mock(return_value=observation())
    state = [np.zeros(6)]
    r.joint_positions = lambda: state[0].copy()
    r.move = Mock(side_effect=lambda q, **kwargs: state.__setitem__(0, q.copy()))
    r.path_fits = Mock(return_value=True)
    return r


def test_guarded_motion_uses_one_trajectory_and_observes_at_destination():
    r = runner(); target = np.ones(6)*.6
    r.guarded_move(target)
    r.move.assert_called_once()
    np.testing.assert_allclose(r.move.call_args.args[0], target)
    assert callable(r.move.call_args.kwargs['monitor'])
    assert r.observed_board.call_count == 2


def test_detection_loss_stops_before_further_motion():
    r = runner()
    r.observed_board.side_effect = [observation(), RuntimeError('lost')]
    with pytest.raises(RuntimeError, match='lost'):
        r.guarded_move(np.ones(6)*.2)
    r.move.assert_called_once()


def test_invalid_correction_sends_no_trajectory():
    r = runner(); r.path_fits.return_value = False
    r.solve = Mock(return_value=None)
    with pytest.raises(RuntimeError, match='No reachable correction'):
        r.guarded_move(np.ones(6)*.2)
    r.move.assert_not_called()


def test_reachable_correction_is_checked_before_motion():
    r = runner()
    r.path_fits.side_effect = [False, True, True]
    r.solve = Mock(return_value=np.ones(6)*.08)
    r.guarded_move(np.ones(6)*.2)
    r.solve.assert_called_once()
    np.testing.assert_allclose(r.move.call_args.args[0], np.ones(6)*.08)
    assert r.path_fits.call_count == 2


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


def test_warning_replans_after_cancelled_motion_and_stops_at_target():
    r = runner()
    r.move.side_effect = [FramingCorrection(), None]
    r.solve = Mock(return_value=np.ones(6)*.15)
    r.guarded_move(np.ones(6)*.2)
    assert r.move.call_count == 2
    r.solve.assert_called_once()
    assert r.observed_board.call_args_list[1].kwargs['recovering'] is True


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


def test_warning_waits_for_terminal_cancellation_before_replanning():
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
    with pytest.raises(FramingCorrection):
        AutomaticCalibration.move(r, np.ones(6)*.2, monitor=Mock(side_effect=FramingCorrection()))
    handle.cancel_goal_async.assert_called_once()
    r.settle.assert_called_once()
    assert pending.done()
    assert not r.stop_event.is_set()


def monitor_runner():
    from builtin_interfaces.msg import Time
    r = runner(); r.monitor_stamp = 1000000000; r.monitor_at = time.monotonic()
    r.fresh_board = Mock(return_value=observation())
    stamped = SimpleNamespace(header=SimpleNamespace(stamp=Time(sec=1)))
    r.node = SimpleNamespace(tf_buffer=SimpleNamespace(lookup_transform=lambda *a: stamped),
        tracking_base_frame='camera', tracking_marker_frame='board',
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1100000000)))
    return r


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
