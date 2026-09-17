"""Bootstrap never receives the virtual camera's ground-truth extrinsic."""
from unittest.mock import Mock
import numpy as np
import pytest

from hand_eye_calibration import bootstrap_calibration
from hand_eye_calibration.automatic_calibration import TrackingInterrupted, Stopped
from test_automatic_calibration import synthetic_sequence


def session(monkeypatch):
    runner, node, _ = synthetic_sequence(monkeypatch)
    runner.names = node.get_parameter('auto_joint_names').value
    runner.lower = np.full(6, -3.); runner.upper = np.full(6, 3.)
    return bootstrap_calibration.BootstrapSession(runner, None), runner, node


def test_initial_transform_is_not_requested_and_checks_never_become_training(monkeypatch):
    r, n, _ = synthetic_sequence(monkeypatch)
    r.run()
    assert r.status['state'] == 'completed'
    r.tf.assert_not_called()
    assert not r.ik.mock_calls
    assert len(n.robot_samples) == len(r.session.views) == 15
    assert r.status['validation'] is None


def test_incorrect_first_estimate_is_not_used_to_predict_motion_or_saved(monkeypatch):
    r, n, _ = synthetic_sequence(monkeypatch)
    n.get_calibration = lambda: [1., 2., 3., 0., 0., 0., 1.]
    r.run()
    assert r.status['state'] == 'failed'
    assert r.session.estimate is None
    n.save_calibration_service_callback.assert_not_called()
    r.tf.assert_not_called()


def test_camera_frame_mismatch_fails_before_motion(monkeypatch):
    r, n, _ = synthetic_sequence(monkeypatch)
    r.camera_info.header.frame_id = 'unrelated_frame'
    r.run()
    assert 'same optical frame' in r.status['message']
    r.move.assert_not_called()


def test_collision_is_reported_as_collision_and_sends_no_motion(monkeypatch):
    s, r, _ = session(monkeypatch)
    r.collision_free_path = lambda *a: False
    assert not s.step(np.full(6, .01))
    assert 'collision' in s.reason
    r.move.assert_not_called()


def test_oversized_increment_is_never_sent(monkeypatch):
    s, r, _ = session(monkeypatch)
    with pytest.raises(RuntimeError, match='step limit'):
        s.step(np.full(6, .05))
    r.move.assert_not_called()


def test_lost_tracking_stops_exploration_in_that_direction(monkeypatch):
    s, r, _ = session(monkeypatch)
    r.move = Mock(side_effect=TrackingInterrupted())
    r.wait_for_tracking = Mock()
    assert not s.step(np.full(6, .01))
    r.wait_for_tracking.assert_called_once()
    r.move.assert_called_once()  # no blind retry toward an unknown view


def test_no_return_motion_when_tracking_does_not_recover(monkeypatch):
    s, r, _ = session(monkeypatch)
    r.move = Mock(side_effect=TrackingInterrupted())
    r.wait_for_tracking = Mock(side_effect=RuntimeError('Board absent'))
    with pytest.raises(RuntimeError, match='Board absent'):
        s.step(np.full(6, .01), retreat=True)
    r.move.assert_called_once()


def test_stop_interrupts_reacquisition_without_return_motion(monkeypatch):
    s, r, _ = session(monkeypatch)
    r.move = Mock(side_effect=TrackingInterrupted())
    r.wait_for_tracking = Mock(side_effect=Stopped())
    with pytest.raises(Stopped):
        s.step(np.full(6, .01))
    r.move.assert_called_once()


def test_recorded_piper_joint_geometry_with_hidden_camera_mount(monkeypatch):
    import json
    from pathlib import Path
    from hand_eye_calibration.bootstrap_calibration import sample_pose
    fixture = Path(__file__).parent/'fixtures'
    profile = {'urdf':(fixture/'piper_calibration_chain.urdf').read_text(),
               'joints':json.loads((fixture/'piper_calibration_joints.json').read_text())}
    r, n, truth = synthetic_sequence(monkeypatch, robot_profile=profile)
    r.run()
    assert r.status['state'] == 'completed', r.status
    np.testing.assert_allclose(sample_pose(n.get_calibration()), truth, atol=1e-5)
    assert r.move.call_count < 180  # bounded route with richer view selection
    assert all(not np.allclose(call.args[0], profile['joints'], atol=.003)
               for call in r.move.call_args_list)  # no repeated returns to the start
    r.tf.assert_not_called()


@pytest.mark.parametrize('distance', [.002, .02, .18])
def test_bootstrap_timing_respects_real_joint_speed_and_acceleration(distance):
    from types import SimpleNamespace
    from test_automatic_calibration import bare_runner
    r = bare_runner(); r.check_collisions = True; r.names = ['joint1']
    r.check = lambda: None; r.joint_positions = lambda: np.zeros(1)
    r.collision_free_path = lambda *a: True
    sent = []
    r.action = SimpleNamespace(send_goal_async=lambda goal:(sent.append(goal) or SimpleNamespace(add_done_callback=lambda _:None)))
    r.wait = Mock(side_effect=Stopped())
    with pytest.raises(Stopped):
        r.move(np.array([distance]), speed_scale=1., minimum_duration=.5)
    stamp = sent[0].trajectory.points[-1].time_from_start
    duration = stamp.sec + stamp.nanosec/1e9
    assert 1.5*distance/duration <= .06 + 1e-8
    assert 6*distance/duration**2 <= .12 + 1e-8
