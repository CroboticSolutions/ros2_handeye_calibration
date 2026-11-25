"""Continuous pivot recording (tip seated in a cup, tool turned by hand)."""
import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from std_srvs.srv import Trigger

from hand_eye_calibration.continuous_pivot_node import quick_estimate, rotation_step_deg

TIP = np.array([0.040, 0.003, 0.177])      # flange -> tip (m)
CUP = np.array([0.30, -0.10, 0.05])        # fixed point in base_link (m)


def flange_pose(rotvec, noise=0.0, rng=None):
    r = Rotation.from_rotvec(rotvec) * Rotation.from_euler('y', 180, degrees=True)
    q = CUP - r.apply(TIP)
    if noise:
        q = q + rng.normal(0, noise, 3)
    return list(q) + list(r.as_quat())


def test_rotation_step_and_quick_estimate():
    a = flange_pose([0, 0, 0])
    b = flange_pose([0, np.radians(5), 0])
    assert rotation_step_deg(a, []) == float('inf')
    assert rotation_step_deg(b, [a[3:]]) == pytest.approx(5.0, abs=1e-6)
    rng = np.random.default_rng(1)
    samples = [flange_pose(rng.normal(0, 0.35, 3), 0.0002, rng) for _ in range(40)]
    est = quick_estimate(samples)
    np.testing.assert_allclose(est['tip'], TIP, atol=5e-4)
    assert est['rms_m'] < 0.001
    assert quick_estimate(samples[:3]) is None


@pytest.fixture
def node(tmp_path):
    from hand_eye_calibration.pivot_calibration_node import PivotCollector
    rclpy.init(args=['--ros-args', '-p', f'calibration_file:={tmp_path / "tcp.yaml"}',
                     '-p', f'dataset_dir:={tmp_path / "runs"}'])
    n = PivotCollector()
    # Synthetic stamps lie in the past (one 20 Hz sweep is faster than real time).
    n.max_tf_age_s = 1e6
    yield n
    n.destroy_node()
    rclpy.shutdown()


def feed(n, pose, t_ns):
    msg = TransformStamped()
    msg.header.frame_id = n.robot_base_frame
    msg.child_frame_id = n.robot_flange_frame
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(t_ns, 1_000_000_000)
    tr = msg.transform
    tr.translation.x, tr.translation.y, tr.translation.z = pose[:3]
    tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w = pose[3:]
    n.tf_buffer.set_transform(msg, 'test')
    n._cp_tick()


def test_recording_keeps_distinct_slow_poses_and_fits_on_stop(node):
    n = node
    n.samples['tip'] = [flange_pose([0.5, 0, 0])] * 3  # old spike touches are cleared
    assert n.cp_start_cb(Trigger.Request(), Trigger.Response()).success
    assert n.samples['tip'] == []
    assert not n.capture_point_cb(Trigger.Request(), Trigger.Response()).success

    rng = np.random.default_rng(3)
    t = n.get_clock().now().nanoseconds - 60 * 1_000_000_000
    dt = 50_000_000  # 20 Hz
    # Slow sweep: tilt about x, then y, then spin about the tool: 1 deg per tick = 20 deg/s.
    path = ([[np.radians(a), 0, 0] for a in range(-30, 31)]
            + [[0, np.radians(a), 0] for a in range(-30, 31)]
            + [[0, 0, np.radians(a)] for a in range(-40, 41)])
    for rv in path:
        feed(n, flange_pose(rv, 0.0002, rng), t)
        t += dt
    kept = len(n.samples['tip'])
    # 1 deg steps with a 3 deg minimum step: about a third of the poses are kept.
    assert 40 <= kept <= 80, kept
    # The jumps between the three sweeps are faster than 30 deg/s and are skipped.
    assert n.cp['too_similar'] > 0 and n.cp['too_fast'] == 2
    fast = n.cp['too_fast']
    status = n._continuous_status()
    assert status['active'] and status['estimate'] is not None

    # A fast flick is not kept.
    before = len(n.samples['tip'])
    feed(n, flange_pose([0.6, 0.6, 0]), t)
    assert len(n.samples['tip']) == before and n.cp['too_fast'] == fast + 1

    resp = n.cp_stop_cb(Trigger.Request(), Trigger.Response())
    assert resp.success and 'tip [' in resp.message, resp.message
    pivot = n._compute_round('tip')
    np.testing.assert_allclose(pivot['tcp_translation'], TIP, atol=5e-4)
    assert not n.cp['active']
    assert n.sample_metadata['tip'][0]['source'] == 'continuous'


def test_recording_stops_at_the_pose_limit(node):
    n = node
    n.set_parameters([rclpy.parameter.Parameter('continuous_max_samples', value=10)])
    n.cp_start_cb(Trigger.Request(), Trigger.Response())
    t = n.get_clock().now().nanoseconds - 60 * 1_000_000_000
    for a in range(0, 200, 1):
        feed(n, flange_pose([np.radians(a / 2), 0, 0]), t)
        t += 50_000_000
        if not n.cp['active']:
            break
    assert not n.cp['active'] and len(n.samples['tip']) == 10
    assert 'reached 10' in n.cp['message']
