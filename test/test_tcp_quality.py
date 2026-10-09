import math

import numpy as np
import pytest
import rclpy
import yaml
from scipy.spatial.transform import Rotation
from std_srvs.srv import Trigger

from hand_eye_calibration import tcp_quality
from hand_eye_calibration.pivot_backend import PivotCalibrationBackend
from hand_eye_calibration.tcp_offline_solve_cli import joint_offset_fit

TIP = np.array([0.0597, -0.0028, 0.1902])
SPIKE = np.array([0.29, -0.17, 0.05])
# First welding-gun TCP (2026-09-23): 4 touches, rotations mostly about one axis.
TCP_V1_TOUCHES = [
    [0.22627126059338637, -0.11708103710698556, 0.23322654961843325,
     0.21139763616796628, 0.913192491871849, -0.1123172804307989, 0.3298110682316602],
]


def touches(n, spread_deg=35., seed=0, noise=0.0, axes=3, minor_deg=3.):
    rng = np.random.default_rng(seed)
    base = Rotation.from_euler('xyz', [180, -20, 0], degrees=True)
    out = []
    for _ in range(n):
        v = rng.uniform(-1, 1, 3) * np.radians(spread_deg)
        v[axes:] *= minor_deg / spread_deg
        r = Rotation.from_rotvec(v) * base
        pos = SPIKE - r.apply(TIP) + rng.normal(0, noise, 3)
        out.append(list(pos) + list(r.as_quat()))
    return out


def test_spans_are_basis_independent_and_catch_single_axis_rotation():
    diagonal = Rotation.from_rotvec(np.ones(3) / math.sqrt(3))
    samples = [[0, 0, 0] + list((Rotation.from_rotvec(diagonal.apply([0, 0, 1]) * a) ).as_quat())
               for a in np.radians([-40, -10, 20, 45])]
    spans = tcp_quality.rotation_spans_deg(samples)
    assert spans[0] > 80 and spans[1] < 1


def test_gate_rejects_single_axis_set_and_accepts_diverse_validated_set():
    single = touches(8, axes=1)
    pivot = PivotCalibrationBackend.compute_pivot(single)
    verdict = tcp_quality.evaluate(pivot, tcp_quality.rotation_spans_deg(single), None)
    assert not verdict['passed']
    assert any(c['name'] == 'second_axis_span_deg' and not c['passed'] for c in verdict['checks'])
    good = touches(10, noise=0.0002)
    pivot = PivotCalibrationBackend.compute_pivot(good)
    validation = PivotCalibrationBackend.validate_pivot(touches(3, seed=5, noise=0.0002), pivot)
    verdict = tcp_quality.evaluate(pivot, tcp_quality.rotation_spans_deg(good), validation)
    assert verdict['passed'], verdict['summary']
    assert not tcp_quality.evaluate(pivot, tcp_quality.rotation_spans_deg(good), None)['passed']


def test_bend_plane_frame_puts_x_in_the_neck_plane():
    axis = [math.sin(math.radians(35)), 0, math.cos(math.radians(35))]
    frame = tcp_quality.bend_plane_frame(axis)
    m = np.array(frame['rotation_matrix'])
    np.testing.assert_allclose(m[:, 2], axis, atol=1e-12)
    assert abs(m[1, 0]) < 1e-12 and m[2, 0] > 0  # X in the x-z plane, towards flange +Z
    assert tcp_quality.bend_plane_frame([0, 0, 1]) is None
    assert tcp_quality.cad_axis_deviation_deg(axis, 35.) == pytest.approx(0, abs=1e-9)


def test_reorientation_waypoints_keep_the_tip_fixed():
    ref = Rotation.from_euler('xyz', [180, -20, 0], degrees=True).as_quat()
    targets = tcp_quality.reorientation_targets(ref)
    assert len(targets) == 6
    angles = [math.degrees((Rotation.from_quat(ref).inv() * Rotation.from_quat(q)).magnitude()) for q in targets]
    np.testing.assert_allclose(angles, [25, 25, 25, 25, 45, 45], atol=1e-6)
    high = SPIKE + [0, 0, .03]
    for w in tcp_quality.tip_preserving_waypoints(ref, targets[0], high, TIP):
        np.testing.assert_allclose(w[:3] + Rotation.from_quat(w[3:]).apply(TIP), high, atol=1e-12)


class Arm:
    names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']

    def camera(self, q):
        def t(rv, p):
            m = np.eye(4); m[:3, :3] = Rotation.from_rotvec(rv).as_matrix(); m[:3, 3] = p; return m
        m = t([0, 0, q[0]], [0, 0, .12]) @ t([0, q[1], 0], [0, 0, 0]) @ t([0, q[2], 0], [.28, 0, 0])
        return m @ t([q[3], 0, 0], [.25, 0, 0]) @ t([0, q[4], 0], [0, 0, 0]) @ t([q[5], 0, 0], [.06, 0, 0])


def test_joint_offsets_explain_what_the_pivot_fit_cannot():
    from scipy.optimize import least_squares
    arm, rng = Arm(), np.random.default_rng(2)
    true = np.zeros(6); true[2] = math.radians(1.0)
    tip, spike = np.array([.05, 0, .19]), np.array([.45, 0, 0])
    joints, samples = [], []
    q = np.array([0, .3, -.9, 0, 1.2, 0.])
    while len(joints) < 14:
        target = Rotation.from_rotvec(rng.uniform(-.6, .6, 3)) * Rotation.from_matrix(arm.camera(q)[:3, :3])
        def res(x):
            f = arm.camera(x + true)
            return np.r_[f[:3, :3] @ tip + f[:3, 3] - spike, (Rotation.from_matrix(f[:3, :3]).inv() * target).as_rotvec()]
        sol = least_squares(res, q)
        if np.linalg.norm(res(sol.x)) > 1e-6:
            continue
        joints.append(dict(zip(arm.names, sol.x)))
        f = arm.camera(sol.x)  # what the controller reports (no offset)
        samples.append(list(f[:3, 3]) + list(Rotation.from_matrix(f[:3, :3]).as_quat()))
    plain = PivotCalibrationBackend.compute_pivot(samples)
    fit = joint_offset_fit(arm, arm.names, joints, prior_sigma_rad=math.radians(3),
                           initial=np.r_[plain['tcp_translation'], plain['fixed_point']])
    assert fit['rms_residual_m'] < plain['rms_residual_m'] / 3
    assert fit['offsets_deg']['j3'] == pytest.approx(1.0, abs=.2)


@pytest.fixture
def node(tmp_path):
    from hand_eye_calibration.pivot_calibration_node import PivotCollector
    cal = tmp_path / 'tcp.yaml'
    cal.write_text('previous: true\n')
    rclpy.init(args=['--ros-args', '-p', f'calibration_file:={cal}', '-p', f'dataset_dir:={tmp_path / "runs"}'])
    n = PivotCollector()
    yield n, cal
    n.destroy_node()
    rclpy.shutdown()


def test_node_rejects_the_first_gun_tcp_pattern_and_accepts_a_good_run(node):
    n, cal = node
    n.samples['tip'] = touches(4, axes=1, noise=0.001)
    n.sample_metadata['tip'] = [{}] * 4
    response = n.save_calibration_cb(Trigger.Request(), Trigger.Response())
    assert not response.success and 'Acceptance failed' in response.message
    assert cal.read_text() == 'previous: true\n'
    rejected = yaml.safe_load(open(str(cal) + '.rejected.yaml'))
    assert (tmp := rejected['dataset_path']) and not rejected['acceptance']['passed']

    n.samples['tip'] = touches(10, noise=0.0002)
    n.sample_metadata['tip'] = [{}] * 10
    n.samples['validation'] = touches(3, seed=7, noise=0.0002)
    n.sample_metadata['validation'] = [{}] * 3
    response = n.save_calibration_cb(Trigger.Request(), Trigger.Response())
    assert response.success, response.message
    saved = yaml.safe_load(cal.read_text())
    assert saved['acceptance']['passed']
    np.testing.assert_allclose([saved['transform'][k] for k in ('tx', 'ty', 'tz')], TIP, atol=5e-4)
    from hand_eye_calibration.tcp_offline_solve_cli import load, solve
    out = solve(load(saved['dataset_path']))
    assert out['acceptance']['passed']


def test_nominal_neck_frame_tilts_the_flange_axis_towards_the_tip():
    frame = tcp_quality.nominal_neck_frame(22.0, TIP)
    m = np.array(frame['rotation_matrix'])
    np.testing.assert_allclose(m.T @ m, np.eye(3), atol=1e-12)
    side = np.array([TIP[0], TIP[1], 0.0]) / np.linalg.norm(TIP[:2])
    assert m[:, 2] @ side == pytest.approx(math.sin(math.radians(22)))
    assert tcp_quality.cad_axis_deviation_deg(m[:, 2], 22.0) == pytest.approx(0, abs=1e-9)
    assert frame['azimuth_deg'] == pytest.approx(math.degrees(math.atan2(TIP[1], TIP[0])))
    assert tcp_quality.nominal_neck_frame(0.0, TIP)['quaternion'] == [0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        tcp_quality.nominal_neck_frame(22.0, [0.0, 0.0, 0.3])


def test_cad_axis_deviation_is_reported_but_never_gates():
    good = touches(10, noise=0.0002)
    pivot = PivotCalibrationBackend.compute_pivot(good)
    validation = PivotCalibrationBackend.validate_pivot(touches(3, seed=5, noise=0.0002), pivot)
    axis = {'sample_count': 3, 'alignment_spread_deg': 0.2}
    verdict = tcp_quality.evaluate(pivot, tcp_quality.rotation_spans_deg(good), validation,
                                   axis=axis, axis_mode=True, cad_deviation=40.0)
    assert verdict['passed'], verdict['summary']
    assert all(c['name'] != 'cad_axis_deviation_deg' for c in verdict['checks'])


def test_node_saves_orientation_from_the_nominal_neck_angle(node):
    n, cal = node
    n.nominal_neck_angle_deg = 22.0
    n.samples['tip'] = touches(10, noise=0.0002)
    n.sample_metadata['tip'] = [{}] * 10
    n.samples['validation'] = touches(3, seed=7, noise=0.0002)
    n.sample_metadata['validation'] = [{}] * 3
    response = n.save_calibration_cb(Trigger.Request(), Trigger.Response())
    assert response.success, response.message
    saved = yaml.safe_load(cal.read_text())
    assert saved['calibration_mode'] == 'nominal'
    assert saved['axis_calibration']['method'] == 'nominal_neck_angle'
    q = [saved['transform'][k] for k in ('qx', 'qy', 'qz', 'qw')]
    z = Rotation.from_quat(q).as_matrix()[:, 2]
    assert math.degrees(math.acos(z[2])) == pytest.approx(22.0, abs=0.01)
