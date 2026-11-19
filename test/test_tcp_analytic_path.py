"""TCP reorientation with analytic IK (tcp_motion_ik:=analytic, experimental) and the
unchanged MoveIt default."""
import inspect
import threading
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip('arm_ik')
from arm_ik import make_solver  # noqa: E402
from hand_eye_calibration import tcp_analytic_path as tap, tcp_quality  # noqa: E402
from hand_eye_calibration.tcp_reorientation import TcpMotion  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1]
PIPER = Path(__file__).resolve().parents[3] / 'robots/arm_ik/test/data/piper_welding_gun.urdf'
NAMES = [f'joint{i}' for i in range(1, 7)]
TCP = np.array([0.0, 0.03, 0.14])          # tip in link6
Q0 = np.array([0.1, 1.1, -1.0, 0.1, 0.9, -0.2])


@pytest.fixture(scope='module')
def solver():
    return make_solver('closed_form', PIPER.read_text(), 'base_link', 'link6')


def flange_pose(solver, q):
    T = solver.fk(q)
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()]


def reorientation_phases(current, tcp=TCP, angle_deg=20.0, clearance=.03, standoff=.005):
    """The waypoints run_step builds (spike = tip now, vertical)."""
    n = np.array([0., 0., 1.])
    tip = current[:3] + Rotation.from_quat(current[3:7]).apply(tcp)
    spike = tip - standoff * n
    target = (Rotation.from_rotvec(np.deg2rad(angle_deg) * np.array([1., 0., 0.]))
              * Rotation.from_quat(current[3:7])).as_quat()
    high, low = spike + clearance * n, spike + standoff * n
    lift = [tcp_quality.flange_pose_for_tip(spike + (standoff + s * (clearance - standoff)) * n, current[3:7], tcp)
            for s in np.linspace(0, 1, 4)[1:]]
    rotate = tcp_quality.tip_preserving_waypoints(current[3:7], target, high, tcp, steps=10)
    descend = [tcp_quality.flange_pose_for_tip(high + s * (low - high), target, tcp) for s in np.linspace(0, 1, 4)[1:]]
    return [lift, rotate, descend], high


def test_dense_poses_keep_the_tip_and_include_every_waypoint(solver):
    current = flange_pose(solver, Q0)
    (lift, rotate, descend), high = reorientation_phases(current)
    poses = tap.dense_flange_poses(lift[-1], rotate, TCP)
    tips = poses[:, :3, 3] + np.einsum('nij,j->ni', poses[:, :3, :3], TCP)
    np.testing.assert_allclose(tips, np.broadcast_to(high, tips.shape), atol=1e-9)   # rotation about the tip
    np.testing.assert_allclose(poses[-1], tap.pose_matrix(rotate[-1]), atol=1e-12)
    steps = [Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude() for a, b in zip(poses, poses[1:])]
    assert max(steps) <= tap.MAX_ROTATION_RAD + 1e-9 and len(poses) >= 20
    lifted = tap.dense_flange_poses(current, lift, TCP)
    assert np.max(np.linalg.norm(np.diff(lifted[:, :3, 3], axis=0), axis=1)) <= tap.MAX_TRANSLATION_M + 1e-9


def test_analytic_path_is_continuous_and_exact(solver):
    current = flange_pose(solver, Q0)
    phases, _ = reorientation_phases(current)
    q, previous, paths = Q0, current, []
    for waypoints in phases:
        poses = tap.dense_flange_poses(previous, waypoints, TCP)
        path = tap.solve_continuous(solver, poses, q)
        assert np.max(np.abs(np.diff(path, axis=0))) <= tap.MAX_JOINT_STEP_RAD
        for T, qi in zip(poses, path[1:]):
            assert np.linalg.norm(solver.fk(qi)[:3, 3] - T[:3, 3]) <= solver.model_tolerance_m + 1e-9
        paths.append(path)
        previous, q = waypoints[-1], path[-1]
    times, positions, velocities, accelerations = tap.timed_phases(paths, np.full(6, .18), .36)
    assert np.all(np.diff(times) > 0) and len(positions) == sum(len(p) - 1 for p in paths) + 1
    np.testing.assert_allclose(positions[0], Q0)
    assert np.abs(velocities[[0, -1]]).max() < 1e-9          # at rest at both ends
    assert np.abs(velocities).max() <= .18 + 1e-9 and np.abs(accelerations).max() <= .36 + 1e-6
    # The finite-difference speed agrees with the reported velocities (smooth timing).
    finite = np.abs(np.diff(positions, axis=0)) / np.diff(times)[:, None]
    assert finite.max() <= .18 * 1.05


def test_jump_and_unreachable_poses_reject_the_whole_path():
    far = np.r_[0., 0., 0., 0., 0., 1.0]
    jumping = NS(kind='fake', joint_names=NAMES,
                 nearest=lambda T, seed, max_step=None: None if max_step is not None else seed + far)
    with pytest.raises(tap.AnalyticPathError, match='Joint jump.*joint6 would move 57.3 deg') as info:
        tap.solve_continuous(jumping, np.tile(np.eye(4), (3, 1, 1)), np.zeros(6))
    assert info.value.index == 0
    calls = []
    unreachable = NS(kind='fake', joint_names=NAMES,
                     nearest=lambda T, seed, max_step=None: calls.append(max_step) or (seed + .01 if len(calls) < 3 else None))
    with pytest.raises(tap.AnalyticPathError, match='pose 3/4 has no IK solution'):
        tap.solve_continuous(unreachable, np.tile(np.eye(4), (4, 1, 1)), np.zeros(6))
    assert calls[0] == tap.MAX_JOINT_STEP_RAD                # the previous state seeds every solve


def motion(solver, mode='analytic', valid=lambda q: True):
    m = TcpMotion.__new__(TcpMotion)
    m.node = NS(get_logger=lambda: Mock())
    m.motion_ik, m.ik_backend, m.solver = mode, 'closed_form', solver
    m.base_frame, m.flange_frame, m.names, m.move_group = 'base_link', 'link6', NAMES, 'arm'
    m.velocity_limits = np.full(6, 3.0)
    m.stop_event, m.heartbeat_timeout, m.lock, m.goal = threading.Event(), 0.0, threading.RLock(), None
    m.state = {'state': 'idle', 'message': '', 'active': False}
    m.joints = NS(name=NAMES + ['gripper'], position=list(Q0) + [0.0])
    m.joint_positions = lambda: Q0.copy()
    m.wait = lambda value, timeout: value
    m.checked = []
    m.check = lambda: None                  # no rclpy context in unit tests

    def validity(req):
        q = np.array(req.robot_state.joint_state.position[:6])
        assert req.group_name == 'arm' and req.robot_state.is_diff
        m.checked.append(q)
        return NS(valid=valid(q))
    m.validity = NS(wait_for_service=lambda timeout_sec: True, call_async=validity)
    return m


def test_plan_analytic_checks_every_state_and_times_the_path(solver):
    rclpy = pytest.importorskip('rclpy')  # noqa: F841 - message types
    m = motion(solver)
    current = flange_pose(solver, Q0)
    phases, _ = reorientation_phases(current)
    trajectory = m.plan_analytic(current, phases, TCP)
    positions = np.array([p.positions for p in trajectory.points])
    assert trajectory.joint_names == NAMES
    np.testing.assert_allclose(positions[0], Q0)
    assert len(m.checked) == len(positions) - 1                        # every planned state
    np.testing.assert_allclose(np.array(m.checked), positions[1:])
    times = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in trajectory.points])
    assert np.all(np.diff(times) > 0)
    assert all(len(p.velocities) == 6 and len(p.accelerations) == 6 for p in trajectory.points)
    # speed_scale 0.1 of the URDF limits (3 rad/s) under move_planned's 0.18 rad/s cap; acceleration 0.1 x 0.36.
    assert max(max(map(abs, p.velocities)) for p in trajectory.points) <= .18 + 1e-9
    assert max(max(map(abs, p.accelerations)) for p in trajectory.points) <= .036 + 1e-6
    m.velocity_limits = np.full(6, 1.0)
    slow = m.plan_analytic(current, phases, TCP)
    assert max(max(map(abs, p.velocities)) for p in slow.points) <= .1 + 1e-9


def test_plan_analytic_refuses_an_invalid_state_or_a_model_mismatch(solver):
    pytest.importorskip('rclpy')
    current = flange_pose(solver, Q0)
    phases, _ = reorientation_phases(current)
    m = motion(solver, valid=lambda q: len(m.checked) < 5)
    with pytest.raises(RuntimeError, match='state 5/.*collision or invalid.*no motion was sent'):
        m.plan_analytic(current, phases, TCP)
    shifted = current.copy()
    shifted[2] += .01
    with pytest.raises(RuntimeError, match='disagrees with the measured flange'):
        motion(solver).plan_analytic(shifted, phases, TCP)


def run_step(m, solver):
    current = flange_pose(solver, Q0)
    tip = current[:3] + Rotation.from_quat(current[3:7]).apply(TCP)
    target = (Rotation.from_rotvec([0.3, 0, 0]) * Rotation.from_quat(current[3:7])).as_quat()
    m.configure = lambda: None
    m.plan_cartesian = Mock(return_value='cartesian')
    m.plan_analytic = Mock(return_value='analytic')
    m.move_planned = Mock()
    done = []
    m.run_step(current, target, TCP, tip - [0, 0, .005], [0, 0, 1], .03, .005, done.append)
    m.thread.join(5)
    return done


def test_moveit_stays_the_default(solver):
    assert inspect.signature(TcpMotion.__init__).parameters['motion_ik'].default == 'moveit'
    source = (PACKAGE / 'hand_eye_calibration/pivot_calibration_node.py').read_text()
    assert "self.declare_parameter('tcp_motion_ik', 'moveit')" in source
    launch = (PACKAGE / 'launch/tcp_calibration.launch.py').read_text()
    assert "{'tcp_motion_ik'" not in launch          # not wired into the launch/GUI path
    m = motion(solver, mode='moveit')
    assert run_step(m, solver) == [True]
    m.plan_analytic.assert_not_called()
    (waypoints,), _ = m.plan_cartesian.call_args
    assert len(waypoints) == 3 + 10 + 3                       # the unchanged Cartesian request
    m.move_planned.assert_called_once_with('cartesian')


def test_analytic_mode_plans_the_same_waypoints(solver):
    moveit, analytic = motion(solver, mode='moveit'), motion(solver)
    run_step(moveit, solver)
    assert run_step(analytic, solver) == [True]
    moveit.plan_analytic.assert_not_called()
    analytic.plan_cartesian.assert_not_called()
    (current, phases, tcp), _ = analytic.plan_analytic.call_args
    (waypoints,), _ = moveit.plan_cartesian.call_args
    np.testing.assert_allclose(np.concatenate(phases), np.array(waypoints))
    analytic.move_planned.assert_called_once_with('analytic')


def test_motion_modes_and_backends_are_validated():
    assert tap.normalize_tcp_motion_ik(None) == 'moveit' and tap.normalize_tcp_motion_ik('Analytic') == 'analytic'
    with pytest.raises(ValueError):
        tap.normalize_tcp_motion_ik('kdl')
    m = TcpMotion.__new__(TcpMotion)
    m.solver, m.ik_backend = None, 'service'
    with pytest.raises(RuntimeError, match="needs an analytic ik_backend"):
        m.analytic_solver()
