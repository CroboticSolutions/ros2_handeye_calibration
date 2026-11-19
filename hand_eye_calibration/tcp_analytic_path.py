"""Analytic-IK joint path for the TCP reorientation (``tcp_motion_ik:=analytic``).

Experimental, off by default: the default (``moveit``) asks MoveIt
``/compute_cartesian_path`` (numeric KDL IK) for the same flange waypoints.

The flange waypoints of each phase (lift, rotate about the tip, descend) are
densified in tip space: the tip moves on straight lines and the tool orientation is
slerped, so the fitted tip stays exactly where the Cartesian request puts it, at
most ``max_translation_m`` / ``max_rotation_rad`` per step. Every dense flange pose
is solved with ``arm_ik`` ``solver.nearest(T, previous_q, max_step)`` (branch and
winding nearest the previous state), so the path is joint-continuous; a step that
needs a larger joint change (branch switch, singularity) rejects the whole path.
Each phase gets its own minimum-jerk time scaling (zero velocity and acceleration
at the phase ends, where the path has corners) within the joint velocity and
acceleration limits. The caller checks every state with GetStateValidity and
executes through the same ``move_planned`` as the MoveIt path.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

TCP_MOTION_IK_MODES = ('moveit', 'analytic')
MAX_TRANSLATION_M = 0.005          # = the Cartesian request's max_step
MAX_ROTATION_RAD = np.deg2rad(1.0)
MAX_JOINT_STEP_RAD = 0.10          # largest joint change between two dense poses
MIN_PHASE_SECONDS = 0.5
# Minimum-jerk s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5: peak ds/dtau, d2s/dtau2.
_PEAK_VELOCITY = 1.875
_PEAK_ACCELERATION = 10.0 / np.sqrt(3.0)


class AnalyticPathError(RuntimeError):
    """The analytic reorientation path is infeasible; ``index`` is the dense pose."""

    def __init__(self, message, index=None):
        super().__init__(message)
        self.index = index


def normalize_tcp_motion_ik(value):
    mode = str(value if value is not None else 'moveit').strip().lower() or 'moveit'
    if mode not in TCP_MOTION_IK_MODES:
        raise ValueError(f'tcp_motion_ik {value!r} is not one of {TCP_MOTION_IK_MODES}')
    return mode


def pose_matrix(pose):
    """4x4 of a [x y z qx qy qz qw] pose."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(pose[3:7], float)).as_matrix()
    T[:3, 3] = np.asarray(pose[:3], float)
    return T


def dense_flange_poses(start_flange, waypoints, tcp_translation, max_translation_m=MAX_TRANSLATION_M,
                       max_rotation_rad=MAX_ROTATION_RAD):
    """(N, 4, 4) flange poses from ``start_flange`` (excluded) through every waypoint
    ([x y z qx qy qz qw], each included exactly), the tip linear and the orientation
    slerped between consecutive waypoints."""
    t = np.asarray(tcp_translation, float)
    poses = []
    previous = np.asarray(start_flange, float)
    for waypoint in waypoints:
        waypoint = np.asarray(waypoint, float)
        key = Rotation.from_quat([previous[3:7], waypoint[3:7]])
        tip_a = previous[:3] + key[0].apply(t)
        tip_b = waypoint[:3] + key[1].apply(t)
        angle = (key[0].inv() * key[1]).magnitude()
        steps = int(max(1, np.ceil(max(np.linalg.norm(tip_b - tip_a) / max_translation_m,
                                       np.linalg.norm(waypoint[:3] - previous[:3]) / max_translation_m,
                                       angle / max_rotation_rad) - 1e-9)))
        slerp = Slerp([0.0, 1.0], key)
        for s in np.arange(1, steps + 1) / steps:
            T = np.eye(4)
            if s == 1.0:
                T = pose_matrix(waypoint)
            else:
                rotation = slerp([s])[0]
                T[:3, :3] = rotation.as_matrix()
                T[:3, 3] = tip_a + s * (tip_b - tip_a) - rotation.apply(t)
            poses.append(T)
        previous = waypoint
    return np.array(poses).reshape(-1, 4, 4)


def solve_continuous(solver, poses, start_q, max_step=MAX_JOINT_STEP_RAD):
    """(N + 1, n) joint path: ``start_q`` then, per pose, ``solver.nearest(T, previous, max_step)``.
    Raises AnalyticPathError on an unreachable pose or a joint jump above ``max_step``."""
    path = [np.asarray(start_q, float)]
    for index, T in enumerate(poses):
        q = solver.nearest(T, path[-1], max_step)
        if q is None:
            nearest = solver.nearest(T, path[-1])
            if nearest is None:
                raise AnalyticPathError(f'Dense reorientation pose {index + 1}/{len(poses)} has no IK solution '
                                        f'within the joint limits ({solver.kind}); no motion was sent.', index)
            jump = np.abs(nearest - path[-1])
            joint = int(np.argmax(jump))
            raise AnalyticPathError(
                f'Joint jump at dense reorientation pose {index + 1}/{len(poses)}: {solver.joint_names[joint]} '
                f'would move {np.degrees(jump[joint]):.1f} deg (limit {np.degrees(max_step):.1f} deg; branch switch '
                'or singularity); no motion was sent.', index)
        path.append(np.asarray(q, float))
    return np.array(path)


def min_jerk_timing(path, velocity_limits, acceleration_limits, min_seconds=MIN_PHASE_SECONDS):
    """(times, velocities, accelerations) for an (M, n) joint path at rest at both
    ends: s = normalised cumulative max-joint distance, s(t) minimum jerk with a
    duration that keeps |dq/ds| ds/dt and |dq/ds| d2s/dt2 within the limits."""
    path = np.asarray(path, float)
    n = path.shape[1]
    step = np.max(np.abs(np.diff(path, axis=0)), axis=1)
    length = float(step.sum())
    if len(path) < 2 or length <= 1e-12:
        raise AnalyticPathError('Analytic reorientation phase has no joint motion.')
    if np.any(step <= 1e-12):
        raise AnalyticPathError('Analytic reorientation path repeats a joint state.')
    s = np.r_[0.0, np.cumsum(step)] / length
    vmax = np.broadcast_to(np.asarray(velocity_limits, float), (n,))
    amax = np.broadcast_to(np.asarray(acceleration_limits, float), (n,))
    dq = np.gradient(path, s, axis=0, edge_order=1)            # rad per unit s
    ddq = np.gradient(dq, s, axis=0, edge_order=1)
    slope = np.max(np.abs(dq), axis=0)
    seconds = max(min_seconds, float(np.max(_PEAK_VELOCITY * slope / vmax)),
                  float(np.sqrt(np.max(_PEAK_ACCELERATION * slope / amax))))
    tau_grid = np.linspace(0.0, 1.0, 20001)
    s_grid = tau_grid**3 * (10 - 15 * tau_grid + 6 * tau_grid**2)
    tau = np.interp(s, s_grid, tau_grid)
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / seconds
    sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / seconds**2
    velocities = dq * sd[:, None]
    accelerations = ddq * sd[:, None]**2 + dq * sdd[:, None]
    # The curvature term (ddq) is not bounded above: stretch time until it fits.
    peak = float(np.max(np.abs(accelerations) / amax))
    if peak > 1.0:
        k = np.sqrt(peak)
        seconds *= k
        velocities /= k
        accelerations /= k**2
    return tau * seconds, velocities, accelerations


def timed_phases(phases, velocity_limits, acceleration_limits):
    """One trajectory (times, positions, velocities, accelerations) from consecutive
    joint phases (each (M_i, n), phase i + 1 starting at the last state of phase i),
    each timed by ``min_jerk_timing``."""
    times, positions, velocities, accelerations = [0.0], [phases[0][0]], [np.zeros(len(phases[0][0]))], \
        [np.zeros(len(phases[0][0]))]
    for phase in phases:
        if len(phase) < 2:
            continue
        t, v, a = min_jerk_timing(phase, velocity_limits, acceleration_limits)
        offset = times[-1]
        for i in range(1, len(phase)):
            times.append(offset + float(t[i]))
            positions.append(phase[i])
            velocities.append(v[i])
            accelerations.append(a[i])
    return np.array(times), np.array(positions), np.array(velocities), np.array(accelerations)
