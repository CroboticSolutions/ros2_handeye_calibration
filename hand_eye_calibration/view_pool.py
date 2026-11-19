"""Main/validation views for automatic hand-eye acquisition.

One analytic IK for every six-joint camera chain: ``arm_ik.make_solver(kind, ...)`` with
``kind`` from the node parameter ``ik_backend`` ('eaik' UR, 'ssik' CRX and the default,
'closed_form' PiPER; 'analytic' = deprecated alias of 'ssik'; 'service' = no analytic
pool, the local random sampler). Every kind returns all principal branches, FK-checked
against the URDF chain; the winding nearest the reference is taken here. A camera before
the last joint (PiPER link5, UR wrist_2_link) cannot reach arbitrary 6-D poses; its
views come from sampled joints (``joint_space_views``). Both share the limits below.

History. The local random sampler (+-6 cm, tilt +-15 deg) barely rotated the camera,
so the camera translation along the flange axis was weakly observable (D435 13.10.,
CRX 21.10.: position sigma 2.7-6.7 mm). A wide shell around the board (0.45 m, roll
+-90 deg) with joint windows of +-60/90 deg fixed that, but the greedy observability
choice always took the extremes of whatever was allowed: CRX sim J6 up to 362 deg,
real UR10 4.11. flange span 0.8 m, wrist_3 171 deg, 179 deg between two views.

Limits (``ViewLimits``) are relative to the reference (start) view, so the same
numbers hold on every robot:
- the camera orbits the board centre: optical axis tilt <= ``max_tilt_deg``,
  roll about it <= ``max_roll_deg``, distance 0.8-1.15 x the reference distance,
  at most 50 deg off the board normal, within ``max_excursion`` (m);
- joints stay in a window around the reference (first three joints
  ``max_arm_joint_change_deg``, the others ``max_wrist_joint_change_deg``) and a
  six-joint wrist does not flip J5 through 0;
- selection: greedy weakest-direction observability; with ``selection_tolerance`` > 0
  the smallest joint move among candidates within that fraction of the best gain
  (off by default: 0.10 raised the offline sigma 40-80 %).
Offline comparison: brain experiments/2026-11-05_handeye_view_limits.

Kinematics only: MoveIt still plans and collision-checks each chosen view.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

JOINT_LIMIT_MARGIN_RAD = 0.10
DISTANCE_SCALE = (0.8, 1.15)          # camera-board distance relative to the reference view
MIN_ELEVATION_DEG = 40.0              # camera direction above the board plane
BATCH = 250                           # orbit samples per ik_many batch
MAX_CANDIDATES = 5000
POOL_PER_VIEW = 3                     # feasible candidates wanted per requested view
ARM_JOINTS = 3                        # joints limited by max_arm_joint_change_deg
WRIST_BEND_JOINT = 4                  # J5: stays on the reference's side of 0
WRIST_CLEAR_RAD = np.deg2rad(10.)
FK_CHECK_TOLERANCE = 1e-6             # m and matrix elements: solver chain vs the runner's effector FK

IK_BACKENDS = ('service', 'eaik', 'ssik', 'closed_form')
IK_BACKEND_ALIASES = {'analytic': 'ssik'}   # deprecated name of the ssik backend
DEFAULT_IK_BACKEND = 'ssik'


def normalize_ik_backend(value):
    """``ik_backend`` parameter value -> one of IK_BACKENDS (shared with hand_teleop
    and launch_server); ValueError for anything else."""
    kind = str(value if value is not None else DEFAULT_IK_BACKEND).strip().lower() or DEFAULT_IK_BACKEND
    kind = IK_BACKEND_ALIASES.get(kind, kind)
    if kind not in IK_BACKENDS:
        raise ValueError(f'ik_backend {value!r} is not one of {IK_BACKENDS} (or the alias \'analytic\')')
    return kind


class ViewPoolUnavailable(ValueError):
    """The analytic view pool cannot be built for this chain; the message says why."""


@dataclass(frozen=True)
class ViewLimits:
    max_tilt_deg: float = 35.0
    max_roll_deg: float = 45.0
    max_arm_joint_change_deg: float = 45.0
    max_wrist_joint_change_deg: float = 60.0
    selection_tolerance: float = 0.0     # > 0 traded too much sigma offline (2026-11-05)

    def joint_window(self, count):
        window = np.full(count, np.deg2rad(self.max_wrist_joint_change_deg))
        window[:min(ARM_JOINTS, count)] = np.deg2rad(self.max_arm_joint_change_deg)
        return window


def joint_window_bounds(initial, lower, upper, limits, active=None):
    """Joint limits (margin already applied) intersected with the window around
    ``initial``; a six-joint wrist keeps J5 on the side of 0 it starts on.
    Inactive joints (not moving the camera) are pinned to ``initial``."""
    initial = np.asarray(initial, float)
    window = limits.joint_window(len(initial))
    lo = np.maximum(lower, initial - window)
    hi = np.minimum(upper, initial + window)
    if len(initial) == 6:
        bend = initial[WRIST_BEND_JOINT]
        if bend <= -WRIST_CLEAR_RAD:
            hi[WRIST_BEND_JOINT] = min(hi[WRIST_BEND_JOINT], -WRIST_CLEAR_RAD)
        elif bend >= WRIST_CLEAR_RAD:
            lo[WRIST_BEND_JOINT] = max(lo[WRIST_BEND_JOINT], WRIST_CLEAR_RAD)
    if active is not None:
        lo = np.where(active, lo, initial)
        hi = np.where(active, hi, initial)
    return lo, hi


def tilt_and_roll(reference, cameras):
    """(tilt, roll) in rad of camera poses (N, 4, 4) relative to ``reference``: angle
    between the optical axes and the twist about the optical axis (swing-twist)."""
    relative = np.einsum('ji,njk->nik', reference[:3, :3], np.asarray(cameras)[:, :3, :3])
    tilt = np.arccos(np.clip(relative[:, 2, 2], -1.0, 1.0))
    quat = R.from_matrix(relative).as_quat()            # x, y, z, w
    roll = 2.0 * np.arctan2(quat[:, 2], quat[:, 3])
    roll = (roll + np.pi) % (2 * np.pi) - np.pi
    return tilt, roll


def view_filter(cameras, reference, center, board, max_excursion, limits):
    """Mask of camera poses (N, 4, 4) inside the shared geometric limits (framing excluded)."""
    cameras = np.asarray(cameras)
    normal = board[:3, 2] if board[2, 2] >= 0 else -board[:3, 2]
    offset = cameras[:, :3, 3] - center
    distance = np.linalg.norm(offset, axis=1)
    reference_distance = float(np.linalg.norm(reference[:3, 3] - center))
    tilt, roll = tilt_and_roll(reference, cameras)
    return ((np.linalg.norm(cameras[:, :3, 3] - reference[:3, 3], axis=1) <= max_excursion)
            & (distance >= DISTANCE_SCALE[0] * reference_distance)
            & (distance <= DISTANCE_SCALE[1] * reference_distance)
            & ((offset / distance[:, None]) @ normal >= np.sin(np.deg2rad(MIN_ELEVATION_DEG)))
            & (tilt <= np.deg2rad(limits.max_tilt_deg) + 1e-9)
            & (np.abs(roll) <= np.deg2rad(limits.max_roll_deg) + 1e-9))


def orbit_camera_poses(reference, center, rng, count, limits):
    """Camera poses (N, 4, 4): the reference camera turned rigidly about the board
    centre (tilt <= max_tilt, area-uniform), moved along the line of sight and
    rolled about its optical axis (<= max_roll). The board stays where it was in the
    image, so framing mostly holds."""
    reference_distance = float(np.linalg.norm(reference[:3, 3] - center))
    direction = (reference[:3, 3] - center) / reference_distance
    helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(direction, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    tilt = np.arccos(rng.uniform(np.cos(np.deg2rad(limits.max_tilt_deg)), 1.0, count))
    azimuth = rng.uniform(0.0, 2 * np.pi, count)
    axes = np.cos(azimuth)[:, None] * e1 + np.sin(azimuth)[:, None] * e2
    turn = R.from_rotvec(axes * tilt[:, None])
    roll = R.from_rotvec(np.outer(rng.uniform(-1, 1, count) * np.deg2rad(limits.max_roll_deg), [0, 0, 1]))
    scale = rng.uniform(*DISTANCE_SCALE, count)
    poses = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    poses[:, :3, :3] = (turn * R.from_matrix(reference[:3, :3]) * roll).as_matrix()
    poses[:, :3, 3] = center + scale[:, None] * turn.apply(reference[:3, 3] - center)
    return poses


def translation_gain(rotations, candidate):
    """Observability added by one view: sum of (R_A - I)^T (R_A - I) over its pairs."""
    gram = np.zeros((3, 3))
    for rotation in rotations:
        m = rotation.T @ candidate - np.eye(3)
        gram += m.T @ m
    return gram


def weakest_observability(gram):
    """Smallest singular value of the stacked (R_A - I) of AX=XB translation."""
    return float(np.sqrt(max(np.linalg.eigvalsh(gram)[0], 0.0)))


def select_view(pool, rotations, initial=None, tolerance=0.0):
    """Index of the pool entry that raises the weakest observability; with
    ``tolerance`` > 0 the smallest joint move from ``initial`` among entries
    within that fraction of the best score."""
    if not pool:
        return None
    gram = np.zeros((3, 3))
    for i, a in enumerate(rotations):
        for b in rotations[i + 1:]:
            m = a.T @ b - np.eye(3)
            gram += m.T @ m
    scores = np.array([weakest_observability(gram + translation_gain(rotations, effector[:3, :3]))
                       for _, effector in pool])
    best = int(np.argmax(scores))
    if tolerance <= 0 or initial is None:
        return best
    eligible = np.flatnonzero(scores >= (1.0 - tolerance) * scores[best])
    move = [np.max(np.abs(np.asarray(pool[i][0]) - initial)) for i in eligible]
    return int(eligible[int(np.argmin(move))])


def shortest_route(start, entries):
    """``entries`` ([(joints, effector_pose)]) ordered as an open path from ``start``.

    Nearest neighbour, then 2-opt, on the largest joint change (the joint that
    sets the move duration). Views are picked for information first, so the
    order changes only travel, not which views are measured."""
    if len(entries) < 2:
        return list(entries)
    points = [np.asarray(start, float)] + [np.asarray(q, float) for q, _ in entries]
    cost = lambda a, b: float(np.max(np.abs(points[a] - points[b])))
    route, left = [0], set(range(1, len(points)))
    while left:
        nearest = min(left, key=lambda j: cost(route[-1], j))
        route.append(nearest)
        left.remove(nearest)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for k in range(i + 1, len(route)):
                after = route[k + 1] if k + 1 < len(route) else None
                old = cost(route[i - 1], route[i]) + (cost(route[k], after) if after is not None else 0.)
                new = cost(route[i - 1], route[k]) + (cost(route[i], after) if after is not None else 0.)
                if new < old - 1e-9:
                    route[i:k + 1] = route[i:k + 1][::-1]
                    improved = True
    return [entries[j - 1] for j in route[1:]]


class AnalyticIKViews:
    """View pool for a six-joint camera chain: orbit samples, every analytic branch
    (``arm_ik``), the winding and branch nearest the reference inside the joint window."""

    def __init__(self, solver, lower, upper, limits=ViewLimits(), margin=JOINT_LIMIT_MARGIN_RAD):
        self.solver = solver
        self.kind = solver.kind
        self.info = dict(getattr(solver, 'info', {}) or {})
        self.label = f'analytic IK ({self.kind})'
        self.limits = limits
        self.lower = np.asarray(lower, float) + margin
        self.upper = np.asarray(upper, float) - margin

    @classmethod
    def build(cls, kind, urdf, base, effector, names, lower, upper, effector_fk, limits=ViewLimits()):
        """The pool for ``kind`` on base -> effector; raises ViewPoolUnavailable saying why
        (no URDF, arm_ik missing, unsupported chain, native backend missing, joint order
        or FK disagreeing with the runner's ``effector_fk``)."""
        kind = normalize_ik_backend(kind)
        if kind == 'service':
            raise ViewPoolUnavailable("ik_backend 'service' has no analytic view pool")
        if not urdf:
            raise ViewPoolUnavailable('no robot_description')
        try:
            from arm_ik import NativeUnavailable, UnsupportedChain, make_solver
        except ImportError as exc:
            raise ViewPoolUnavailable(f'arm_ik is not installed ({exc})') from exc
        try:
            solver = make_solver(kind, urdf, base, effector)
        except NativeUnavailable as exc:
            raise ViewPoolUnavailable(f'{kind} native backend unavailable: {exc}') from exc
        except UnsupportedChain as exc:
            raise ViewPoolUnavailable(f'{kind} does not support {base} -> {effector}: {exc.reason}') from exc
        except (ValueError, ImportError) as exc:
            raise ViewPoolUnavailable(f'{kind} solver for {base} -> {effector} failed: {exc}') from exc
        if list(solver.joint_names) != list(names):
            raise ViewPoolUnavailable(f'{kind} chain joints {list(solver.joint_names)} differ from the robot joints '
                                      f'{list(names)}')
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        span = np.minimum(upper, np.pi) - np.maximum(lower, -np.pi)
        probes = [(lower + upper) / 2] + [np.maximum(lower, -np.pi) + f * span for f in (.27, .71)]
        for probe in probes:
            if not np.allclose(solver.fk(probe), effector_fk(probe), atol=FK_CHECK_TOLERANCE):
                raise ViewPoolUnavailable(f'{kind} chain FK disagrees with the effector FK at q={np.round(probe, 3)}')
        return cls(solver, lower, upper, limits)

    @classmethod
    def create(cls, urdf, base, effector, names, lower, upper, effector_fk, limits=ViewLimits(),
               kind=DEFAULT_IK_BACKEND):
        """``build``, or None when the pool is unavailable."""
        try:
            return cls.build(kind, urdf, base, effector, names, lower, upper, effector_fk, limits)
        except ViewPoolUnavailable:
            return None

    def describe(self):
        """One log line: solver kind and its info (family, native backend, model tolerance)."""
        info = ', '.join(f'{k}={v}' for k, v in self.info.items() if k not in ('rr_cache_path',))
        return (f'{self.label}: {self.solver.base} -> {self.solver.tip}, model tolerance '
                f'{self.solver.model_tolerance_m * 1000:.3g} mm ({info})')

    def solve(self, effectors, initial, lo, hi):
        """Per effector pose (N, 4, 4): the in-window solution nearest ``initial``
        (smallest largest joint change, winding nearest the reference) or None."""
        from arm_ik import windings
        solver = self.solver
        out = []
        for Q in solver.ik_many(effectors):
            if not len(Q):
                out.append(None)
                continue
            W = windings.nearest_winding(Q, initial, solver.lower, solver.upper)
            W = W[np.isfinite(W).all(axis=1)]
            W = W[np.all(W >= lo, axis=1) & np.all(W <= hi, axis=1)]
            out.append(W[int(np.argmin(np.max(np.abs(W - initial), axis=1)))] if len(W) else None)
        return out

    def pool(self, board, center, estimate, framing, reference, max_excursion, initial, rng, wanted,
             check=lambda: None):
        """[(joints, effector_pose)]: framed (13 % margin) camera poses inside the
        shared limits, reachable inside the joint window (nearest branch). The pose is
        the chain FK of the joints (within the solver's model tolerance of the target)."""
        initial = np.asarray(initial, float)
        lo, hi = joint_window_bounds(initial, self.lower, self.upper, self.limits)
        if np.any(lo >= hi):   # reference outside the limit margin
            return [], 0
        mount = np.linalg.inv(estimate)
        # Analytic IK needs an exact rotation (a rounded mount made every ssik target unsolvable).
        mount[:3, :3] = R.from_matrix(mount[:3, :3]).as_matrix()
        found, sampled = [], 0
        while len(found) < wanted and sampled < MAX_CANDIDATES:
            cameras = orbit_camera_poses(reference, center, rng, BATCH, self.limits)
            keep = view_filter(cameras, reference, center, board, max_excursion, self.limits)
            framed = [camera for camera in cameras[keep]
                      if framing.contains(np.linalg.inv(camera) @ board, margin=.13)]
            if framed:
                solutions = self.solve(np.array(framed) @ mount, initial, lo, hi)
                Q = np.array([q for q in solutions if q is not None])
                if len(Q):
                    rotations, positions = self.solver.fk_many(Q)
                    for q, rotation, position in zip(Q, rotations, positions):
                        effector = np.eye(4)
                        effector[:3, :3], effector[:3, 3] = rotation, position
                        found.append((q, effector))
            sampled += BATCH
            check()
        return found, sampled
