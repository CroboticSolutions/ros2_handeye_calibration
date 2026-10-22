"""Main/validation views for intrinsic acquisition from closed-form IK.

The local random sampler (``random_camera_pose``: +-6 cm, tilt +-15 deg, roll
+-35 deg around one reference) gave relative flange rotations of only 13.7 deg
(median) on the D435 run 20261013_095104. The camera translation along the
flange axis was then barely observable: AX=XB solvers put it at 22 mm, the
board-spread optimum at 48 mm (2026-10-13). Here camera poses are sampled on a
wide shell around the board, every closed-form IK branch is checked against the
joint limits with a margin, and views are picked greedily to raise the weakest
direction of the hand-eye translation observability. Offline on the same scene:
weakest singular value 3.41 -> 6.37 for 26 views, median relative rotation
13.7 -> 24.7 deg, every selected joint >= 0.10 rad from its limit.

Kinematics only: MoveIt still plans and collision-checks each chosen view.
Needs a 6R ortho-parallel chain to the effector (PiPER link6); otherwise
``ClosedFormViews.create`` returns None. ``NumericIKViews`` then builds the same
pool with a damped least-squares IK on the URDF chain (CRX: y/x joint axes have
no closed form here). Without it the local sampler mostly rolled the CRX camera
about its optical axis = J6 (21.10.: rotation RMS about flange x/y/z 15/4/10 deg,
camera translation along J6 the weakest direction in every run).
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

JOINT_LIMIT_MARGIN_RAD = 0.10
DISTANCE_RANGE_M = (0.28, 0.75)
ELEVATION_RANGE_DEG = (40.0, 90.0)   # camera direction above the board plane
ROLL_RANGE_DEG = 90.0                # about the optical axis
AIM_JITTER_M = 0.03
BATCH = 5000
MAX_CANDIDATES = 60000
POOL_PER_VIEW = 3                    # feasible candidates wanted per requested view
# Numeric pool: shell radius relative to the reference view (a large arm starts
# farther than the PiPER range above), smaller batches as each IK costs ~ms.
NUMERIC_DISTANCE_SCALE = (0.8, 1.15)
NUMERIC_BATCH = 250
NUMERIC_MAX_CANDIDATES = 5000
NUMERIC_IK_TOLERANCE = (0.001, np.deg2rad(0.5))   # position m, rotation rad


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


def look_at(position, aim, roll):
    z = aim - position
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    if np.linalg.norm(x) < 1e-6:   # looking straight down base z
        x = np.cross(z, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, np.cross(z, x), z]) @ R.from_euler('z', roll).as_matrix()
    pose[:3, 3] = position
    return pose


def shell_camera_poses(board, center, rng, count, distance_range=DISTANCE_RANGE_M):
    """Camera poses (base frame) around ``center``, looking at it from above the board."""
    normal = board[:3, 2] if board[2, 2] >= 0 else -board[:3, 2]
    e1 = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(normal, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    distance = rng.uniform(*distance_range, count)
    elevation = np.deg2rad(rng.uniform(*ELEVATION_RANGE_DEG, count))
    azimuth = rng.uniform(0.0, 2 * np.pi, count)
    roll = np.deg2rad(rng.uniform(-ROLL_RANGE_DEG, ROLL_RANGE_DEG, count))
    for d, el, az, rl in zip(distance, elevation, azimuth, roll):
        direction = np.sin(el) * normal + np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2)
        aim = center + rng.uniform(-AIM_JITTER_M, AIM_JITTER_M, 3)
        yield look_at(center + d * direction, aim, rl)


class ClosedFormViews:
    """Feasible view pool for one acquisition phase plus greedy selection."""

    label = 'closed-form IK'
    batch = BATCH
    max_candidates = MAX_CANDIDATES

    def distance_range(self, reference, center):
        return DISTANCE_RANGE_M

    def solutions(self, effector, initial):
        return self.ik.ik(effector)

    def __init__(self, ik, lower, upper, margin=JOINT_LIMIT_MARGIN_RAD):
        self.ik = ik
        self.lower = np.asarray(lower, float) + margin
        self.upper = np.asarray(upper, float) - margin

    @classmethod
    def create(cls, urdf, base, effector, names, lower, upper, effector_fk):
        """Planner, or None when the chain has no closed form or disagrees with ``effector_fk``."""
        if not urdf:
            return None
        try:
            from hand_teleop.closed_form_ik import ClosedFormIK
        except ImportError:
            return None
        ik = ClosedFormIK.from_urdf(urdf, base, effector)
        if ik is None or list(ik.joint_names) != list(names):
            return None
        probe = (np.asarray(lower, float) + np.asarray(upper, float)) / 2
        if not np.allclose(ik.fk(probe), effector_fk(probe), atol=1e-6):
            return None
        return cls(ik, lower, upper)

    def pool(self, board, center, estimate, framing, reference, max_excursion, initial, rng, wanted,
             check=lambda: None):
        """[(joints, effector_pose)] for camera poses that are framed (13 % margin),
        within ``max_excursion`` of the reference camera and reachable with the
        joint-limit margin; the branch nearest ``initial`` (the return pose)."""
        mount = np.linalg.inv(estimate)
        found, sampled = [], 0
        distance_range = self.distance_range(reference, center)
        while len(found) < wanted and sampled < self.max_candidates:
            for camera in shell_camera_poses(board, center, rng, self.batch, distance_range):
                if np.linalg.norm(camera[:3, 3] - reference[:3, 3]) > max_excursion:
                    continue
                if not framing.contains(np.linalg.inv(camera) @ board, margin=.13):
                    continue
                effector = camera @ mount
                solutions = [q for q in self.solutions(effector, initial)
                             if np.all(q >= self.lower) and np.all(q <= self.upper)]
                if solutions:
                    q = min(solutions, key=lambda q: np.max(np.abs(q - initial)))
                    found.append((q, effector))
            sampled += self.batch
            check()
        return found, sampled

    @staticmethod
    def best(pool, rotations):
        """Index of the pool entry that most raises the weakest observability."""
        gram = np.zeros((3, 3))
        for i, a in enumerate(rotations):
            for b in rotations[i + 1:]:
                m = a.T @ b - np.eye(3)
                gram += m.T @ m
        scores = [weakest_observability(gram + translation_gain(rotations, effector[:3, :3]))
                  for _, effector in pool]
        return int(np.argmax(scores)) if scores else None


class NumericIK:
    """Damped least-squares IK on the URDF chain, seeded at ``seed`` (the return pose)."""

    def __init__(self, fk, lower, upper):
        self.fk = fk
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)

    def ik(self, pose, seed):
        def residual(q):
            actual = self.fk(q)
            return np.r_[(actual[:3, 3] - pose[:3, 3]) / NUMERIC_IK_TOLERANCE[0],
                         R.from_matrix(pose[:3, :3].T @ actual[:3, :3]).as_rotvec() / NUMERIC_IK_TOLERANCE[1]]
        start = np.clip(seed, self.lower + 1e-8, self.upper - 1e-8)
        fit = least_squares(residual, start, bounds=(self.lower, self.upper), max_nfev=60)
        error = residual(fit.x)
        return [fit.x] if np.linalg.norm(error[:3]) <= 1 and np.linalg.norm(error[3:]) <= 1 else []


class NumericIKViews(ClosedFormViews):
    """The closed-form pool and greedy selection for chains without a closed form.

    One IK branch only: the solution nearest the seed (the return pose), so a
    view never needs a wrist flip. The joint-limit margin bounds the solver.
    """

    label = 'numeric IK'
    batch = NUMERIC_BATCH
    max_candidates = NUMERIC_MAX_CANDIDATES

    def __init__(self, effector_fk, lower, upper, margin=JOINT_LIMIT_MARGIN_RAD):
        super().__init__(None, lower, upper, margin)
        self.ik = NumericIK(effector_fk, self.lower, self.upper)

    def distance_range(self, reference, center):
        distance = float(np.linalg.norm(reference[:3, 3] - center))
        return tuple(distance * scale for scale in NUMERIC_DISTANCE_SCALE)

    def solutions(self, effector, initial):
        return self.ik.ik(effector, initial)
