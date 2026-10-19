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
``ClosedFormViews.create`` returns None and the caller keeps its sampler.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

JOINT_LIMIT_MARGIN_RAD = 0.10
DISTANCE_RANGE_M = (0.28, 0.75)
ELEVATION_RANGE_DEG = (40.0, 90.0)   # camera direction above the board plane
ROLL_RANGE_DEG = 90.0                # about the optical axis
AIM_JITTER_M = 0.03
BATCH = 5000
MAX_CANDIDATES = 60000
POOL_PER_VIEW = 3                    # feasible candidates wanted per requested view


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


def shell_camera_poses(board, center, rng, count):
    """Camera poses (base frame) around ``center``, looking at it from above the board."""
    normal = board[:3, 2] if board[2, 2] >= 0 else -board[:3, 2]
    e1 = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(normal, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    distance = rng.uniform(*DISTANCE_RANGE_M, count)
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
        while len(found) < wanted and sampled < MAX_CANDIDATES:
            for camera in shell_camera_poses(board, center, rng, BATCH):
                if np.linalg.norm(camera[:3, 3] - reference[:3, 3]) > max_excursion:
                    continue
                if not framing.contains(np.linalg.inv(camera) @ board, margin=.13):
                    continue
                effector = camera @ mount
                solutions = [q for q in self.ik.ik(effector)
                             if np.all(q >= self.lower) and np.all(q <= self.upper)]
                if solutions:
                    q = min(solutions, key=lambda q: np.max(np.abs(q - initial)))
                    found.append((q, effector))
            sampled += BATCH
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
