"""Main/validation views for a camera before the last joint (PiPER link5).

Five joints move such a camera, so a sampled 6-D camera pose is almost never
reachable: the local sampler took the nearest pose found by least squares,
which kept pushing joint5 onto its limit (run 20261014_090523: 11/20 views at
j5 >= 1.215 rad, return path refused at 1.22028 > 1.22). Here the joints are
sampled instead and the camera pose follows from FK, so every candidate is
reachable by construction and stays a margin inside the limits. Candidates are
filtered with the same limits as the six-joint pool (``view_pool``: joint window,
tilt/roll from the reference view, distance, view angle, excursion, framing) and
picked with the same selection (``view_pool.select_view``).

Kinematics only: MoveIt still plans and collision-checks each chosen view.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from .motion_planning import SAMPLER_JOINT_MARGIN_RAD
from .view_pool import ViewLimits, joint_window_bounds, view_filter

LOCAL_SPREAD_RAD = 0.35     # std of joint samples around the reference pose
UNIFORM_FRACTION = 0.2      # rest of each batch: uniform within the joint window
BATCH = 5000
MAX_CANDIDATES = 60000


class JointSpaceViews:
    """Feasible view pool from sampled joints of the camera chain plus greedy selection."""

    label = 'joint-space FK'

    def __init__(self, fk, names, lower, upper, limits=ViewLimits(), margin=SAMPLER_JOINT_MARGIN_RAD):
        self.fk = fk
        self.limits = limits
        self.lower = np.asarray(lower, float) + margin
        self.upper = np.asarray(upper, float) - margin
        moving = {name for _, _, kind, name in fk.chain if kind != 'fixed'}
        # Joints after the camera (PiPER joint6) do not move it; they keep the reference value.
        self.active = np.array([name in moving for name in names])
        index = {name: i for i, name in enumerate(names)}
        self.chain = [(origin, axis, kind, index.get(name)) for origin, axis, kind, name in reversed(fk.chain)]

    def effectors(self, q):
        """Batched ``fk.camera``: (N, joints) -> (N, 4, 4)."""
        result = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        for origin, axis, kind, i in self.chain:
            result = result @ origin
            if kind in ('revolute', 'continuous'):
                motion = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
                motion[:, :3, :3] = R.from_rotvec(q[:, i, None] * axis).as_matrix()
                result = result @ motion
            elif kind == 'prismatic':
                result[:, :3, 3] += result[:, :3, :3] @ axis * q[:, i, None]
        return result @ self.fk.ik_camera

    def samples(self, initial, rng, count, lower, upper):
        """Joint samples inside [lower, upper] (the window): mostly normal around
        ``initial``, a fraction uniform; inactive joints keep their reference value."""
        q = np.repeat(np.asarray(initial, float)[None], count, axis=0)
        uniform = rng.random(count) < UNIFORM_FRACTION
        local = initial + rng.normal(0.0, LOCAL_SPREAD_RAD, q.shape)
        spread = rng.uniform(lower, upper, q.shape)
        q[:, self.active] = np.where(uniform[:, None], spread, local)[:, self.active]
        inside = np.all((q >= lower) | ~self.active, axis=1) & np.all((q <= upper) | ~self.active, axis=1)
        return q[inside]

    def pool(self, board, center, estimate, framing, reference, max_excursion, initial, rng, wanted,
             check=lambda: None):
        """[(joints, effector_pose)] for framed camera poses (13 % margin) inside the
        shared view limits; joints inside the window and the limit margin."""
        initial = np.asarray(initial, float)
        lower, upper = joint_window_bounds(initial, self.lower, self.upper, self.limits, self.active)
        if np.any((lower >= upper) & self.active):   # reference outside the limit margin
            return [], 0
        found, sampled = [], 0
        while len(found) < wanted and sampled < MAX_CANDIDATES:
            q = self.samples(initial, rng, BATCH, lower, upper)
            effector = self.effectors(q)
            camera = effector @ estimate
            keep = view_filter(camera, reference, center, board, max_excursion, self.limits)
            for i in np.flatnonzero(keep):
                if framing.contains(np.linalg.inv(camera[i]) @ board, margin=.13):
                    found.append((q[i], effector[i]))
            sampled += BATCH
            check()
        return found, sampled
