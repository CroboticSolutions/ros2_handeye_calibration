"""Bounded, board-centred view planning using only this run's current mount estimate."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .view_information import ViewInformation


def informative_view(pose, previous):
    """Translation alone must not fill the dataset with parallel views."""
    for old in previous:
        angle = Rotation.from_matrix(old[:3, :3].T @ pose[:3, :3]).magnitude()
        distance = np.linalg.norm(old[:3, 3] - pose[:3, 3])
        if angle < np.deg2rad(5) and not (angle >= np.deg2rad(2.5) and distance >= .03):
            return False
    return True


def translation_information(poses):
    information = np.eye(3)*1e-4
    for i, a in enumerate(poses):
        for b in poses[i+1:]:
            relative = a[:3, :3].T @ b[:3, :3] - np.eye(3)
            information += relative.T @ relative
    return information


def information_gain(pose, previous, information=None):
    if information is None:
        information = translation_information(previous)
    added = np.zeros((3, 3))
    for old in previous:
        relative = old[:3, :3].T @ pose[:3, :3] - np.eye(3)
        added += relative.T @ relative
    # Reward constraining the weakest direction as well as total information.
    before = np.linalg.eigvalsh(information)
    after = np.linalg.eigvalsh(information+added)
    return float(np.log(after/before).sum() + 2*np.log(after[0]/before[0]))


class InformativeViews:
    MAX_PLAN_REQUESTS = 6

    def __init__(self, session, initial, current, observation):
        self.s = session
        self.camera = lambda q: session.fk.camera(q) @ session.estimate
        self.board = self.camera(current) @ observation
        self.reference = self.camera(current)
        self.center = (self.board @ np.r_[session.r.framing.center, 1])[:3]
        self.depth = float((np.linalg.inv(self.reference) @ np.r_[self.center, 1])[2])
        self.lower = session.r.lower.copy()
        self.upper = session.r.upper.copy()
        self.targets = []
        self.goal = None
        self.route = []
        self.rejections = {}
        self.remaining = None
        self.advancing = False
        self.tried_targets = set()
        self.selection_view_count = -1
        self.pending_candidates = False
        # Camera rotations, not individual joint rotations. All targets look at
        # the measured board centre; several distances add depth diversity.
        offsets = []
        for angle in (6, 10, 16, 22, 30):
            offsets += [(angle,0,0),(-angle,0,0),(0,angle,0),(0,-angle,0),
                        (angle,angle,0),(-angle,-angle,0),(angle,-angle,0),(-angle,angle,0),
                        (0,0,angle),(0,0,-angle),(angle,0,angle),(-angle,0,-angle)]
        for i, angles in enumerate(offsets):
            session.r.check()
            desired = self.reference.copy()
            desired[:3,:3] = self.reference[:3,:3] @ Rotation.from_euler('xyz',angles,degrees=True).as_matrix()
            scale = (1., .92, 1.08)[i%3]
            desired[:3,3] = self.center - desired[:3,:3] @ [0.,0.,self.depth*scale]
            if not session.r.framing.contains(np.linalg.inv(desired)@self.board, margin=.13):
                continue
            def residual(q):
                session.r.check()
                actual = self.camera(q)
                return np.r_[(actual[:3,3]-desired[:3,3])/.01,
                             Rotation.from_matrix(desired[:3,:3].T@actual[:3,:3]).as_rotvec()/.05]
            fit = least_squares(residual, np.clip(current,self.lower+1e-8,self.upper-1e-8),
                                bounds=(self.lower,self.upper), max_nfev=65)
            error = residual(fit.x)
            if np.linalg.norm(error[:3]) <= .5 and np.linalg.norm(error[3:]) <= .7:
                self.targets.append(fit.x)

    def next_step(self, current, observation):
        s = self.s
        # Observations are measured at sample endpoints; during transit the
        # caller propagates the fixed board estimate without requiring detection.
        board = self.camera(current) @ observation
        if hasattr(s.r, "show_board_estimate") and hasattr(s.r, "board_marker_pub"):
            s.r.show_board_estimate(board)
        if self.goal is not None and np.max(np.abs(self.goal-current)) < .003:
            self.goal = None
            self.route = []
        if self.goal is None:
            self.remaining = None
            self.advancing = False
            if getattr(self, 'selection_view_count', -1) != len(s.views):
                self.tried_targets = set()
                self.selection_view_count = len(s.views)
            information = ViewInformation(s.views,s.estimate,board,s.r.framing)
            ranked = sorted(self.targets, key=lambda q: information.gain(s.fk.camera(q))
                            / (1.+np.max(np.abs(q-current))/.1), reverse=True)
            requests = 0
            self.rejections = {}
            for target in ranked:
                s.r.check()
                key = tuple(np.round(target,6))
                if key in self.tried_targets:
                    continue
                if not informative_view(s.fk.camera(target),s.views):
                    self.targets = [q for q in self.targets if q is not target]
                    continue
                if requests >= self.MAX_PLAN_REQUESTS:
                    break
                requests += 1
                self.tried_targets.add(key)
                path = s.r.plan_joint_path(current, target)
                if path is None:
                    self.rejections['no_path'] = self.rejections.get('no_path',0)+1
                    continue
                if not s.r.framing.contains(np.linalg.inv(self.camera(target)) @ board, margin=.12):
                    self.rejections['visibility'] = self.rejections.get('visibility',0)+1
                    continue
                self.targets = [q for q in self.targets if q is not target]
                self.trajectory = s.r.planned_trajectory
                self.goal = np.asarray(self.trajectory.points[-1].positions, dtype=float)
                self.route = [q.copy() for q in path[1:]]
                break
        self.pending_candidates = self.goal is None and any(
            tuple(np.round(q,6)) not in self.tried_targets for q in self.targets)
        if self.goal is None:
            return None
        while self.route and np.max(np.abs(self.route[0]-current)) < .003:
            self.route.pop(0)
        remaining = sum(float(np.max(np.abs(b-a))) for a,b in zip([current]+self.route,self.route))
        self.advancing = self.remaining is not None and remaining < self.remaining-.002
        self.remaining = remaining
        return self.goal.copy()

    def reject_goal(self):
        self.goal = None
        self.route = []
        self.remaining = None
        self.advancing = False
