"""Eye-in-hand acquisition without an initial robot-to-camera transform.

Only robot FK, measured image geometry and checked joint increments are used
until six observations provide an internally consistent initial estimate.
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .automatic_geometry import has_rotation_diversity, validation_error
from .visibility import BoardFraming, CameraKinematics
from .image_motion import ImageMotionModel
from .calibration_quality import position_uncertainty_ok
from .acquisition_progress import AcquisitionProgress
from .informative_views import InformativeViews, informative_view


def sample_pose(sample):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(sample[3:]).as_matrix()
    pose[:3, 3] = sample[:3]
    return pose


class BootstrapSession:
    STEP = .02  # radians per stopped-and-observed increment, ~1.15 degrees
    MAX_STEP = .05
    EXCURSION = .35
    MAX_ATTEMPTS = 180
    MAX_VIEWS = 15
    INITIAL_SAMPLES = 6
    TRAINING = 15

    def __init__(self, runner, root):
        self.r = runner
        self.n = runner.node
        info = runner.camera_info
        if info is None or info.header.frame_id != self.n.tracking_base_frame:
            raise ValueError('Bootstrap requires the detector and CameraInfo in the same optical frame; no camera mount TF is used.')
        runner.framing = BoardFraming(runner.board_spec, info)
        runner.optical_tracking = np.eye(4)
        # The chain ends at the ROBOT effector. Camera mount joints are never read.
        self.fk = CameraKinematics(root, self.n.robot_base_frame,
                                  self.n.robot_effector_frame, runner.names, np.eye(4))
        self.estimate = None
        self.views = []
        self.attempts = 0
        self.reason = ''
        self.image_motion = ImageMotionModel(len(runner.names))
        self.allowed_step = self.STEP
        self.sigma_limit = getattr(runner, 'max_position_sigma', .002)
        self.max_training = self.TRAINING
        self.uncertainty = None
        self.view_planner = None
        self.visited_joints = []

    def observe(self):
        return self.r.observed_board(recovering=True)

    def monitor(self, retreat):
        from .automatic_calibration import FramingCorrection
        try:
            self.r.motion_monitor()
        except FramingCorrection:
            # A recorded return segment may start in the warning band. The hard
            # image boundary, timestamps and joint feedback remain mandatory.
            if not retreat:
                raise

    def step(self, target, retreat=False, planned_transit=False):
        from .automatic_calibration import FramingCorrection, TrackingInterrupted
        start = self.r.joint_positions()
        if np.max(np.abs(target-start)) > self.allowed_step + .01:
            raise RuntimeError('Bootstrap increment exceeds the joint step limit.')
        if np.any(target < self.r.lower) or np.any(target > self.r.upper):
            self.reason = 'joint limit'
            return False
        if not self.r.collision_free_path(start, target):
            self.reason = 'collision in MoveIt scene'
            return False
        if planned_transit:
            if self.estimate is None:
                raise RuntimeError('Planned transit requires an initial camera estimate.')
            # The path is collision-checked; only joint feedback is required in
            # transit. Board detection is required again at the sample endpoint.
            self.r.move(target, monitor=self.r.joint_positions, speed_scale=1., minimum_duration=.5)
            return True
        board = self.observe()
        if not retreat and not self.r.framing.contains(board):
            self.reason = 'measured image margin'
            return False
        if self.estimate is not None and not retreat:
            # This approximate estimate comes only from this run; live tracking remains mandatory.
            fixed_board = self.fk.camera(start) @ self.estimate @ board
            for fraction in np.linspace(0, 1, 5):
                camera = self.fk.camera(start + fraction*(target-start)) @ self.estimate
                if not self.r.framing.contains(np.linalg.inv(camera) @ fixed_board):
                    self.reason = 'predicted image margin using current estimate'
                    return False
        for retry in range(3):
            self.r.check()
            self.r.monitor_stamp, self.r.monitor_at = None, time.monotonic()
            try:
                self.r.move(target, monitor=lambda: self.monitor(retreat), speed_scale=(.5 if retreat else 1.)/(retry+1), minimum_duration=.5)
                board = self.observe()
                self.reason = 'measured image margin'
                return retreat or self.r.framing.contains(board)
            except FramingCorrection:
                self.reason = 'measured image warning margin'
                return False  # controller has cancelled and settled
            except TrackingInterrupted:
                self.r.update('paused', 'Stopped; waiting for stable board detection before a small joint step.')
                self.r.wait_for_tracking()
                if not retreat:
                    self.reason = 'tracking interruption; trying another joint direction'
                    return False
        raise RuntimeError('Board tracking did not stabilize during the recorded return path; robot remains stopped.')

    def retrace(self, route):
        # Retrace only measured, previously traversed joint positions. No camera
        # TF, inverse kinematics or blind image-based correction is involved.
        for waypoint in reversed(route):
            for _ in range(30):
                current = self.r.joint_positions()
                delta = waypoint-current
                if np.max(np.abs(delta)) < .003:
                    break
                target = current + delta * min(1., self.STEP/np.max(np.abs(delta)))
                if not self.step(target, retreat=True):
                    raise RuntimeError('Recorded return path rejected: '+self.reason+'. Robot remains stopped.')
            else:
                raise RuntimeError('Robot did not reach a recorded return waypoint.')

    def distinct(self, pose):
        if self.estimate is None:
            # Translation alone cannot establish the initial hand-eye rotation.
            return all(Rotation.from_matrix(old[:3,:3].T@pose[:3,:3]).magnitude()
                       >= np.deg2rad(2.5) for old in self.views)
        return informative_view(pose, self.views)

    def update_estimate(self):
        robots = [sample_pose(s) for s in self.n.robot_samples]
        if not has_rotation_diversity(robots):
            raise RuntimeError('Initial six samples need stronger rotation about two axes; no new calibration was saved.')
        cal = self.n.get_calibration()
        if cal is None:
            raise RuntimeError('Camera transform could not be estimated from collected samples.')
        estimate = sample_pose(cal)
        tracking = [sample_pose(s) for s in self.n.tracking_samples]
        # Internal consistency only: every observation remains in the solve.
        fit = validation_error(robots, tracking, estimate, list(zip(robots, tracking)))
        if (not np.isfinite(estimate).all() or not np.isfinite(list(fit.values())).all()
                or fit['max_translation_m'] > .01 or fit['max_rotation_deg'] > 3):
            raise RuntimeError('Camera estimate is inconsistent across the collected samples (limit 10 mm / 3 degrees).')
        self.estimate = estimate
        self.view_planner = None  # re-plan remaining views using each new measurement
        self.r.update('preparing', 'Camera estimate updated; selecting the next targeted sample.',
                      phase='refinement', fit_consistency=fit, validation=None,
                      initial_samples=min(len(self.views), self.INITIAL_SAMPLES),
                      targeted_samples=max(0, len(self.views)-self.INITIAL_SAMPLES))

    def capture(self):
        if len(self.n.robot_samples) >= self.max_training:
            return
        robot = self.fk.camera(self.r.joint_positions())
        if not self.distinct(robot):
            return
        # Reserve the sixth initial observation for sufficient multi-axis motion.
        if len(self.views) == self.INITIAL_SAMPLES-1 and not has_rotation_diversity(self.views+[robot]):
            return
        if not self.r.capture():
            return
        measured = sample_pose(self.n.robot_samples[-1])
        if (not self.distinct(measured) or
                (len(self.views) == self.INITIAL_SAMPLES-1 and not has_rotation_diversity(self.views+[measured]))):
            self.n.robot_samples.pop(); self.n.tracking_samples.pop(); self.n.sample_metrics.pop()
            self.n._publish_status(None, None)
            return
        self.views.append(measured)
        self.r.update('capturing', 'Accepted a distinct calibration sample.', accepted=len(self.views),
                      pose=len(self.views), total=self.TRAINING, target_samples=self.TRAINING,
                      phase='bootstrap' if self.estimate is None else 'refinement',
                      initial_samples=min(len(self.views), self.INITIAL_SAMPLES),
                      targeted_samples=max(0, len(self.views)-self.INITIAL_SAMPLES),
                      required_validation_views=0, validation_views=0, validation=None)
        if len(self.views) >= self.INITIAL_SAMPLES:
            self.update_estimate()

    def quality_ready(self):
        if len(self.n.robot_samples) < self.TRAINING:
            return False
        self.r.check()
        self.r.update('assessing', 'Robot stationary; assessing uncertainty from all 15 samples.',
                      position_sigma_limit_m=self.sigma_limit, max_training_samples=self.max_training)
        cal = self.n.get_calibration()
        self.uncertainty = self.n._compute_uncertainty(cal) if cal is not None else None
        self.r.check()
        self.n._last_uncertainty = self.uncertainty
        self.n._publish_status(cal, None)
        sigma = self.uncertainty.get('worst_direction_sigma_m') if self.uncertainty else None
        if sigma is not None and not np.isfinite(sigma):
            sigma = None
        self.r.update('assessing', 'Checking the uncertainty target; no independent validation views collected.',
                      position_sigma_m=sigma, position_sigma_limit_m=self.sigma_limit)
        if position_uncertainty_ok(self.uncertainty, self.sigma_limit):
            return True
        detail = 'unavailable' if sigma is None else f'{sigma*1000:.2f} mm'
        raise RuntimeError(f'Position uncertainty {detail}; target {self.sigma_limit*1000:.2f} mm. '
                           'Sample limit reached (15); no new calibration was saved.')

    def candidates(self, current, initial, pixels, blocked):
        candidates = []
        correction = self.image_motion.centered_delta(current, pixels)
        for axis in reversed(range(len(current))):
            for sign in (1., -1.):
                probe = np.zeros(len(current)); probe[axis] = sign*.01
                known = self.image_motion.predict(current, probe, pixels) is not None
                limit = (self.MAX_STEP if self.image_motion.confident(current, probe) else self.STEP) if known else .01
                delta = np.zeros(len(current)); delta[axis] = sign*limit
                variants = [delta]
                if known:
                    centered = delta + .35*correction
                    centered *= min(1., limit/max(np.max(np.abs(centered)), 1e-9))
                    variants.append(centered)
                for delta in variants:
                    target = current+delta
                    if (np.any(target < self.r.lower) or np.any(target > self.r.upper)
                            or np.max(np.abs(target-initial)) > self.EXCURSION
                            or tuple(np.round(target,4)) in blocked):
                        continue
                    predicted = self.image_motion.predict(current, delta, pixels)
                    if predicted is not None and (np.min(predicted) < .12 or np.max(predicted) > .88):
                        continue
                    pose = self.fk.camera(target)
                    # Favor novel poses and rotation about axes not yet well sampled.
                    rotations = [Rotation.from_matrix(v[:3,:3].T@pose[:3,:3]).magnitude() for v in self.views]
                    translations = [np.linalg.norm(v[:3,3]-pose[:3,3]) for v in self.views]
                    novelty = min([max(a/.044, t/.01) for a,t in zip(rotations,translations)], default=1.)
                    if self.estimate is None:
                        novelty = min(rotations, default=.044)/.044
                    vectors = [Rotation.from_matrix(self.views[0][:3,:3].T@v[:3,:3]).as_rotvec()
                               for v in self.views] if self.views else []
                    vector = Rotation.from_matrix(self.views[0][:3,:3].T@pose[:3,:3]).as_rotvec() if self.views else np.zeros(3)
                    gram = np.eye(3)*.005 + sum((np.outer(v,v) for v in vectors), np.zeros((3,3)))
                    gain = float(np.log1p(vector@np.linalg.solve(gram,vector)))
                    margin = float(min(np.min(predicted), 1-np.max(predicted))) if predicted is not None else .12
                    weak_gain = 0.
                    if self.uncertainty is not None and self.estimate is not None:
                        weak_axis = np.asarray(self.uncertainty.get('worst_direction_axis', []), float)
                        if weak_axis.shape == (3,) and np.isfinite(weak_axis).all() and np.linalg.norm(weak_axis) > 0:
                            weak_axis /= np.linalg.norm(weak_axis)
                            # Hand-eye translation observability: (R_relative-I)t.
                            # Favor rotations that constrain the weakest translation direction.
                            weak_gain = sum(np.linalg.norm((v[:3,:3].T@pose[:3,:3]-np.eye(3))@weak_axis)**2
                                            for v in self.views) / max(1, len(self.views))
                    axis_gain = 0.
                    if self.estimate is None and self.views:
                        current_vector = Rotation.from_matrix(self.views[0][:3,:3].T@self.fk.camera(current)[:3,:3]).as_rotvec()
                        direction = vector-current_vector
                        if np.linalg.norm(direction)>1e-6:
                            direction /= np.linalg.norm(direction)
                            # Prefer excitation of an undersampled rotation axis,
                            # not just a larger angle along the already dominant one.
                            axis_gain = .5*np.sqrt(direction@np.linalg.solve(gram,direction))
                    visits = sum(np.max(np.abs(target-old)) < .008 for old in self.visited_joints)
                    score = novelty + axis_gain + 2*gain + 10*weak_gain + margin + (.25 if not known else 0.) - .5*visits
                    candidates.append((score, target, limit))
        return sorted(candidates, key=lambda c:c[0], reverse=True)

    def run(self):
        initial = self.r.joint_positions()
        board = self.observe()
        if not self.r.framing.contains(board):
            raise ValueError('Center the full board with 10% image margin before bootstrap. No motion was sent.')
        self.n.robot_samples.clear(); self.n.tracking_samples.clear(); self.n.sample_metrics.clear()
        self.n._last_uncertainty = self.n._last_calibration_detail = None
        self.n._publish_status(None, None)
        self.r.update('capturing', 'Collecting the first observation without a camera mount transform.', phase='bootstrap',
                      pose=0, total=self.MAX_VIEWS)
        self.capture()
        last_good = initial.copy()
        self.visited_joints = [initial.copy()]
        progress = AcquisitionProgress(len(self.views))
        blocked = set()
        for attempt in range(self.MAX_ATTEMPTS):
            self.r.check()
            if len(self.views) >= self.MAX_VIEWS:
                raise RuntimeError('Calibration sample limit reached (15); no new calibration was saved.')
            progress.observe(len(self.views))
            current = self.r.joint_positions()
            if self.estimate is not None and self.view_planner is not None:
                # Propagate the fixed board estimate while it may be out of view.
                board = np.linalg.inv(self.view_planner.camera(current)) @ self.view_planner.board
            else:
                board = self.observe()
            pixels = self.r.framing.pixels(board) / [self.r.framing.width, self.r.framing.height]
            target = None
            planned = False
            if self.estimate is not None:
                if self.view_planner is None:
                    self.r.update('planning', 'Selecting diverse camera views around the observed board.')
                    self.view_planner = InformativeViews(self, initial, current, board)
                target = self.view_planner.next_step(current, board)
                planned = target is not None
                if not planned and self.view_planner.pending_candidates:
                    self.r.update('planning', 'Robot stationary; checking more candidate paths with MoveIt.',
                                  planner_rejections=self.view_planner.rejections)
                    continue
            if self.estimate is not None and not planned:
                raise RuntimeError('No reachable diverse target with a visible board remains; no new calibration was saved.')
            if planned:
                limit = self.MAX_STEP
            else:
                candidates = self.candidates(current, initial, pixels, blocked)
                if not candidates:
                    raise RuntimeError('No feasible local exploration step remains. Robot stopped; no new calibration was saved. Reposition the board to allow different viewing angles.')
                _, target, limit = candidates[0]
            progress.before_step(planned, advancing=planned and self.view_planner.advancing)
            self.allowed_step = limit
            self.r.update('moving', ('Moving along the planned trajectory to the next calibration view.' if planned else
                                     f'Local exploration: {progress.local_without_sample}/{progress.MAX_LOCAL_WITHOUT_SAMPLE} attempts without a new view.'),
                          attempts=self.attempts+1, step_degrees=float(np.degrees(limit)),
                          search_mode='targeted' if planned else 'local',
                          planner_rejections=getattr(self.view_planner,'rejections',{}), **progress.status())
            if planned:
                self.r.move_planned(self.view_planner.trajectory)
                accepted = True
            else:
                accepted = self.step(target)
            actual = self.r.joint_positions()
            if not planned:
                after = self.observe()
                after_pixels = self.r.framing.pixels(after) / [self.r.framing.width, self.r.framing.height]
                self.image_motion.add(current, actual, pixels, after_pixels)
            self.attempts += 1
            if np.max(np.abs(actual-current)) < .003:
                blocked.add(tuple(np.round(target, 4)))
                if planned:
                    self.view_planner.reject_goal()
                continue
            self.visited_joints.append(actual.copy())
            if not accepted:
                if planned:
                    self.view_planner.reject_goal()
                    continue
                # Recover only the last bounded segment, never an entire excursion.
                self.allowed_step = self.STEP
                self.r.update('paused', 'View lost its margin; returning only the last observed step.')
                self.retrace([last_good])
                self.image_motion.good_predictions = 0
                continue
            last_good = actual.copy()
            # Capture only when the measured view passes the stronger novelty
            # gate; small travel increments therefore do not fill the dataset.
            # Transit increments never consume the nine targeted samples.
            if not planned or np.max(np.abs(actual-self.view_planner.goal)) < .003:
                if planned:
                    self.r.update('capturing', 'At target; waiting for stable board detection before capturing.')
                    self.r.wait_for_tracking()
                    observed = self.r.observed_board(recovering=False)
                    self.view_planner.board = self.view_planner.camera(actual) @ observed
                self.capture()
            progress.observe(len(self.views))
            self.r.update(self.r.status['state'], self.r.status['message'], **progress.status())
            if self.quality_ready():
                return
        raise RuntimeError('Calibration did not finish within the movement attempt limit; no new calibration was saved. '
                           f'Collected {len(self.n.robot_samples)}/15 samples.')
