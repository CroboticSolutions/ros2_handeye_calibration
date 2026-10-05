"""Eye-in-hand acquisition without an initial robot-to-camera transform.

Only robot FK, measured image geometry and checked joint increments are used
until diverse observations provide a stable, internally consistent initial estimate.
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .acquisition_quality import bootstrap_geometry, common_board, coverage_score
from . import acceptance
from .visibility import BoardFraming, CameraKinematics
from .image_motion import ImageMotionModel
from .calibration_quality import position_uncertainty_ok
from .acquisition_progress import AcquisitionProgress
from .informative_views import InformativeViews, informative_view


def _clear_samples(node):
    if hasattr(node, 'clear_samples'):
        node.clear_samples()
        return
    node.robot_samples.clear(); node.tracking_samples.clear(); node.sample_metrics.clear()
    node._last_uncertainty = node._last_calibration_detail = None


def _drop_last_sample(node):
    if hasattr(node, 'drop_last_sample'):
        node.drop_last_sample()
        return
    node.robot_samples.pop(); node.tracking_samples.pop(); node.sample_metrics.pop()


def _training_count(node):
    return node.training_count() if hasattr(node, 'training_count') else len(node.robot_samples)


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
    INITIAL_SAMPLES = 6  # earliest assessment, not a fixed bootstrap quota
    TRAINING = 20
    MAX_VIEWS = 30

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
        self.min_training = int(getattr(runner, 'min_training_samples', self.TRAINING))
        self.max_training = int(getattr(runner, 'max_training_samples', self.MAX_VIEWS))
        self.validation_target = max(3, int(getattr(runner, 'validation_views', 5)))
        self.board = None
        self.bootstrap_count = None
        self.previous_candidate = None
        self.training_complete = False
        self.last_quality_count = 0
        self.validation = []   # held-out camera poses, never in self.views
        self.validation_tracking = []
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
            fixed_board = self.board
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
        training = self.n.training_indices() if hasattr(self.n, 'training_indices') else range(len(self.views))
        robots = [sample_pose(self.n.robot_samples[i]) for i in training]
        tracking = [sample_pose(self.n.tracking_samples[i]) for i in training]
        geometry = bootstrap_geometry(robots)
        self.r.update('assessing', 'Assessing initial rotation coverage.', bootstrap_geometry=geometry)
        if self.estimate is None and not geometry['ready']:
            return False
        cal = self.n.get_calibration()
        if cal is None:
            if self.estimate is None:
                return False
            raise RuntimeError('Camera transform could not be estimated from collected samples.')
        estimate = sample_pose(cal)
        report = getattr(self.n, '_last_reprojection', None)
        board, fit = common_board(robots, tracking, estimate, self.r.framing.center, report)
        if (not np.isfinite(estimate).all() or not np.isfinite(board).all()
                or fit['max_translation_m'] > .01 or fit['max_rotation_deg'] > 6.):
            self.r.update('assessing', 'Estimate is not consistent enough to plan targeted views.', fit_consistency=fit)
            if self.estimate is None:
                self.previous_candidate = None
                return False
            raise RuntimeError('Collected views are inconsistent; no new calibration was saved.')
        if self.estimate is None:
            uncertainty = self.n._compute_uncertainty(cal)
            previous = self.previous_candidate
            self.previous_candidate = estimate.copy()
            shift = float(np.linalg.norm(estimate[:3, 3]-previous[:3, 3])) if previous is not None else float('inf')
            turn = float(np.degrees(Rotation.from_matrix(previous[:3, :3].T @ estimate[:3, :3]).magnitude())) if previous is not None else float('inf')
            ready = (position_uncertainty_ok(uncertainty, .01) and shift <= .01 and turn <= 3.
                     and fit['max_rotation_deg'] <= 3.)
            self.r.update('assessing', 'Checking stability of the initial estimate.',
                          bootstrap_ready=ready, bootstrap_sigma_m=(uncertainty or {}).get('worst_direction_sigma_m'),
                          bootstrap_shift_m=shift if np.isfinite(shift) else None,
                          bootstrap_rotation_change_deg=turn if np.isfinite(turn) else None)
            if not ready:
                return False
            self.bootstrap_count = len(self.views)
        self.estimate, self.board = estimate, board
        self.view_planner = None
        self.r.update('preparing', 'Joint camera/board estimate updated from training views.',
                      phase='refinement', fit_consistency=fit,
                      initial_samples=self.bootstrap_count,
                      targeted_samples=len(self.views)-self.bootstrap_count)
        return True

    @property
    def validating(self):
        return self.training_complete

    def capture(self):
        if self.validating:
            return self.capture_validation()
        robot = self.fk.camera(self.r.joint_positions())
        if not self.distinct(robot):
            return
        if not self.r.capture():
            return
        measured = sample_pose(self.n.robot_samples[-1])
        if not self.distinct(measured):
            _drop_last_sample(self.n)
            self.n._publish_status(None, None)
            return
        self.views.append(measured)
        self.r.update('capturing', 'Accepted a distinct training sample.', accepted=len(self.views),
                      pose=len(self.views), total=self.max_training + self.validation_target,
                      target_samples=self.min_training, max_training_samples=self.max_training,
                      phase='bootstrap' if self.estimate is None else 'refinement',
                      initial_samples=self.bootstrap_count or len(self.views),
                      targeted_samples=0 if self.bootstrap_count is None else len(self.views)-self.bootstrap_count,
                      required_validation_views=self.validation_target, validation_views=0, validation=None)
        if len(self.views) >= self.INITIAL_SAMPLES:
            self.update_estimate()

    def capture_validation(self):
        """Held-out view: distinct from every training and validation view,
        stored with role 'validation' and never used in the fit."""
        if len(self.validation) >= self.validation_target:
            return
        robot = self.fk.camera(self.r.joint_positions())
        if not informative_view(robot, self.views + self.validation):
            return
        self.r.capture_role = 'validation'
        try:
            if not self.r.capture():
                return
        finally:
            self.r.capture_role = 'training'
        self.validation.append(sample_pose(self.n.robot_samples[-1]))
        self.validation_tracking.append(sample_pose(self.n.tracking_samples[-1]))
        self.r.update('capturing', 'Accepted a held-out validation view.', validation_views=len(self.validation),
                      required_validation_views=self.validation_target, phase='validation')

    def done(self):
        if not self.training_complete:
            self.quality_ready()
        if not self.training_complete or len(self.validation) < self.validation_target:
            return False
        centre = np.r_[self.r.framing.center, 1.]
        predictions = [r @ self.estimate @ t for r,t in zip(self.validation, self.validation_tracking)]
        position = max(float(np.linalg.norm((b @ centre)[:3]-(self.board @ centre)[:3])) for b in predictions)
        rotation = max(float(np.degrees(Rotation.from_matrix(self.board[:3,:3].T @ b[:3,:3]).magnitude())) for b in predictions)
        passed = position <= .004 and rotation <= 3.
        self.r.update('assessing', 'Checking independent views against the frozen training estimate.',
                      validation={'passed': passed, 'position_max_m': position, 'rotation_max_deg': rotation,
                                  'views': len(self.validation), 'training_frozen': True})
        if not passed:
            raise RuntimeError(f'Independent validation failed: {position*1000:.2f} mm / {rotation:.2f} deg; no calibration saved.')
        return True

    def quality_ready(self):
        count = _training_count(self.n)
        if count < self.min_training:
            return False
        if self.estimate is None:
            if count >= self.max_training:
                geometry = bootstrap_geometry(self.views)
                if not geometry['ready']:
                    raise RuntimeError(
                        f"Initial rotation coverage insufficient at {count} samples: "
                        f"maximum {geometry['max_rotation_deg']:.2f}/20 deg, "
                        f"second-axis span {geometry['second_span_deg']:.2f}/10 deg. "
                        "Initial solve was not attempted; no calibration saved.")
                raise RuntimeError('Initial fit or stability checks failed at the sample limit; no calibration saved.')
            return False
        # Reassess every two additional poses. Validation never feeds this decision.
        if count == self.last_quality_count or (count < self.max_training and count-self.last_quality_count < 2):
            return False
        self.last_quality_count = count
        self.r.check()
        self.r.update('assessing', f'Assessing {count} training poses before independent validation.',
                      position_sigma_limit_m=self.sigma_limit, max_training_samples=self.max_training)
        cal = self.n.get_calibration(full=True)
        self.uncertainty = self.n._compute_uncertainty(cal) if cal is not None else None
        self.r.check()
        self.n._last_uncertainty = self.uncertainty
        self.n._publish_status(cal, None)
        report = getattr(self.n, '_last_reprojection', None)
        limits = dict(getattr(self.n, 'acceptance_limits', {}))
        limits['max_position_sigma_m'] = self.sigma_limit
        if report is not None:
            report = dict(report, uncertainty=self.uncertainty)
        quality = acceptance.evaluate(report, count, limits)
        sigma = (self.uncertainty or {}).get('worst_direction_sigma_m')
        self.r.update('assessing', quality['summary'], position_sigma_m=sigma, validation=quality)
        if quality['passed'] and position_uncertainty_ok(self.uncertainty, self.sigma_limit):
            self.estimate = sample_pose(cal)
            training = self.n.training_indices()
            self.board, _ = common_board([sample_pose(self.n.robot_samples[i]) for i in training],
                                         [sample_pose(self.n.tracking_samples[i]) for i in training],
                                         self.estimate, self.r.framing.center, report)
            self.training_complete = True
            self.view_planner = None
            self.r.update('planning', f'Training frozen; collecting {self.validation_target} independent validation views.',
                          phase='validation', required_validation_views=self.validation_target)
            return True
        if count >= self.max_training:
            raise RuntimeError(f'Training quality failed at {count} poses: {quality["summary"]} No calibration saved.')
        self.view_planner = None
        self.r.update('planning', 'More informative training views needed; final thresholds are unchanged.',
                      phase='refinement', validation=quality)
        return False

    def candidates(self, current, initial, pixels, blocked):
        candidates = []
        self.candidate_rejections = dict(joint_limits=0, excursion=0, previously_blocked=0, image_margin=0)
        correction = self.image_motion.centered_delta(current, pixels)
        coverage = coverage_score(bootstrap_geometry(self.views))
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
                    if np.any(target < self.r.lower) or np.any(target > self.r.upper):
                        self.candidate_rejections['joint_limits'] += 1
                        continue
                    if np.max(np.abs(target-initial)) > self.EXCURSION:
                        self.candidate_rejections['excursion'] += 1
                        continue
                    if tuple(np.round(target,4)) in blocked:
                        self.candidate_rejections['previously_blocked'] += 1
                        continue
                    predicted = self.image_motion.predict(current, delta, pixels)
                    if predicted is not None and (np.min(predicted) < .12 or np.max(predicted) > .88):
                        self.candidate_rejections['image_margin'] += 1
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
                    coverage_gain = 0.
                    if self.estimate is None:
                        coverage_gain = coverage_score(bootstrap_geometry(self.views + [pose])) - coverage
                    visits = sum(np.max(np.abs(target-old)) < .008 for old in self.visited_joints)
                    score = novelty + 100*coverage_gain + 2*gain + 10*weak_gain + margin + (.25 if not known else 0.) - .5*visits
                    candidates.append((score, target, limit))
        return sorted(candidates, key=lambda c:c[0], reverse=True)

    def run(self):
        initial = self.r.joint_positions()
        board = self.observe()
        if not self.r.framing.contains(board):
            raise ValueError('Center the full board with 10% image margin before bootstrap. No motion was sent.')
        _clear_samples(self.n)
        self.n._publish_status(None, None)
        self.r.update('capturing', 'Collecting the first observation without a camera mount transform.', phase='bootstrap',
                      pose=0, total=self.max_training + self.validation_target)
        self.capture()
        last_good = initial.copy()
        self.visited_joints = [initial.copy()]
        progress = AcquisitionProgress(len(self.views))
        blocked = set()
        for attempt in range(self.MAX_ATTEMPTS):
            self.r.check()
            if len(self.views) >= self.max_training and not self.validating:
                raise RuntimeError('Calibration sample limit reached; no new calibration was saved.')
            progress.observe(len(self.views) + len(self.validation))
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
                if self.validating:
                    raise RuntimeError(f'Only {len(self.validation)}/{self.validation_target} validation views were reachable; '
                                       'no new calibration was saved.')
                raise RuntimeError('No reachable diverse target with a visible board remains; no new calibration was saved.')
            if planned:
                limit = self.MAX_STEP
            else:
                candidates = self.candidates(current, initial, pixels, blocked)
                if not candidates:
                    self.r.require_joint_limits(current)
                    self.r.update('planning', 'Local candidates rejected.', planner_rejections=self.candidate_rejections)
                    raise RuntimeError('No feasible local exploration step remains. Rejections: ' +
                                       str(self.candidate_rejections) + '. Robot stopped; no new calibration was saved.')
                _, target, limit = candidates[0]
            progress.before_step(planned, advancing=planned and self.view_planner.advancing)
            self.allowed_step = limit
            self.r.update('moving', ('Moving along the planned trajectory to the next calibration view.' if planned else
                                     f'Local exploration: {progress.local_without_sample}/{progress.MAX_LOCAL_WITHOUT_SAMPLE} attempts without a new view.'),
                          attempts=self.attempts+1, step_degrees=float(np.degrees(limit)),
                          search_mode='targeted' if planned else 'local',
                          planner_rejections=getattr(self.view_planner,'rejections',{}), **progress.status())
            if planned:
                trajectory = getattr(self.view_planner, 'trajectory', None)
                if trajectory is not None:
                    # A timed trajectory starts at the previous measured pose.
                    # It is single-use, including when the endpoint has a small
                    # tracking residual accepted by the controller and settle().
                    self.view_planner.trajectory = None
                    self.r.move_planned(trajectory)
                else:
                    # No timed trajectory: follow the planner's joint path in
                    # bounded, individually collision-checked increments.
                    for waypoint in list(self.view_planner.route or [target]):
                        for _ in range(200):
                            here = self.r.joint_positions()
                            delta = waypoint - here
                            if np.max(np.abs(delta)) < .003:
                                break
                            step = here + delta*min(1., self.MAX_STEP/np.max(np.abs(delta)))
                            if not self.r.collision_free_path(here, step):
                                raise RuntimeError('Planned path is now in collision; robot stopped.')
                            self.r.move(step, monitor=self.r.joint_positions, speed_scale=1., minimum_duration=.5)
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
            if planned:
                # move_planned()/move() already require controller success and
                # measured settling. A second, tighter .003 rad gate used to
                # skip capture and replay the old timed path on the next loop.
                # Calibrate from the measured endpoint, never from the target.
                self.r.update('capturing', 'At target; waiting for stable board detection before capturing.')
                self.r.wait_for_tracking()
                observed = self.r.observed_board(recovering=False)
                # Endpoint observation verifies visibility, never reanchors the common board.
                consumed_planner = self.view_planner
            self.capture()
            if planned:
                consumed_planner.reject_goal()  # consumed; next path starts at actual
            progress.observe(len(self.views) + len(self.validation))
            self.r.update(self.r.status['state'], self.r.status['message'], **progress.status())
            if self.done():
                return
        raise RuntimeError('Calibration did not finish within the movement attempt limit; no new calibration was saved. '
                           f'Collected {len(self.n.robot_samples)}/15 samples.')
