"""Cancellable eye-in-hand sequence using IK and direct controller trajectories."""
import copy
import json
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetPositionIK
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState, CameraInfo
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from .automatic_geometry import (camera_targets, validation_error, pose_variants,
                                 distinct_view, has_rotation_diversity)


from .visibility import BoardFraming, CameraKinematics, visible_joint_path


def matrix(tf):
    t, q = tf.translation, tf.rotation
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    m[:3, 3] = [t.x, t.y, t.z]
    return m


def sample_matrix(sample):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(sample[3:]).as_matrix()
    m[:3, 3] = sample[:3]
    return m


class FramingCorrection(Exception):
    """Motion was cancelled and stopped to permit a framing correction."""


class Stopped(Exception):
    pass


class AutomaticCalibration:
    def __init__(self, node):
        self.node = node
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.goal = None
        self.joints = None
        self.joint_at = 0.0
        self.visible = False
        self.visible_at = 0.0
        self.board_revision = None
        self.camera_info = None
        self.board_spec = {key: node.get_parameter(key).value for key in
                           ('squares_x', 'squares_y', 'square_length_m')}
        self.status = {'state': 'idle', 'active': False, 'pose': 0, 'total': 24,
                       'accepted': 0, 'message': 'Position the camera to see the board, then Calibrate.'}
        for key, value in {'auto_enabled': False, 'auto_group': 'arm',
                           'auto_ik_link': 'arm_tcp',
                           'auto_controller': '/arm_controller/follow_joint_trajectory',
                           'auto_joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']}.items():
            node.declare_parameter(key, value)
        self.status["enabled"] = bool(node.get_parameter("auto_enabled").value)
        self.group = ReentrantCallbackGroup()
        self.ik = node.create_client(GetPositionIK, '/compute_ik', callback_group=self.group)
        self.description = node.create_client(GetParameters, '/robot_state_publisher/get_parameters', callback_group=self.group)
        self.action = ActionClient(node, FollowJointTrajectory, node.get_parameter('auto_controller').value, callback_group=self.group)
        if node.camera_info_topic:
            node.create_subscription(CameraInfo, node.camera_info_topic, self._camera_info, qos_profile_sensor_data, callback_group=self.group)
        node.create_subscription(JointState, '/joint_states', self._joints, qos_profile_sensor_data, callback_group=self.group)
        node.create_subscription(Bool, '/hand_eye_calibration/chessboard_visible', self._visible, 10, callback_group=self.group)
        node.create_subscription(String, '/hand_eye_calibration/board_spec', self._board, 10, callback_group=self.group)
        node.create_service(Trigger, 'hand_eye_calibration/auto_start', self.start, callback_group=node._service_cb_group)
        node.create_service(Trigger, 'hand_eye_calibration/auto_stop', self.stop, callback_group=self.group)
        node.create_timer(0.5, self.publish, callback_group=self.group)

    @property
    def active(self):
        return bool(self.status['active'])

    def _joints(self, msg):
        self.joints, self.joint_at = msg, time.monotonic()

    def _visible(self, msg):
        self.visible, self.visible_at = msg.data, time.monotonic()

    def _camera_info(self, msg):
        previous = self.camera_info
        if previous is not None and self.active:
            fields = ('width', 'height', 'distortion_model')
            if (any(getattr(previous, k) != getattr(msg, k) for k in fields)
                    or previous.header.frame_id != msg.header.frame_id
                    or not np.array_equal(previous.k, msg.k)
                    or not np.array_equal(previous.d, msg.d)):
                self.stop_event.set()
        self.camera_info = msg

    def _board(self, msg):
        if self.active and self.board_revision is not None and msg.data != self.board_revision:
            self.stop_event.set()
        self.board_revision = msg.data
        try:
            data = json.loads(msg.data)
            self.board_spec.update({k: data[k] for k in self.board_spec if k in data})
        except (ValueError, TypeError):
            self.stop_event.set()

    def publish(self):
        # Reuse the existing GUI status channel; no additional subscription required.
        msg = String()
        msg.data = json.dumps({**self.node._status_payload, 'automatic': dict(self.status)})
        self.node.status_pub.publish(msg)

    def update(self, state, message, **values):
        with self.lock:
            self.status = {**self.status, 'state': state, 'message': message, **values}
        self.publish()

    def check(self):
        if self.stop_event.is_set() or not rclpy.ok():
            raise Stopped('Calibration stopped.')

    def wait(self, future, timeout):
        end = time.monotonic() + timeout
        while not future.done():
            self.check()
            if time.monotonic() > end:
                raise RuntimeError('ROS request timed out.')
            self.stop_event.wait(0.025)
        self.check()
        return future.result()

    def start(self, req, resp):
        with self.lock:
            if not self.node.get_parameter('auto_enabled').value:
                resp.message = 'Automatic calibration is configured for the Piper simulation profile.'
            elif self.active or (self.thread is not None and self.thread.is_alive()):
                resp.message = 'Calibration is already running.'
            elif self.node.calibration_type != 'eye-in-hand':
                resp.message = 'Automatic calibration requires eye-in-hand.'
            elif not self.board_ready():
                resp.message = 'Position the camera so the board is detected, then Calibrate.'
            else:
                self.stop_event.clear()
                self.status = {**self.status, 'active': True, 'state': 'preparing', 'pose': 0,
                               'accepted': 0, 'message': 'Checking initial view and controller.', 'validation': None, 'skipped': 0}
                self.thread = threading.Thread(target=self.run, daemon=True)
                self.thread.start()
                resp.success, resp.message = True, 'Automatic calibration started.'
        return resp

    def stop(self, req, resp):
        self.stop_event.set()
        with self.lock:
            if self.goal is not None:
                self.goal.cancel_goal_async()
        resp.success, resp.message = True, 'Stop requested; cancelling active trajectory.'
        return resp

    def board_ready(self):
        return self.visible and time.monotonic() - self.visible_at < 1.5

    def joint_positions(self):
        if self.joints is None or time.monotonic() - self.joint_at > 1.5:
            raise RuntimeError('Joint states are missing or stale.')
        values = dict(zip(self.joints.name, self.joints.position))
        q = np.array([values[n] for n in self.names])
        if not np.isfinite(q).all():
            raise RuntimeError('Invalid joint positions.')
        return q

    def tf(self, parent, child):
        return self.node.tf_buffer.lookup_transform(parent, child, rclpy.time.Time()).transform

    def fresh_board(self):
        if not self.board_ready():
            raise ValueError('Board is not detected.')
        tf = self.node.tf_buffer.lookup_transform(self.node.tracking_base_frame, self.node.tracking_marker_frame, rclpy.time.Time())
        age = (self.node.get_clock().now().nanoseconds - rclpy.time.Time.from_msg(tf.header.stamp).nanoseconds) / 1e9
        if not -0.1 <= age < 1.0:
            raise ValueError('Board pose is stale; check simulation time.')
        return matrix(tf.transform)

    def solve(self, camera, seed):
        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = self.node.get_parameter('auto_group').value
        ik.ik_link_name = self.node.get_parameter('auto_ik_link').value
        ik.avoid_collisions = False
        ik.timeout = Duration(seconds=0.25).to_msg()
        ik.robot_state.joint_state = copy.deepcopy(self.joints)
        values = dict(zip(self.names, seed))
        ik.robot_state.joint_state.position = [float(values.get(n, p)) for n, p in zip(self.joints.name, self.joints.position)]
        ik.pose_stamped.header.frame_id = self.node.robot_base_frame
        target = camera @ self.camera_ik
        p, q = target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_quat()
        pose = ik.pose_stamped.pose
        pose.position.x, pose.position.y, pose.position.z = map(float, p)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, q)
        result = self.wait(self.ik.call_async(req), 3)
        if result.error_code.val != 1:
            return None
        vals = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
        q = np.array([vals[n] for n in self.names])
        if not np.isfinite(q).all() or np.any(q < self.lower) or np.any(q > self.upper):
            return None
        if np.max(np.abs(q - seed)) > 0.85:
            return None  # reject IK branch jumps
        return q

    def move(self, q, monitor=None):
        self.check()
        start = self.joint_positions()
        seconds = max(2.5, float(np.max(np.abs(q - start))) / 0.12)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.names
        from trajectory_msgs.msg import JointTrajectoryPoint
        for positions, stamp in ((start, 0.0), (q, seconds)):
            point = JointTrajectoryPoint()
            point.positions = list(map(float, positions))
            point.velocities = [0.0] * len(q)
            point.time_from_start = Duration(seconds=stamp).to_msg()
            goal.trajectory.points.append(point)
        goal.goal_time_tolerance = Duration(seconds=2).to_msg()
        abandoned = threading.Event()
        future = self.action.send_goal_async(goal)
        def accepted(f):
            handle = f.result()
            with self.lock:
                if handle.accepted and (self.stop_event.is_set() or abandoned.is_set()):
                    handle.cancel_goal_async()
                    return
                self.goal = handle if handle.accepted else None
        future.add_done_callback(accepted)
        try:
            handle = self.wait(future, 5)
            if not handle.accepted:
                raise RuntimeError('Controller rejected trajectory.')
            result_future = handle.get_result_async()
            end = time.monotonic() + seconds * 3 + 10
            while not result_future.done():
                self.check()
                if time.monotonic() > end:
                    raise RuntimeError('Trajectory timed out.')
                if monitor is not None:
                    try:
                        monitor()
                    except FramingCorrection:
                        # Wait for terminal cancellation before issuing any new
                        # command; a cancellation request alone is insufficient.
                        self.wait(handle.cancel_goal_async(), 3)
                        result = self.wait(result_future, 5)
                        if result.status not in (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_SUCCEEDED):
                            raise RuntimeError('Controller failed while stopping for framing correction.')
                        self.settle(self.joint_positions())
                        raise
                self.stop_event.wait(.05)
            result = self.wait(result_future, 1)
            if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code != 0:
                raise RuntimeError('Trajectory failed: ' + result.result.error_string)
            self.settle(q)
        except FramingCorrection:
            raise
        except Exception:
            # Also cancel goals whose acceptance arrives after a timeout.
            abandoned.set()
            self.stop_event.set()
            with self.lock:
                if self.goal is not None:
                    self.goal.cancel_goal_async()
            raise
        finally:
            with self.lock:
                self.goal = None

    def settle(self, target):
        end, stable, previous = time.monotonic() + 8, None, self.joint_positions()
        while time.monotonic() < end:
            self.check()
            self.stop_event.wait(0.1)
            q = self.joint_positions()
            still = np.max(np.abs(q - previous)) < 0.001 and np.max(np.abs(q - target)) < 0.015
            previous = q
            stable = (stable or time.monotonic()) if still else None
            if stable is not None and time.monotonic() - stable >= 0.7:
                return
        raise RuntimeError('Robot did not settle at the requested pose.')

    def capture(self):
        for _ in range(2):
            self.check()
            end = time.monotonic() + 3
            while not self.board_ready() and time.monotonic() < end:
                self.check()
                self.stop_event.wait(0.1)
            try:
                self.fresh_board()
            except ValueError:
                continue
            response = self.node.capture_point_service_callback(Trigger.Request(), Trigger.Response())
            self.check()
            if response.success:
                return True
        return False

    def prepare_framing(self, root):
        if self.camera_info is None:
            raise ValueError('Waiting for CameraInfo; retry when the camera is streaming.')
        self.framing = BoardFraming(self.board_spec, self.camera_info)
        # Detector expresses its pose in tracking_base_frame. Transform to the
        # actual optical frame named by CameraInfo before projecting.
        self.optical_tracking = matrix(self.tf(self.camera_info.header.frame_id, self.node.tracking_base_frame))
        self.kinematics = CameraKinematics(root, self.node.robot_base_frame,
            self.node.get_parameter('auto_ik_link').value, self.names, self.camera_ik)

    def fits(self, observation):
        return self.framing.contains(self.optical_tracking @ observation)

    def optical_camera(self, q):
        return self.kinematics.camera(q) @ np.linalg.inv(self.optical_tracking)

    def path_fits(self, start, end, base_board, margin=None):
        return visible_joint_path(start, end, self.optical_camera, self.framing, base_board, margin)

    def observed_board(self, recovering=False):
        # After settling, require a detection captured after the stop, not the
        # last transform from the preceding movement.
        after = self.node.get_clock().now().nanoseconds
        end = time.monotonic() + 4
        while time.monotonic() < end:
            self.check()
            try:
                board = self.fresh_board()
                stamped = self.node.tf_buffer.lookup_transform(self.node.tracking_base_frame,
                    self.node.tracking_marker_frame, rclpy.time.Time())
                if rclpy.time.Time.from_msg(stamped.header.stamp).nanoseconds > after:
                    if not self.framing.contains(self.optical_tracking @ board, margin=.03 if recovering else .10):
                        raise RuntimeError('Board reached the image margin; sequence stopped. Recenter the board and retry.')
                    return board
            except ValueError:
                pass
            self.stop_event.wait(.05)
        raise RuntimeError('Fresh board detection lost; sequence stopped at the current pose.')

    def motion_monitor(self):
        # Check new image timestamps as well as ROS age: a paused simulation or
        # stalled detector must not keep an old detection valid indefinitely.
        self.joint_positions()
        try:
            board = self.fresh_board()
        except ValueError as exc:
            raise RuntimeError('Board detection lost during motion; stopping.') from exc
        stamped = self.node.tf_buffer.lookup_transform(self.node.tracking_base_frame,
            self.node.tracking_marker_frame, rclpy.time.Time())
        stamp = rclpy.time.Time.from_msg(stamped.header.stamp).nanoseconds
        if stamp != self.monitor_stamp:
            self.monitor_stamp, self.monitor_at = stamp, time.monotonic()
        age = (self.node.get_clock().now().nanoseconds - stamp) / 1e9
        if age > .35 or time.monotonic() - self.monitor_at > .75:
            raise RuntimeError('Camera feedback stale during motion; stopping.')
        observation = self.optical_tracking @ board
        if not self.framing.contains(observation, margin=.03):
            raise RuntimeError('Board too close to image edge; stopping.')
        if not self.framing.contains(observation, margin=.08):
            raise FramingCorrection('Board approaching image edge.')

    def guarded_move(self, target):
        desired = target.copy()
        recovering = False
        for attempt in range(4):
            self.check()
            start = self.joint_positions()
            board = self.observed_board(recovering=recovering)
            base_board = self.kinematics.camera(start) @ board
            if np.max(np.abs(desired-start)) < .003 and not recovering:
                return
            if recovering or not self.path_fits(start, desired, base_board):
                camera = self.optical_camera(desired)
                corrected = self.framing.corrected(camera, base_board)
                q = None if corrected is None else self.solve(corrected @ self.optical_tracking, start)
                if q is None:
                    raise RuntimeError('No reachable correction keeps the board in frame; sequence stopped.')
                desired = q
                # Recovery starts inside the normal margin. Require full board
                # visibility throughout and restore the 10% margin at its end.
                margin = .03 if recovering else None
                if not self.path_fits(start, desired, base_board, margin=margin):
                    raise RuntimeError('Correction path leaves the image margin; sequence stopped.')
                self.update('moving', 'Correcting board framing while preserving camera tilt.')
            self.monitor_stamp, self.monitor_at = None, time.monotonic()
            def monitor():
                try:
                    self.motion_monitor()
                except FramingCorrection:
                    # During recovery tolerate the warning band, but the hard
                    # edge and freshness checks remain active.
                    if not recovering:
                        raise
            try:
                self.move(desired, monitor=monitor)
                self.observed_board()
                return
            except FramingCorrection:
                recovering = True
                self.update('moving', 'Paused near image margin; checking a correction.')
        raise RuntimeError('Framing correction did not converge; sequence stopped.')

    def run(self):
        n = self.node
        try:
            self.names = list(n.get_parameter('auto_joint_names').value)
            if not self.ik.wait_for_service(timeout_sec=2) or not self.action.wait_for_server(timeout_sec=2):
                raise RuntimeError('IK service or trajectory controller is unavailable.')
            if not self.description.wait_for_service(timeout_sec=2):
                raise RuntimeError('Robot description is unavailable.')
            req = GetParameters.Request(names=['robot_description'])
            urdf = self.wait(self.description.call_async(req), 3).values[0].string_value
            root = ET.fromstring(urdf)
            limits = [root.find(f"joint[@name='{name}']/limit") for name in self.names]
            self.lower = np.array([float(l.attrib['lower']) for l in limits])
            self.upper = np.array([float(l.attrib['upper']) for l in limits])
            initial = self.joint_positions()
            self.settle(initial)
            board = self.fresh_board()
            base_camera = matrix(self.tf(n.robot_base_frame, n.tracking_base_frame))
            self.camera_ik = matrix(self.tf(n.tracking_base_frame, n.get_parameter('auto_ik_link').value))
            self.prepare_framing(root)
            if not self.fits(board):
                raise ValueError('Center the entire board with 10% image margin before Calibrate. No motion was sent.')
            base_board = base_camera @ board
            targets = camera_targets(base_camera, board)
            plan, views, seed = [], [], initial
            for i, target in enumerate(targets):
                self.check()
                self.update('preparing', 'Checking reachable views; adapting motion to the initial pose.', pose=i + 1)
                q, chosen = None, None
                for candidate in pose_variants(base_camera, target):
                    if not self.fits(np.linalg.inv(candidate) @ base_board):
                        optical = candidate @ np.linalg.inv(self.optical_tracking)
                        corrected = self.framing.corrected(optical, base_board)
                        if corrected is None:
                            continue
                        candidate = corrected @ self.optical_tracking
                    if not distinct_view(candidate, views):
                        continue
                    q = self.solve(candidate, seed)
                    if q is not None and self.path_fits(seed, q, base_board):
                        chosen = candidate
                        break
                    q = None
                if q is not None:
                    views.append(chosen)
                    plan.append((i, q, chosen))
                    seed = q
            training = [pose for i, _, pose in plan if i < len(targets) - 3]
            checks = [i for i, _, _ in plan if i >= len(targets) - 3]
            if len(training) < 12 or len(checks) < 2 or not has_rotation_diversity(training):
                raise RuntimeError('Too few valid views from this starting pose. Bend the arm further from its limits, keep the board visible, and retry. No motion was sent.')
            self.check()
            self.settle(initial)
            current_board = self.fresh_board()
            if (np.linalg.norm(current_board[:3, 3] - board[:3, 3]) > 0.01 or
                    Rotation.from_matrix(board[:3, :3].T @ current_board[:3, :3]).magnitude() > np.deg2rad(3)):
                raise RuntimeError('Initial view changed during preparation; keep robot and board still and retry.')
            n.robot_samples.clear()
            n.tracking_samples.clear()
            n.sample_metrics.clear()
            n._last_uncertainty = n._last_calibration_detail = None
            n._publish_status(None, None)
            skipped, holdouts = len(targets) - len(plan), []
            for i, q, _ in plan:
                self.check()
                validation = i >= len(targets) - 3
                self.update('validating' if validation else 'moving', 'Moving to the next view.', pose=i + 1, skipped=skipped)
                self.guarded_move(q)
                self.update('capturing', 'Robot stationary; collecting fresh board frames.')
                if self.capture():
                    if validation:
                        holdouts.append((sample_matrix(n.robot_samples.pop()), sample_matrix(n.tracking_samples.pop())))
                        n.sample_metrics.pop()
                        n._last_calibration_detail = None
                        n._publish_status(None, None)
                    else:
                        self.update('capturing', 'Sample accepted.', accepted=len(n.robot_samples))
                else:
                    skipped += 1
                    self.update('skipped', 'No stable board detection; sample skipped.', skipped=skipped)
            self.update('returning', 'Returning to the initial position.')
            self.guarded_move(initial)
            if len(n.robot_samples) < 12 or len(holdouts) < 2:
                raise RuntimeError('Too few valid views. Need 12 samples and 2 validation views; choose a better initial view and retry.')
            self.update('solving', 'Computing camera calibration and checking independent views.')
            cal = n.get_calibration()
            if cal is None:
                raise RuntimeError('Calibration could not be solved.')
            check = validation_error([sample_matrix(s) for s in n.robot_samples],
                                     [sample_matrix(s) for s in n.tracking_samples], sample_matrix(cal), holdouts)
            self.update('validating', 'Checking board consistency in base frame.', validation=check)
            if check['max_translation_m'] > 0.01 or check['max_rotation_deg'] > 3:
                raise RuntimeError('Independent validation failed (limit 10 mm / 3 degrees). Result was not saved.')
            self.check()
            self.update('saving', 'Validation passed; estimating uncertainty and saving.')
            response = n.save_calibration_service_callback(Trigger.Request(), Trigger.Response())
            self.check()
            if not response.success:
                raise RuntimeError(response.message)
            self.update('completed', response.message, active=False)
        except Stopped as exc:
            self.update('stopped', str(exc), active=False)
        except Exception as exc:
            n.get_logger().error(f'Automatic calibration: {exc}')
            self.update('failed', str(exc), active=False)
        finally:
            with self.lock:
                if self.goal is not None:
                    self.goal.cancel_goal_async()
                    self.goal = None

    def close(self):
        self.stop(None, Trigger.Response())
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=4)
