"""Cancellable eye-in-hand acquisition using observed joint-space bootstrap."""
import copy
import json
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetStateValidity, GetMotionPlan
from controller_manager_msgs.srv import ListControllers
from .motion_planning import plan_joint_path
from .robot_configuration import chain_joints, matching_group, matching_controller
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState, CameraInfo
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker





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


class TrackingInterrupted(RuntimeError):
    """Cancel and settle before attempting stationary board reacquisition."""


class Stopped(Exception):
    pass


class AutomaticCalibration:
    def __init__(self, node):
        self.node = node
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.goal = None
        self.board_marker = None
        self.board_marker_pub = node.create_publisher(Marker, 'hand_eye_calibration/board_estimate', 1)
        self.joints = None
        self.joint_at = 0.0
        self.visible = False
        self.visible_at = 0.0
        self.board_revision = None
        self.camera_info = None
        self.board_spec = {key: node.get_parameter(key).value for key in
                           ('squares_x', 'squares_y', 'square_length_m')}
        self.status = {'state': 'idle', 'active': False, 'pose': 0, 'total': 15, 'strategy': 'adaptive_joint_bootstrap', 'phase': 'bootstrap',
                       'accepted': 0, 'message': 'Position the camera to see the board, then Calibrate.'}
        for key, value in {'auto_enabled': False, 'auto_check_collisions': True, 'auto_group': '',
                           'auto_max_position_sigma_m': .002, 'auto_max_training_samples': 15,
                           'auto_controller': ''}.items():
            node.declare_parameter(key, value)
        self.max_position_sigma = float(node.get_parameter('auto_max_position_sigma_m').value)
        self.max_training_samples = 15  # fixed six initial + nine targeted samples
        if not np.isfinite(self.max_position_sigma) or self.max_position_sigma <= 0:
            raise ValueError('Invalid automatic calibration quality limits.')
        self.check_collisions = bool(node.get_parameter("auto_check_collisions").value)
        self.status["enabled"] = bool(node.get_parameter("auto_enabled").value)
        self.group = ReentrantCallbackGroup()
        self.validity = node.create_client(GetStateValidity, "/check_state_validity", callback_group=self.group)
        self.description = node.create_client(GetParameters, '/robot_state_publisher/get_parameters', callback_group=self.group)
        self.action = None
        self.planning = node.create_client(GetMotionPlan, '/plan_kinematic_path', callback_group=self.group)
        self.semantic = node.create_client(GetParameters, '/move_group/get_parameters', callback_group=self.group)
        self.controllers = node.create_client(ListControllers, '/controller_manager/list_controllers', callback_group=self.group)
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
        if self.board_marker is not None:
            self.board_marker.header.stamp = self.node.get_clock().now().to_msg()
            self.board_marker.action = Marker.ADD if self.active else Marker.DELETE
            self.board_marker_pub.publish(self.board_marker)

    def show_board_estimate(self, board):
        marker = Marker()
        marker.header.frame_id = self.node.robot_base_frame
        marker.ns = 'estimated_calibration_board'
        marker.id = 0
        marker.type = Marker.CUBE
        center = (board @ np.r_[self.framing.center, 1])[:3]
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, center)
        q = Rotation.from_matrix(board[:3,:3]).as_quat()
        marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = map(float,q)
        marker.scale.x = self.board_spec['squares_x']*self.board_spec['square_length_m']
        marker.scale.y = self.board_spec['squares_y']*self.board_spec['square_length_m']
        marker.scale.z = .002
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1., .65, .1, .35
        marker.lifetime.sec = 2
        self.board_marker = marker

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
                resp.message = 'Automatic calibration is not enabled for this robot profile.'
            elif self.active or (self.thread is not None and self.thread.is_alive()):
                resp.message = 'Calibration is already running.'
            elif self.node.calibration_type != 'eye-in-hand':
                resp.message = 'Automatic calibration requires eye-in-hand.'
            elif not self.board_ready():
                resp.message = 'Position the camera so the board is detected, then Calibrate.'
            else:
                self.stop_event.clear()
                if self.board_marker is not None:
                    self.board_marker.action = Marker.DELETE
                    self.board_marker_pub.publish(self.board_marker)
                    self.board_marker = None
                self.status = {**self.status, 'active': True, 'state': 'preparing', 'pose': 0,
                               'accepted': 0, 'message': 'Checking initial view and controller.', 'validation': None, 'phase': 'bootstrap', 'skipped': 0,
                               'position_sigma_m': None, 'position_sigma_limit_m': self.max_position_sigma,
                               'target_samples': 15, 'max_training_samples': self.max_training_samples,
                               'attempts': 0, 'attempts_without_sample': 0, 'local_attempts_without_sample': 0,
                               'max_attempts_without_sample': 16, 'max_local_attempts_without_sample': 8,
                               'initial_samples': 0, 'targeted_samples': 0, 'validation_views': 0, 'required_validation_views': 0, 'fit_consistency': None,
                               'search_mode': None, 'planner_rejections': {}, 'robot_group': None,
                               'robot_joints': [], 'robot_controller': None}
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

    def fresh_board(self):
        if not self.board_ready():
            raise ValueError('Board is not detected.')
        tf = self.node.tf_buffer.lookup_transform(self.node.tracking_base_frame, self.node.tracking_marker_frame, rclpy.time.Time())
        age = (self.node.get_clock().now().nanoseconds - rclpy.time.Time.from_msg(tf.header.stamp).nanoseconds) / 1e9
        if not -0.1 <= age < 1.0:
            raise ValueError('Board pose is stale; check camera timestamps and the ROS clock.')
        return matrix(tf.transform)

    def collision_free_path(self, start, end):
        if not self.check_collisions:
            return True
        if not self.validity.wait_for_service(timeout_sec=1):
            raise RuntimeError('MoveIt collision checking is unavailable; no motion was sent.')
        # Same straight joint-space path as the zero-endpoint-velocity cubic.
        steps = max(2, int(np.ceil(np.max(np.abs(end-start)) / .015)) + 1)
        feedback = copy.deepcopy(self.joints)
        for fraction in np.linspace(0, 1, steps):
            self.check()
            req = GetStateValidity.Request()
            req.group_name = getattr(self, 'move_group', self.node.get_parameter('auto_group').value)
            req.robot_state.is_diff = True  # retain attached tools from the scene
            req.robot_state.joint_state = copy.deepcopy(feedback)
            values = dict(zip(self.names, start + fraction * (end-start)))
            req.robot_state.joint_state.position = [float(values.get(n, p))
                for n, p in zip(feedback.name, feedback.position)]
            result = self.wait(self.validity.call_async(req), 3)
            if result is None or not result.valid:
                return False
        return True

    def move(self, q, monitor=None, speed_scale=1.0, minimum_duration=2.5):
        self.check()
        start = self.joint_positions()
        if not self.collision_free_path(start, q):
            raise RuntimeError('Calibration path is in collision in the MoveIt scene; no motion was sent.')
        self.check()
        if np.max(np.abs(self.joint_positions() - start)) > .01:
            raise RuntimeError('Robot moved during path checking; retry calibration.')
        distance = float(np.max(np.abs(q - start)))
        seconds = max(minimum_duration, distance / 0.12)
        if self.check_collisions:
            # Cubic with zero endpoint velocities: peak v=1.5*d/T, a=6*d/T².
            seconds = max(minimum_duration, 1.5 * distance / .06, np.sqrt(6 * distance / .12))
        if hasattr(self, 'velocity_limits'):
            seconds = max(seconds, float(np.max(1.5*np.abs(q-start)/self.velocity_limits)))
        seconds /= max(.25, min(1., speed_scale))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.names
        from trajectory_msgs.msg import JointTrajectoryPoint
        for positions, stamp in ((start, 0.0), (q, seconds)):
            point = JointTrajectoryPoint()
            point.positions = list(map(float, positions))
            point.velocities = [0.0] * len(q)
            point.time_from_start = Duration(seconds=stamp).to_msg()
            goal.trajectory.points.append(point)
        self.execute_trajectory(goal, seconds, q, monitor)

    def move_planned(self, trajectory):
        import copy
        self.check()
        trajectory = copy.deepcopy(trajectory)
        if len(trajectory.points) < 2 or trajectory.joint_names != list(self.names):
            raise RuntimeError('Invalid planned calibration trajectory.')
        start = self.joint_positions()
        if np.max(np.abs(start-np.asarray(trajectory.points[0].positions))) > .01:
            raise RuntimeError('Robot moved since planning; trajectory was not sent.')
        times = np.array([p.time_from_start.sec+p.time_from_start.nanosec/1e9 for p in trajectory.points])
        if not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
            raise RuntimeError('MoveIt trajectory has invalid timing.')
        velocity_limit = np.minimum(self.velocity_limits, .18)
        scale = 1.
        previous = None
        for point, stamp in zip(trajectory.points, times):
            self.check()
            q = np.asarray(point.positions)
            if not np.isfinite(q).all() or np.any(q < self.lower) or np.any(q > self.upper):
                raise RuntimeError('Planned trajectory exceeds joint limits.')
            if len(point.velocities) != len(self.names) or len(point.accelerations) != len(self.names):
                raise RuntimeError('MoveIt trajectory lacks timed velocity/acceleration data.')
            scale = max(scale, float(np.max(np.abs(point.velocities)/velocity_limit)),
                        float(np.sqrt(np.max(np.abs(point.accelerations)/.36))))
            if previous is not None:
                old, old_stamp = previous
                scale = max(scale, float(np.max(np.abs(q-old)/(stamp-old_stamp)/velocity_limit)))
                if not self.collision_free_path(old, q):
                    raise RuntimeError('Planned path is now in collision; trajectory was not sent.')
            previous = (q, stamp)
        if not np.isfinite(scale):
            raise RuntimeError('Invalid planned trajectory speed.')
        # Uniform time scaling preserves the MoveIt path and its smooth derivatives.
        for point, stamp in zip(trajectory.points, times):
            point.time_from_start = Duration(seconds=float(stamp*scale)).to_msg()
            point.velocities = [float(v/scale) for v in point.velocities]
            point.accelerations = [float(a/scale**2) for a in point.accelerations]
        if np.max(np.abs(self.joint_positions()-start)) > .01:
            raise RuntimeError('Robot moved during path checks; trajectory was not sent.')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        self.execute_trajectory(goal, float(times[-1]*scale),
                                np.asarray(trajectory.points[-1].positions), self.joint_positions)

    def execute_trajectory(self, goal, seconds, q, monitor):
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
                    except (FramingCorrection, TrackingInterrupted):
                        # Wait for terminal cancellation before issuing any new
                        # command; a cancellation request alone is insufficient.
                        self.wait(handle.cancel_goal_async(), 3)
                        result = self.wait(result_future, 5)
                        if result.status not in (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_SUCCEEDED):
                            raise RuntimeError('Controller failed while pausing calibration.')
                        self.settle(self.joint_positions())
                        raise
                self.stop_event.wait(.05)
            result = self.wait(result_future, 1)
            if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code != 0:
                raise RuntimeError('Trajectory failed: ' + result.result.error_string)
            self.settle(q)
        except (FramingCorrection, TrackingInterrupted):
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
        # A rejected RGB frame does not invalidate the preceding accepted pose.
        # During motion use that pose only inside the existing 350 ms freshness
        # bound. Start/capture still require a currently detected board.
        try:
            stamped = self.node.tf_buffer.lookup_transform(self.node.tracking_base_frame,
                self.node.tracking_marker_frame, rclpy.time.Time())
        except TransformException as exc:
            raise TrackingInterrupted('Board detection lost during motion; pausing.') from exc
        board = matrix(stamped.transform)
        stamp = rclpy.time.Time.from_msg(stamped.header.stamp).nanoseconds
        if stamp != self.monitor_stamp:
            self.monitor_stamp, self.monitor_at = stamp, time.monotonic()
        age = (self.node.get_clock().now().nanoseconds - stamp) / 1e9
        if not -.1 <= age <= .35 or time.monotonic() - self.monitor_at > .75:
            raise TrackingInterrupted('Camera feedback stale during motion; pausing.')
        observation = self.optical_tracking @ board
        if not self.framing.contains(observation, margin=.03):
            raise RuntimeError('Board too close to image edge; stopping.')
        if not self.framing.contains(observation, margin=.08):
            raise FramingCorrection('Board approaching image edge.')

    def wait_for_tracking(self):
        """Called only after controller cancellation and measured settling."""
        deadline = time.monotonic() + 5
        first_stamp = last_stamp = None
        count = 0
        while time.monotonic() < deadline:
            self.check()
            self.joint_positions()
            try:
                board = self.fresh_board()
                stamped = self.node.tf_buffer.lookup_transform(self.node.tracking_base_frame,
                    self.node.tracking_marker_frame, rclpy.time.Time())
                stamp = rclpy.time.Time.from_msg(stamped.header.stamp).nanoseconds
                age = (self.node.get_clock().now().nanoseconds-stamp)/1e9
                if not -.1 <= age <= .2 or not self.framing.contains(self.optical_tracking @ board, margin=.03):
                    raise ValueError('Waiting for fresh, fully visible board.')
                if last_stamp is not None and stamp < last_stamp:
                    raise ValueError('Camera timestamp went backwards.')
                if stamp != last_stamp:
                    first_stamp = stamp if first_stamp is None else first_stamp
                    last_stamp = stamp
                    count += 1
                if count >= 3 and stamp-first_stamp >= 200_000_000:
                    return
            except (ValueError, TransformException):
                first_stamp = last_stamp = None
                count = 0
            self.stop_event.wait(.05)
        raise RuntimeError('Board did not return with stable detection within 5 seconds. Robot remains stopped; recenter the board and retry.')

    def configure_robot(self, root):
        self.names = chain_joints(root, self.node.robot_base_frame, self.node.robot_effector_frame)
        if len(self.names)<3:
            raise ValueError('Calibration requires at least three controlled joints.')
        limits = [root.find(f"joint[@name='{name}']/limit") for name in self.names]
        if any(limit is None for limit in limits):
            raise ValueError('Controlled joint limits are missing from the robot description.')
        self.lower = np.array([float(limit.get('lower')) for limit in limits])
        self.upper = np.array([float(limit.get('upper')) for limit in limits])
        self.velocity_limits = np.array([float(limit.get('velocity','nan')) for limit in limits])
        if not np.isfinite(self.velocity_limits).all() or np.any(self.velocity_limits<=0):
            raise ValueError('Robot URDF is missing positive joint velocity limits.')
        if not np.isfinite([self.lower,self.upper]).all() or np.any(self.lower>=self.upper):
            raise ValueError('Invalid robot joint limits.')
        if not self.semantic.wait_for_service(timeout_sec=2):
            raise RuntimeError('MoveIt semantic robot description is unavailable.')
        result = self.wait(self.semantic.call_async(GetParameters.Request(names=['robot_description_semantic'])),3)
        self.move_group = matching_group(root,result.values[0].string_value,self.names,
                                        self.node.get_parameter('auto_group').value)
        controller = self.node.get_parameter('auto_controller').value
        if not controller:
            if not self.controllers.wait_for_service(timeout_sec=2):
                raise RuntimeError('Controller discovery unavailable; configure auto_controller.')
            result = self.wait(self.controllers.call_async(ListControllers.Request()),3)
            controller = matching_controller(result.controller,self.names)
        if self.action is not None:
            self.action.destroy()
        self.action = ActionClient(self.node,FollowJointTrajectory,controller,callback_group=self.group)
        self.update('preparing', f'Robot configured: {self.move_group}, {len(self.names)} joints.',
                    robot_group=self.move_group, robot_joints=self.names, robot_controller=controller)

    def plan_joint_path(self, start, target):
        return plan_joint_path(self, start, target)

    def run(self):
        n = self.node
        try:
            if not self.description.wait_for_service(timeout_sec=2):
                raise RuntimeError('Robot description is unavailable.')
            req = GetParameters.Request(names=['robot_description'])
            urdf = self.wait(self.description.call_async(req), 3).values[0].string_value
            root = ET.fromstring(urdf)
            self.configure_robot(root)
            if not self.action.wait_for_server(timeout_sec=2):
                raise RuntimeError('Trajectory controller is unavailable.')
            if not self.planning.wait_for_service(timeout_sec=2):
                raise RuntimeError('MoveIt /plan_kinematic_path is unavailable; no motion was sent.')
            initial = self.joint_positions()
            if not self.collision_free_path(initial, initial):
                raise RuntimeError('Current robot pose is in collision in the MoveIt scene; no motion was sent.')
            self.settle(initial)
            from .bootstrap_calibration import BootstrapSession
            BootstrapSession(self, root).run()
            self.check()
            self.update('saving', 'Internal consistency and uncertainty passed; saving without independent validation.', validation=None)
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
