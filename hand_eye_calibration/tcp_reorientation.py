"""Reorientation check for the tool TCP (the standard ABB/KUKA/FANUC/Yaskawa/UR
validation): the robot rotates the tool about its fitted tip above the spike;
the operator then closes the small stand-off and touches the spike, and those
touches are scored as held-out validation poses without refitting.

Each move is one MoveIt Cartesian path for the flange that keeps the estimated
tip on a line: lift to `clearance` above the spike, rotate about the tip, then
descend to `standoff` above it. A wrong TCP shows up as the tip not arriving
above the spike, and quantitatively as the validation residual.

Execution reuses the automatic hand-eye trajectory safety: MoveIt collision
checks on every segment, speed/acceleration caps, controller cancel on stop,
optional operator heartbeat.
"""
import copy
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import Pose
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath, GetStateValidity
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from .automatic_calibration import AutomaticCalibration, Stopped
from .robot_configuration import chain_joints, matching_controller, matching_group
from . import tcp_quality


class TcpMotion:
    # Trajectory execution and checks are shared with automatic hand-eye.
    wait = AutomaticCalibration.wait
    settle = AutomaticCalibration.settle
    collision_free_path = AutomaticCalibration.collision_free_path
    move_planned = AutomaticCalibration.move_planned
    execute_trajectory = AutomaticCalibration.execute_trajectory
    joint_positions = AutomaticCalibration.joint_positions

    def __init__(self, node, base_frame, flange_frame, group='', controller='', check_collisions=True,
                 heartbeat_timeout=0.0):
        self.node = node
        self.base_frame, self.flange_frame = base_frame, flange_frame
        self.group_param, self.controller_param = group, controller
        self.check_collisions = bool(check_collisions)
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.heartbeat_at = None
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.goal = None
        self.action = None
        self.thread = None
        self.joints, self.joint_at = None, 0.0
        self.names = None
        self.stationary_ns = None
        self.state = {'state': 'idle', 'message': '', 'active': False, 'index': 0, 'total': 0}
        cb = ReentrantCallbackGroup()
        self.cb = cb
        self.validity = node.create_client(GetStateValidity, '/check_state_validity', callback_group=cb)
        self.cartesian = node.create_client(GetCartesianPath, '/compute_cartesian_path', callback_group=cb)
        self.description = node.create_client(GetParameters, '/robot_state_publisher/get_parameters', callback_group=cb)
        self.semantic = node.create_client(GetParameters, '/move_group/get_parameters', callback_group=cb)
        self.controllers = node.create_client(ListControllers, '/controller_manager/list_controllers', callback_group=cb)
        node.create_subscription(JointState, '/joint_states', self._joints, qos_profile_sensor_data, callback_group=cb)
        node.create_subscription(String, 'tool_tcp_calibration/motion_heartbeat', self._heartbeat, 10, callback_group=cb)

    @property
    def active(self):
        return bool(self.state['active'])

    def _joints(self, msg):
        self.joints, self.joint_at = msg, time.monotonic()

    def _heartbeat(self, msg):
        self.heartbeat_at = time.monotonic()

    def heartbeat_ok(self):
        return self.heartbeat_timeout <= 0 or (
            self.heartbeat_at is not None and time.monotonic() - self.heartbeat_at <= self.heartbeat_timeout)

    def check(self):
        if self.stop_event.is_set() or not rclpy.ok():
            raise Stopped('Motion stopped.')
        if not self.heartbeat_ok():
            self.stop_event.set()
            with self.lock:
                if self.goal is not None:
                    self.goal.cancel_goal_async()
            raise Stopped('Operator heartbeat lost (GUI closed or bridge down); robot stopped.')

    def update(self, state, message, **values):
        self.state = {**self.state, 'state': state, 'message': message, **values}

    def stop(self):
        self.stop_event.set()
        with self.lock:
            if self.goal is not None:
                self.goal.cancel_goal_async()

    # -- configuration ------------------------------------------------------
    def configure(self):
        if self.names is not None and self.action is not None:
            return
        if not self.description.wait_for_service(timeout_sec=2):
            raise RuntimeError('Robot description is unavailable.')
        urdf = self.wait(self.description.call_async(GetParameters.Request(names=['robot_description'])), 3)
        root = ET.fromstring(urdf.values[0].string_value)
        self.names = chain_joints(root, self.base_frame, self.flange_frame)
        limits = [root.find(f"joint[@name='{n}']/limit") for n in self.names]
        if any(limit is None for limit in limits):
            raise ValueError('Joint limits missing from the robot description.')
        self.lower = np.array([float(l.get('lower')) for l in limits])
        self.upper = np.array([float(l.get('upper')) for l in limits])
        self.velocity_limits = np.array([float(l.get('velocity', 'nan')) for l in limits])
        if not np.isfinite(self.velocity_limits).all() or np.any(self.velocity_limits <= 0):
            raise ValueError('Robot URDF is missing positive joint velocity limits.')
        if not self.semantic.wait_for_service(timeout_sec=2):
            raise RuntimeError('MoveIt semantic robot description is unavailable.')
        srdf = self.wait(self.semantic.call_async(GetParameters.Request(names=['robot_description_semantic'])), 3)
        self.move_group = matching_group(root, srdf.values[0].string_value, self.names, self.group_param)
        controller = self.controller_param
        if not controller:
            if not self.controllers.wait_for_service(timeout_sec=2):
                raise RuntimeError('Controller discovery unavailable; set auto_controller.')
            controller = matching_controller(self.wait(self.controllers.call_async(ListControllers.Request()), 3).controller,
                                             self.names)
        self.action = ActionClient(self.node, FollowJointTrajectory, controller, callback_group=self.cb)
        if not self.action.wait_for_server(timeout_sec=2):
            raise RuntimeError('Trajectory controller is unavailable.')

    # -- planning -------------------------------------------------------------
    def plan_cartesian(self, flange_waypoints, speed_scale=0.1):
        if not self.cartesian.wait_for_service(timeout_sec=2):
            raise RuntimeError('MoveIt /compute_cartesian_path is unavailable; no motion was sent.')
        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.start_state.is_diff = True
        req.start_state.joint_state = copy.deepcopy(self.joints)
        req.group_name = self.move_group
        req.link_name = self.flange_frame
        for w in flange_waypoints:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, w[:3])
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, w[3:7])
            req.waypoints.append(pose)
        req.max_step = 0.005
        req.jump_threshold = 0.0
        req.prismatic_jump_threshold = 0.0
        req.revolute_jump_threshold = 0.5
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = float(speed_scale)
        req.max_acceleration_scaling_factor = float(speed_scale)
        result = self.wait(self.cartesian.call_async(req), 10)
        if result.error_code.val != MoveItErrorCodes.SUCCESS or result.fraction < 0.999:
            raise RuntimeError(f'Only {result.fraction * 100:.0f}% of the reorientation path is feasible '
                               '(reach, joint limits or collision); no motion was sent.')
        trajectory = copy.deepcopy(result.solution.joint_trajectory)
        if set(trajectory.joint_names) != set(self.names):
            raise RuntimeError('MoveIt returned a trajectory for different joints.')
        order = [trajectory.joint_names.index(n) for n in self.names]
        trajectory.joint_names = list(self.names)
        for point in trajectory.points:
            point.positions = [point.positions[i] for i in order]
            for field in ('velocities', 'accelerations'):
                values = getattr(point, field)
                if len(values) != len(order):
                    raise RuntimeError('MoveIt trajectory lacks timing derivatives.')
                setattr(point, field, [values[i] for i in order])
        return trajectory

    # -- one reorientation step ------------------------------------------------
    def run_step(self, current_flange, target_quat, tcp_translation, spike_point, spike_axis,
                 clearance, standoff, on_done):
        def worker():
            try:
                self.check()
                self.configure()
                n = np.asarray(spike_axis, float) / np.linalg.norm(spike_axis)
                spike = np.asarray(spike_point, float)
                R_now = current_flange[3:7]
                tip_now = tcp_quality.flange_pose_for_tip
                t = np.asarray(tcp_translation, float)
                from scipy.spatial.transform import Rotation
                tip_current = np.asarray(current_flange[:3]) + Rotation.from_quat(R_now).apply(t)
                high = spike + clearance * n
                low = spike + standoff * n
                waypoints = []
                # 1. lift the tip straight up to the clearance height (keep orientation)
                lift_from = max(float((tip_current - spike) @ n), standoff)
                for s in np.linspace(0, 1, 4)[1:]:
                    tip = spike + (lift_from + s * (clearance - lift_from)) * n
                    waypoints.append(tip_now(tip, R_now, t))
                # 2. rotate about the tip at clearance height
                waypoints += tcp_quality.tip_preserving_waypoints(R_now, target_quat, high, t, steps=10)
                # 3. descend to the stand-off
                for s in np.linspace(0, 1, 4)[1:]:
                    waypoints.append(tip_now(high + s * (low - high), target_quat, t))
                self.update('planning', 'Planning the reorientation path with MoveIt.')
                trajectory = self.plan_cartesian(waypoints)
                self.update('moving', 'Moving slowly to the reoriented pose above the spike.')
                self.move_planned(trajectory)
                self.update('done', f'At the reoriented pose, tip {standoff * 1000:.0f} mm above the spike. '
                                    'Lower the tip onto the spike by hand (no other motion), then Capture.',
                            active=False)
                on_done(True)
            except Stopped as exc:
                self.update('stopped', str(exc), active=False)
                on_done(False)
            except Exception as exc:  # noqa: BLE001 - reported to the operator
                self.node.get_logger().error(f'TCP reorientation: {exc}')
                self.update('failed', str(exc), active=False)
                on_done(False)
            finally:
                with self.lock:
                    if self.goal is not None:
                        self.goal.cancel_goal_async()
                        self.goal = None
        self.stop_event.clear()
        self.update('preparing', 'Checking robot, MoveIt and controller.', active=True)
        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()
