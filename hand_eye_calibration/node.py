# !/usr/bin/env python3
"""
Collect poses and perform calibration
"""

import math
import os
import time
import threading
from datetime import datetime, timezone
import yaml

import numpy as np
import rclpy
from rclpy.wait_for_message import wait_for_message

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Duration
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from geometry_msgs.msg import Transform
from sensor_msgs.msg import CameraInfo, PointCloud2
from scipy.spatial.transform import Rotation as Rot
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from .calibration_backend import CalibrationBackend
from .calibration_status import build_calibration_status, status_to_json
from .capture_timing import LatencyModel, ObservationBuffer, frame_record, parse_observation
from . import acceptance as acceptance_gate
from . import calibration_dataset


def get_transform(tf_message: Transform):
    tr = tf_message.translation
    qt = tf_message.rotation
    out = [tr.x, tr.y, tr.z, qt.x, qt.y, qt.z, qt.w]
    return out

def tf_list_to_string(mlist: list):
    return "tx, ty, tz, qx, qy, qz, qw: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]" % tuple(mlist)

def urdf_list_to_string(mlist: list):
    return "translation: %.4f, %.4f, %.4f   rpy: %.4f, %.4f, %.4f" % tuple(mlist)

def tf_to_urdf_tf(mlist: list):
    """
    Transform tx, ty, tz, qx, qy, qz, qw into tx, ty, tz, r, p, y
    """
    res = mlist[0:3]

    e = list(Rot.from_quat(mlist[3:]).as_euler(seq="ZYX"))
    """
    The roll-pitchRyaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.
    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    """
    r, p, y = e[2], e[1], e[0]
    res += [r, p, y]
    return res

def transform_to_matrix(tfl: list):
    mat = np.eye(4)
    mat[:3, :3] = Rot.from_quat(tfl[3:]).as_matrix()
    mat[:3, 3] = np.array(tfl[:3], dtype=float)
    return mat

def matrix_to_residual(mat):
    translation_error = float(np.linalg.norm(mat[:3, 3]))
    rotation_error = float(Rot.from_matrix(mat[:3, :3]).magnitude())
    return translation_error, rotation_error


class DataCollector(Node):

    def __init__(self):
        mname = "hand_eye_calibration"
        super().__init__(mname)

        self.declare_parameter('tracking_base_frame', "")
        self.declare_parameter('tracking_marker_frame', "")
        self.declare_parameter('robot_base_frame', "")
        self.declare_parameter('robot_effector_frame', "")
        # Link that ends the moved joint chain (MoveIt group/controller). Empty: robot_effector_frame.
        # Differs when the camera sits before the last joint, e.g. Piper camera on link5, chain to link6.
        self.declare_parameter('robot_motion_tip_frame', "")
        # options are eye-in-hand or eye-on-base
        self.declare_parameter('calibration_type', "eye-on-base")
        self.declare_parameter('calibration_file', os.path.expanduser("~/.ros/hand_eye_calibration.yaml"))
        self.declare_parameter('pointcloud_topic', "/oak/rgbd/points")
        self.declare_parameter('image_topic', "")
        self.declare_parameter('camera_info_topic', "")
        self.declare_parameter('squares_x', 13)
        self.declare_parameter('squares_y', 9)
        self.declare_parameter('square_length_m', 0.015)
        self.declare_parameter('marker_size', 0.0)
        # Burst capture: instead of a single TF lookup per sample, gather a
        # short burst of freshly-published (robot, tracking) pairs — each pair
        # synchronized on the tracking TF's own timestamp rather than a guessed
        # "now - 1s" offset — and robustly average them into one sample.
        self.declare_parameter('capture_burst_duration_s', 0.6)
        self.declare_parameter('capture_burst_samples', 5)
        # Bootstrap resamples used to estimate how uncertain the calibration is.
        # This costs a full refit per resample (order 10 s at 20 samples / 30
        # resamples), which is why it runs on save or on explicit request
        # rather than after every capture. 0 disables it.
        self.declare_parameter('bootstrap_samples', 30)
        # Raw ChArUco corners from the detector: the final solve minimises
        # their reprojection error, and their stamps measure camera latency.
        self.declare_parameter('observation_topic', '/charuco_detector/observation')
        # < 0: measure latency from observation stamps. >= 0: use this value.
        self.declare_parameter('camera_latency_s', -1.0)
        self.declare_parameter('latency_margin_s', 0.25)
        # A frame is used only if the robot was already at the same pose this
        # long before the frame's stamp (exposure after the robot stopped).
        self.declare_parameter('stationary_window_s', 0.3)
        # 'reprojection' (default) or 'axxb' (closed form + AX=XB refinement only).
        self.declare_parameter('solver', 'intrinsic_pose')
        self.declare_parameter('estimate_intrinsics', False)
        # Every save writes the raw dataset (corners, poses, joints, images)
        # here, so the result can be re-solved and compared offline.
        self.declare_parameter('dataset_dir', os.path.expanduser('~/.ros/hand_eye_calibration_runs'))
        self.declare_parameter('dataset_save_images', True)
        # 'enforce': a result that fails acceptance is written only as
        # <calibration_file>.rejected.yaml. 'warn': written anyway, flagged.
        self.declare_parameter('acceptance_mode', 'enforce')
        for key, value in acceptance_gate.DEFAULT_LIMITS.items():
            self.declare_parameter('accept_' + key, value)
        # Touch-off validation: TCP frame that is brought onto a board corner.
        self.declare_parameter('touchoff_tcp_frame', 'arm_tcp')
        self.declare_parameter('touchoff_corner_id', 0)

        self.tracking_base_frame = str(self.get_parameter('tracking_base_frame').value)
        self.tracking_marker_frame = str(self.get_parameter('tracking_marker_frame').value)
        self.robot_base_frame = str(self.get_parameter('robot_base_frame').value)
        self.robot_effector_frame = str(self.get_parameter('robot_effector_frame').value)
        self.robot_motion_tip_frame = (str(self.get_parameter('robot_motion_tip_frame').value)
                                       or self.robot_effector_frame)
        self.calibration_type = str(self.get_parameter('calibration_type').value)
        self.pointcloud_topic = str(self.get_parameter('pointcloud_topic').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.marker_size = float(self.get_parameter('marker_size').value)
        self.capture_burst_duration_s = float(self.get_parameter('capture_burst_duration_s').value)
        self.capture_burst_samples = int(self.get_parameter('capture_burst_samples').value)
        self.bootstrap_samples = int(self.get_parameter('bootstrap_samples').value)
        self.solver_name = str(self.get_parameter('solver').value)
        if self.solver_name not in ('intrinsic_pose', 'reprojection', 'axxb'):
            raise ValueError("solver must be 'intrinsic_pose', 'reprojection' or 'axxb'")
        self.estimate_intrinsics = bool(self.get_parameter('estimate_intrinsics').value)
        self.stationary_window_s = float(self.get_parameter('stationary_window_s').value)
        self.acceptance_mode = str(self.get_parameter('acceptance_mode').value)
        if self.acceptance_mode not in ('enforce', 'warn'):
            raise ValueError("acceptance_mode must be 'enforce' or 'warn'")
        self.acceptance_limits = {key: type(value)(self.get_parameter('accept_' + key).value)
                                  for key, value in acceptance_gate.DEFAULT_LIMITS.items()}
        self.latency = LatencyModel(margin_s=float(self.get_parameter('latency_margin_s').value),
                                    configured_s=float(self.get_parameter('camera_latency_s').value))
        self.observations = ObservationBuffer()
        self._camera_model = None
        self._board_spec_seen = None
        self._latest_image = None

        # The capture callback blocks for the duration of the capture burst
        # while it waits for fresh TF. Put the services in their own callback
        # group so that, under the MultiThreadedExecutor set up in main(), the
        # TF listener's (reentrant) subscriptions keep running on another
        # thread and the buffer actually advances while we wait. Without this
        # the burst would sit on a buffer that never updates.
        self._service_cb_group = MutuallyExclusiveCallbackGroup()

        self.capture_point_service_name = mname + "/capture_point"
        self.capture_point_service = self.create_service(
            Trigger,
            self.capture_point_service_name,
            self.capture_point_service_callback,
            callback_group=self._service_cb_group)

        self.save_calibration_service_name = mname + "/save_calibration"
        self.save_calibration_service = self.create_service(
            Trigger,
            self.save_calibration_service_name,
            self.save_calibration_service_callback,
            callback_group=self._service_cb_group)

        # On-demand uncertainty: the number that answers "have I collected
        # enough samples yet?", which is exactly the question you want answered
        # DURING collection, not only at save time.
        self.estimate_uncertainty_service_name = mname + "/estimate_uncertainty"
        self.estimate_uncertainty_service = self.create_service(
            Trigger,
            self.estimate_uncertainty_service_name,
            self.estimate_uncertainty_service_callback,
            callback_group=self._service_cb_group)

        self.status_topic = mname + "/status"
        status_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.status_pub = self.create_publisher(String, self.status_topic, status_qos)

        # Transform listener.
        self.tf_buffer = Buffer()
        self._listener = TransformListener(self.tf_buffer, self)

        self.robot_samples = list()
        self.tracking_samples = list()
        self.sample_metrics = list()
        # Parallel to robot_samples: raw frames, joints, role, one image.
        self.sample_frames = list()
        self.sample_joints = list()
        self.sample_roles = list()
        self.sample_images = list()
        self._last_reprojection = None
        self._last_acceptance = None
        self._reprojection_state = None
        self.touchoff_references = {}
        self.touchoff_results = []
        self._preflight_logged = False
        self._last_pointcloud_frame = None
        self._last_camera_info_frame = None
        self._last_calibration_detail = None
        self._last_uncertainty = None

        self.create_subscription(String, str(self.get_parameter('observation_topic').value),
                                 self._observation_callback, 20)
        if bool(self.get_parameter('dataset_save_images').value) and self.image_topic:
            from sensor_msgs.msg import Image
            from rclpy.qos import qos_profile_sensor_data
            self.create_subscription(Image, self.image_topic, self._image_callback, qos_profile_sensor_data)
        for name, callback in (('touchoff_reference', self.touchoff_reference_callback),
                               ('touchoff_capture', self.touchoff_capture_callback)):
            self.create_service(Trigger, mname + '/' + name, callback, callback_group=self._service_cb_group)

        self.create_timer(2.0, self.preflight_timer_callback)
        self._publish_status(None, None)
        from .automatic_calibration import AutomaticCalibration
        self.automatic = AutomaticCalibration(self)

    def _publish_status(self, cal, last_metrics):
        diversity = self._diversity_summary()
        residuals = self._calibration_residuals(cal) if cal is not None else None
        status = build_calibration_status(
            sample_count=len(self.robot_samples),
            diversity=diversity,
            residuals=residuals,
            last_sample_metrics=last_metrics,
            estimate=cal,
            uncertainty=self._last_uncertainty,
            training_count=self.training_count(),
            min_save_samples=self.acceptance_limits['min_samples'],
            acceptance=self._last_acceptance,
            reprojection=self._reprojection_summary(),
            timing=self.latency.stats(),
            touchoff=self.touchoff_results[-5:],
        )
        if self.solver_name == 'intrinsic_pose':
            report = self._last_reprojection
            passed = bool(self._last_acceptance and self._last_acceptance['passed'])
            status.update(solver='intrinsic_pose', pose_calibration=self._reprojection_summary(),
                          reprojection=None, ready_to_save=passed,
                          readiness='excellent' if passed else 'collecting',
                          summary='Pose calibration accepted.' if passed else
                          'Collect diverse poses; joint solve and independent validation determine acceptance.')
        self._status_payload = status
        if hasattr(self, "automatic"):
            status = {**status, "automatic": dict(self.automatic.status)}
        msg = String()
        msg.data = status_to_json(status)
        self.status_pub.publish(msg)

    # ------------------------------------------------------------------
    # Raw observations, latency and sample bookkeeping
    # ------------------------------------------------------------------
    def _observation_callback(self, msg):
        observation = parse_observation(msg.data)
        if observation is None:
            return
        self.latency.add((self.get_clock().now().nanoseconds - observation['stamp_ns']) / 1e9)
        self.observations.add(observation)
        if observation.get('camera'):
            self._camera_model = observation['camera']
        if observation.get('board'):
            self._board_spec_seen = observation['board']

    def _image_callback(self, msg):
        self._latest_image = msg

    def freshness_limit(self, base_s):
        return self.latency.limit(base_s)

    def training_indices(self):
        return [i for i, role in enumerate(self.sample_roles) if role == 'training']

    def validation_indices(self):
        return [i for i, role in enumerate(self.sample_roles) if role == 'validation']

    def training_count(self):
        return len(self.training_indices())

    def clear_samples(self):
        for values in (self.robot_samples, self.tracking_samples, self.sample_metrics, self.sample_frames,
                       self.sample_joints, self.sample_roles, self.sample_images):
            values.clear()
        self._last_uncertainty = self._last_calibration_detail = None
        self._last_reprojection = self._last_acceptance = self._reprojection_state = None

    def drop_last_sample(self):
        for values in (self.robot_samples, self.tracking_samples, self.sample_metrics, self.sample_frames,
                       self.sample_joints, self.sample_roles, self.sample_images):
            if values:
                values.pop()
        self._last_uncertainty = None

    def _current_joints(self):
        joints = getattr(getattr(self, 'automatic', None), 'joints', None)
        if joints is None:
            return None
        return {n: float(p) for n, p in zip(joints.name, joints.position)}

    def _encode_image(self, stamps):
        msg = self._latest_image
        if msg is None:
            return None
        try:
            import cv2
            from cv_bridge import CvBridge
            frame = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
            ok, data = cv2.imencode('.png', frame)
            if not ok:
                return None
            stamp = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds
            return {'stamp_ns': stamp, 'png': data.tobytes(), 'matches_frame': stamp in stamps}
        except Exception as exc:  # noqa: BLE001 - images are optional evidence
            self.get_logger().warning(f'Could not store calibration image: {exc}')
            return None

    def _robot_was_stationary(self, stamp_msg, reference):
        """Robot pose `stationary_window_s` before the frame equals the pose at it."""
        if self.stationary_window_s <= 0:
            return True
        earlier = rclpy.time.Time.from_msg(stamp_msg) - Duration(seconds=self.stationary_window_s)
        try:
            before = get_transform(self._lookup_robot_at(earlier.to_msg(), timeout_s=0.1).transform)
        except TransformException:
            return False
        delta = np.linalg.inv(transform_to_matrix(before)) @ transform_to_matrix(reference)
        moved_m, moved_rad = matrix_to_residual(delta)
        return moved_m < 0.0005 and math.degrees(moved_rad) < 0.05

    def preflight_timer_callback(self):
        if self._preflight_logged:
            return
        if self.log_preflight():
            self._preflight_logged = True

    def _lookup_ok(self, target_frame, source_frame, label, lookup_time=None):
        if lookup_time is None:
            lookup_time = rclpy.time.Time()
        try:
            self.tf_buffer.lookup_transform(target_frame, source_frame, lookup_time, Duration(seconds=0.2))
            self.get_logger().info(f"Preflight {label}: {target_frame} -> {source_frame} OK")
            return True
        except TransformException as ex:
            self.get_logger().warning(f"Preflight {label}: missing {target_frame} -> {source_frame}: {ex}")
            return False

    def _read_pointcloud_frame(self):
        if not self.pointcloud_topic:
            return None
        ok, msg = wait_for_message(PointCloud2, self, self.pointcloud_topic, time_to_wait=0.5)
        if ok:
            self._last_pointcloud_frame = msg.header.frame_id
            return msg.header.frame_id
        return None

    def _read_camera_info_frame(self):
        if not self.camera_info_topic:
            return None
        ok, msg = wait_for_message(CameraInfo, self, self.camera_info_topic, time_to_wait=0.5)
        if ok:
            self._last_camera_info_frame = msg.header.frame_id
            return msg.header.frame_id
        return None

    def log_preflight(self):
        self.get_logger().info(
            "Preflight config: "
            f"calibration_type={self.calibration_type}, "
            f"robot={self.robot_base_frame}->{self.robot_effector_frame}, "
            f"tracking={self.tracking_base_frame}->{self.tracking_marker_frame}"
        )
        robot_ok = self._lookup_ok(self.robot_base_frame, self.robot_effector_frame, "robot")
        tracking_ok = self._lookup_ok(self.tracking_base_frame, self.tracking_marker_frame, "tracking")

        pointcloud_frame = self._read_pointcloud_frame()
        if pointcloud_frame:
            self.get_logger().info(f"Preflight pointcloud: {self.pointcloud_topic}.header.frame_id={pointcloud_frame}")
            if pointcloud_frame != self.tracking_base_frame:
                self.get_logger().warning(
                    f"Pointcloud frame '{pointcloud_frame}' differs from tracking_base_frame "
                    f"'{self.tracking_base_frame}'. For pointcloud calibration, these should usually match."
                )
        elif self.pointcloud_topic:
            self.get_logger().warning(f"Preflight pointcloud: no message received on {self.pointcloud_topic}")

        camera_info_frame = self._read_camera_info_frame()
        if camera_info_frame:
            self.get_logger().info(f"Preflight camera_info: {self.camera_info_topic}.header.frame_id={camera_info_frame}")
            if camera_info_frame != self.tracking_base_frame:
                self.get_logger().warning(
                    f"CameraInfo frame '{camera_info_frame}' differs from tracking_base_frame "
                    f"'{self.tracking_base_frame}'. The ChArUco detector should publish board TF from the same optical frame."
                )

        if self.marker_size > 0.0:
            self.get_logger().info(f"Preflight marker_size={self.marker_size:.4f} m")
        return robot_ok and tracking_ok

    def _sample_metrics(self, robot_tf, tracking_tf):
        """robot_tf / tracking_tf are already-extracted [tx,ty,tz,qx,qy,qz,qw] lists
        (e.g. the burst-averaged sample), not TransformStamped messages."""
        marker_distance = float(np.linalg.norm(tracking_tf[:3]))
        tracking_rot = Rot.from_quat(tracking_tf[3:]).as_matrix()
        marker_normal = tracking_rot[:, 2]
        normal_z = max(-1.0, min(1.0, abs(float(marker_normal[2]))))
        marker_angle_deg = float(math.degrees(math.acos(normal_z)))

        if self.robot_samples:
            last_robot = transform_to_matrix(self.robot_samples[-1])
            current_robot = transform_to_matrix(robot_tf)
            delta = np.linalg.inv(last_robot) @ current_robot
            robot_delta_m, robot_delta_rad = matrix_to_residual(delta)
        else:
            robot_delta_m, robot_delta_rad = None, None

        return {
            'marker_distance_m': marker_distance,
            'marker_view_angle_deg': marker_angle_deg,
            'robot_delta_translation_m': robot_delta_m,
            'robot_delta_rotation_deg': None if robot_delta_rad is None else float(math.degrees(robot_delta_rad)),
        }

    def _log_sample_quality(self, metrics):
        pieces = [
            f"marker_distance={metrics['marker_distance_m']:.3f}m",
            f"marker_view_angle={metrics['marker_view_angle_deg']:.1f}deg",
        ]
        if metrics['robot_delta_translation_m'] is not None:
            pieces.append(f"robot_delta_translation={metrics['robot_delta_translation_m']:.3f}m")
            pieces.append(f"robot_delta_rotation={metrics['robot_delta_rotation_deg']:.1f}deg")
        self.get_logger().info("Sample quality: " + ", ".join(pieces))

        if metrics['robot_delta_translation_m'] is not None:
            if metrics['robot_delta_translation_m'] < 0.015 and metrics['robot_delta_rotation_deg'] < 5.0:
                self.get_logger().warning(
                    "This sample is very close to the previous robot pose. Add more wrist rotation/translation diversity."
                )
        if metrics['marker_view_angle_deg'] > 70.0:
            self.get_logger().warning("Marker is viewed at a steep angle; pose estimate may be noisy.")
        if metrics['marker_distance_m'] < 0.15 or metrics['marker_distance_m'] > 1.5:
            self.get_logger().warning("Marker distance is outside the usual comfortable range for ArUco calibration.")

    def _diversity_summary(self):
        if len(self.robot_samples) < 2:
            return {
                'sample_count': len(self.robot_samples),
                'translation_span_m': [0.0, 0.0, 0.0],
                'max_rotation_from_first_deg': 0.0,
                'guidance': ['Need more samples from different wrist poses.'],
            }

        translations = np.array([s[:3] for s in self.robot_samples], dtype=float)
        span = (translations.max(axis=0) - translations.min(axis=0)).tolist()
        first_rot = Rot.from_quat(self.robot_samples[0][3:])
        rotation_deltas = [
            (first_rot.inv() * Rot.from_quat(sample[3:])).magnitude()
            for sample in self.robot_samples[1:]
        ]
        max_rot_deg = float(math.degrees(max(rotation_deltas)))
        guidance = []
        if max(span) < 0.05:
            guidance.append("Need more translation spread.")
        if max_rot_deg < 25.0:
            guidance.append("Need more wrist rotation variation, especially around multiple axes.")
        if len(self.robot_samples) < 10:
            guidance.append("More samples recommended; 10-20 diverse poses is a better target than the 4-sample minimum.")
        if not guidance:
            guidance.append("Pose diversity looks reasonable.")
        return {
            'sample_count': len(self.robot_samples),
            'translation_span_m': [float(v) for v in span],
            'max_rotation_from_first_deg': max_rot_deg,
            'guidance': guidance,
        }

    def _log_diversity(self):
        summary = self._diversity_summary()
        self.get_logger().info(
            "Sample diversity: "
            f"count={summary['sample_count']}, "
            f"translation_span_m={[round(v, 3) for v in summary['translation_span_m']]}, "
            f"max_rotation_from_first={summary['max_rotation_from_first_deg']:.1f}deg"
        )
        for item in summary['guidance']:
            self.get_logger().info("Diversity guidance: " + item)

    def _calibration_residuals(self, cal):
        if cal is None or len(self.robot_samples) < 2:
            return None
        # Prefer the residuals the fit itself reported. Those are computed over
        # the samples the fit actually used (outliers excluded), so they
        # describe the calibration being published; recomputing over every
        # sample would fold the rejected outliers back in and overstate the
        # error the user is being shown.
        detail = self._last_calibration_detail
        if detail is not None and detail.get('transform') == cal and detail.get('residuals'):
            residuals = dict(detail['residuals'])
            residuals['samples_used'] = len(detail.get('kept_indices') or [])
            residuals['samples_rejected'] = len(detail.get('rejected_indices') or [])
            return residuals
        # Fallback (e.g. a calibration loaded from elsewhere): AX=XB residual
        # over ALL sample pairs, not just consecutive ones.
        return CalibrationBackend.pairwise_residuals(self.robot_samples, self.tracking_samples, cal)

    def _log_residuals(self, cal):
        residuals = self._calibration_residuals(cal)
        if residuals is None:
            return
        self.get_logger().info(
            "Calibration residuals: "
            f"mean_translation={residuals['mean_translation_m']:.4f}m, "
            f"max_translation={residuals['max_translation_m']:.4f}m, "
            f"mean_rotation={residuals['mean_rotation_deg']:.2f}deg, "
            f"max_rotation={residuals['max_rotation_deg']:.2f}deg"
        )

    def _lookup_robot_at(self, lookup_time, timeout_s):
        # For eye-on-base ("eye-to-hand" in OpenCV's terminology, static camera
        # observing a marker on the moving end-effector) we look up the
        # INVERSE of the usual forward-kinematics transform (effector<-base
        # instead of base<-effector). This is not a hack: it is exactly the
        # R_gripper2base convention cv2.calibrateHandEye's own documentation
        # specifies for the eye-to-hand case, so the same solver produces the
        # correct base->camera result without any special-casing downstream.
        if self.calibration_type == "eye-in-hand":
            return self.tf_buffer.lookup_transform(
                self.robot_base_frame, self.robot_effector_frame, lookup_time,
                Duration(seconds=timeout_s))
        elif self.calibration_type == "eye-on-base":
            return self.tf_buffer.lookup_transform(
                self.robot_effector_frame, self.robot_base_frame, lookup_time,
                Duration(seconds=timeout_s))
        raise ValueError(
            "Invalid calibration_type: " + self.calibration_type + ". Options are eye-in-hand or eye-on-base")

    def capture_point_service_callback(self, req: Trigger.Request, resp: Trigger.Response):
        if (hasattr(self, 'automatic') and self.automatic.active
                and threading.current_thread() is not self.automatic.thread):
            resp.success = False
            resp.message = 'Automatic calibration is running; use Stop first.'
            return resp
        self.log_preflight()

        if self.calibration_type not in ("eye-in-hand", "eye-on-base"):
            msg = "Invalid calibration_type: " + self.calibration_type + ". Options are eye-in-hand or eye-on-base"
            self.get_logger().error(msg)
            resp.success = False
            resp.message = msg
            return resp

        # Gather a short burst of (robot, tracking) pairs. Each tracking sample
        # is looked up at Time() ("latest") so it is always an actual detector
        # broadcast (never an interpolation across a gap of rejected/bad
        # frames); the robot sample is then looked up AT THAT EXACT STAMP
        # instead of a guessed "now - 1s" offset, so the two halves of the
        # sample are properly time-synchronized.
        #
        # We only sleep here — we must NOT pump the executor ourselves. This
        # callback is already being run by the executor, and re-entering it
        # (e.g. rclpy.spin_once) raises "Executor is already spinning" and
        # aborts the whole capture. The TF buffer is instead kept fresh by the
        # listener running on another executor thread; see _service_cb_group.
        max_age = self.freshness_limit(1.0)
        burst_deadline = time.monotonic() + max(1.5, self.capture_burst_duration_s) + max(0.0, max_age - 1.0)
        # Automatic mode: only frames exposed after the measured settle time.
        min_stamp_ns = getattr(getattr(self, 'automatic', None), 'min_capture_stamp_ns', None) \
            if hasattr(self, 'automatic') and self.automatic.active else None
        seen_stamps = set()
        tracking_burst = []
        robot_burst = []
        frames = []
        moving_frames = 0

        while len(tracking_burst) < self.capture_burst_samples and time.monotonic() < burst_deadline:
            if hasattr(self, 'automatic') and self.automatic.active:
                self.automatic.check()
            time.sleep(0.02)
            try:
                tracking_k = self.tf_buffer.lookup_transform(
                    self.tracking_base_frame, self.tracking_marker_frame, rclpy.time.Time())
            except TransformException:
                continue
            stamp_ns = rclpy.time.Time.from_msg(tracking_k.header.stamp).nanoseconds
            age = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
            if not -0.1 <= age < max_age:
                continue
            if min_stamp_ns is not None and stamp_ns < min_stamp_ns:
                continue  # exposed before the robot had settled
            stamp_key = (tracking_k.header.stamp.sec, tracking_k.header.stamp.nanosec)
            if stamp_key in seen_stamps:
                continue  # no new detector frame published yet
            seen_stamps.add(stamp_key)
            try:
                robot_k = self._lookup_robot_at(tracking_k.header.stamp, timeout_s=0.3)
            except TransformException:
                continue
            robot_pose = get_transform(robot_k.transform)
            if not self._robot_was_stationary(tracking_k.header.stamp, robot_pose):
                moving_frames += 1
                continue
            tracking_burst.append(tracking_k)
            robot_burst.append(robot_k)
            observation = self.observations.get(stamp_ns)
            if observation is not None:
                frames.append(frame_record(observation, robot_pose, stamp_ns))

        if len(tracking_burst) < 3:
            resp.success = False
            resp.message = ('Need at least 3 fresh, time-synchronized board frames taken while the robot was still'
                            f' (max stamp age {max_age:.2f} s, {moving_frames} frame(s) during motion). '
                            'Hold the robot still; check detection, camera latency and use_sim_time.')
            return resp

        robot_list = [get_transform(t.transform) for t in robot_burst]
        tracking_list = [get_transform(t.transform) for t in tracking_burst]

        robot_avg, robot_spread = CalibrationBackend.average_transforms(robot_list)
        tracking_avg, tracking_spread = CalibrationBackend.average_transforms(tracking_list)

        if robot_spread['max_translation_dev_m'] > 0.003 or robot_spread['max_rotation_dev_deg'] > 0.3:
            resp.success = False
            resp.message = 'Robot moved during capture; sample rejected.'
            return resp

        dropped = tracking_spread['rejected_frames'] + robot_spread['rejected_frames']
        self.get_logger().info(
            f"Captured {len(tracking_burst)} time-synced frame(s) for this sample"
            + (f", {dropped} outlier frame(s) dropped before averaging." if dropped else ".")
        )
        self.get_logger().info("robot (avg): " + tf_list_to_string(robot_avg))
        self.get_logger().info("tracking (avg): " + tf_list_to_string(tracking_avg))

        metrics = self._sample_metrics(robot_avg, tracking_avg)
        metrics['burst_frame_count'] = tracking_spread['count']
        metrics['burst_frames_rejected'] = tracking_spread['rejected_frames'] + robot_spread['rejected_frames']
        metrics['burst_tracking_translation_dev_m'] = tracking_spread['max_translation_dev_m']
        metrics['burst_tracking_rotation_dev_deg'] = tracking_spread['max_rotation_dev_deg']
        metrics['burst_robot_translation_dev_m'] = robot_spread['max_translation_dev_m']
        metrics['burst_robot_rotation_dev_deg'] = robot_spread['max_rotation_dev_deg']
        self._log_sample_quality(metrics)

        metrics['raw_frames'] = len(frames)
        metrics['camera_latency'] = self.latency.stats()
        self.robot_samples.append(robot_avg)
        self.tracking_samples.append(tracking_avg)
        self.sample_metrics.append(metrics)
        self.sample_frames.append(frames)
        self.sample_joints.append(self._current_joints())
        role = getattr(getattr(self, 'automatic', None), 'capture_role', 'training') \
            if hasattr(self, 'automatic') and self.automatic.active else 'training'
        self.sample_roles.append(role)
        self.sample_images.append(self._encode_image({f['stamp_ns'] for f in frames})
                                  if bool(self.get_parameter('dataset_save_images').value) else None)
        if not frames:
            self.get_logger().warning(
                'No raw ChArUco observations matched this sample; is the detector publishing '
                f"{self.get_parameter('observation_topic').value}? The reprojection solver will skip it.")
        # The cached uncertainty described the previous sample set; it is stale
        # the moment a new sample lands. Re-estimating here would add seconds to
        # every capture, so drop it and let save/estimate_uncertainty redo it.
        self._last_uncertainty = None
        self._log_diversity()

        cal = None if self.automatic.active else self.get_calibration()
        if cal is None:
            msg = "Sample captured." if self.automatic.active else "Not enough samples yet..."
        else:
            self.get_logger().info("Current estimate of: " + self.tracking_base_frame + " -> " + self.robot_effector_frame)
            self.get_logger().info("transform: " + tf_list_to_string(cal))
            self.get_logger().info("as euler: " + urdf_list_to_string(tf_to_urdf_tf(cal)))
            self._log_residuals(cal)
            msg = "Current estimate: " + tf_list_to_string(cal) + " as euler: " + urdf_list_to_string(tf_to_urdf_tf(cal))
        self._publish_status(cal, metrics)
        resp.success = True
        resp.message = msg
        return resp

    def _training_samples(self):
        idx = self.training_indices()
        return [self.robot_samples[i] for i in idx], [self.tracking_samples[i] for i in idx]

    def _dataset_samples(self, indices):
        return [{'robot': self.robot_samples[i], 'tracking': self.tracking_samples[i],
                 'frames': self.sample_frames[i], 'joints': self.sample_joints[i]} for i in indices]

    def _camera(self):
        from .reprojection_calibration import Camera
        if self._camera_model is None:
            return None
        return Camera.from_dict(self._camera_model)

    def _reprojection_summary(self):
        report = self._last_reprojection
        if not report:
            return None
        keys = ('board_in_base', 'pose_metrics', 'geometry', 'algorithm', 'optimizer', 'upstream_commit', 'solver', 'reprojection_rms_px', 'training_views', 'rejected_views', 'board_spread',
                'leave_one_out', 'validation_views', 'initial_delta_translation_m',
                'initial_delta_rotation_deg', 'intrinsics', 'refinement_comparison',
                'axxb_rejected_views', 'reprojection_rejected_views', 'kept_views')
        summary = {k: report.get(k) for k in keys if report.get(k) is not None}
        for key in ('board_spread', 'leave_one_out', 'validation_views'):
            if isinstance(summary.get(key), dict):
                summary[key] = {k: v for k, v in summary[key].items() if not k.startswith('per_view')}
        return summary

    def _solve_reprojection(self, closed_form, full=False, rejected_indices=()):
        """Final estimate from corner pixels. `full` adds leave-one-out,
        validation views and bootstrap (seconds); otherwise only the fit."""
        from .reprojection_calibration import Dataset, calibrate, pose_matrix
        if self.solver_name != 'reprojection':
            return None
        camera = self._camera()
        training, validation = self.training_indices(), self.validation_indices()
        if camera is None or any(not self.sample_frames[i] for i in training):
            return None
        order = training + validation
        dataset = Dataset(self._dataset_samples(order), camera, self.calibration_type)
        if len(dataset) != len(order):
            return None
        report = calibrate(dataset, pose_matrix(closed_form), estimate_intrinsics=self.estimate_intrinsics,
                           validation_indices=list(range(len(training), len(order))),
                           bootstrap_samples=self.bootstrap_samples if full else 0, leave_one_out=full,
                           excluded_indices=rejected_indices, compare_refinement_cv=full)
        solution = report.pop('_solution')
        # Map dataset indices back to sample indices for reporting.
        for key in ('rejected_views', 'axxb_rejected_views', 'reprojection_rejected_views', 'kept_views'):
            report[key] = [order[i] for i in report[key]]
        for fold in (report.get('refinement_comparison') or {}).get('folds', []):
            fold['held_out'] = order[fold['held_out']]
            fold['fit_indices'] = [order[i] for i in fold['fit_indices']]
        report['training_sample_indices'] = training
        report['validation_sample_indices'] = validation
        report['closed_form_transform'] = list(closed_form)
        self._reprojection_state = (dataset, solution)
        return report

    def _get_intrinsic_calibration(self, full=False):
        from . import intrinsic_solver as solver
        robot, tracking = self._training_samples()
        if len(robot) < 3:
            return None
        # Cache only training inputs: held-out frames cannot influence the fit.
        key = (tuple(map(tuple, robot)), tuple(map(tuple, tracking)))
        cached = getattr(self, '_intrinsic_fit_cache', None)
        try:
            report = dict(cached[1]) if cached is not None and cached[0] == key else solver.solve(robot, tracking)
            if full and 'uncertainty' not in report:
                report['uncertainty'] = solver.uncertainty(robot, tracking, report, self.bootstrap_samples)
            self._intrinsic_fit_cache = (key, dict(report))
            ids = self.validation_indices()
            if ids:
                report['validation_views'] = solver.metrics(
                    np.array([solver.matrix(self.robot_samples[i]) for i in ids]),
                    np.array([solver.matrix(self.tracking_samples[i]) for i in ids]),
                    solver.matrix(report['transform']), solver.matrix(report['board_in_base']))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            self._last_reprojection = self._last_calibration_detail = None
            self.get_logger().error(f'Intrinsic pose solve failed: {exc}')
            return None
        self._last_reprojection = report
        self._last_acceptance = solver.evaluate(report, require_validation=bool(getattr(getattr(self, 'automatic', None), 'active', False)))
        self._reprojection_state = None
        self._last_uncertainty = report.get('uncertainty')
        self._last_calibration_detail = dict(transform=report['transform'], algorithm_used='SHAH+NONLINEAR',
                                             kept_indices=list(range(len(robot))), rejected_indices=[],
                                             refinement={'optimizer':'scipy_least_squares'})
        return report['transform']

    def get_calibration(self, full=False):
        if self.solver_name == 'intrinsic_pose':
            return self._get_intrinsic_calibration(full)
        robot, tracking = self._training_samples()
        if len(robot) < 4:
            self.get_logger().info("Not enough samples yet...")
            return None

        self.get_logger().info("Estimating ...")
        try:
            detail = CalibrationBackend.compute_calibration_detailed(
                samples_robot=robot, samples_tracking=tracking)
        except (RuntimeError, ValueError) as exc:
            # The backend translates OpenCV's cv2.error into RuntimeError, so
            # a degenerate pose set surfaces here as a failed service call
            # rather than as an exception escaping into the executor.
            self.get_logger().error(f"Calibration failed: {exc}")
            self._last_calibration_detail = None
            return None

        self._last_reprojection = None
        try:
            report = self._solve_reprojection(detail['transform'], full=full,
                                              rejected_indices=detail['rejected_indices'])
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            self.get_logger().error(f"Reprojection solve failed, keeping the closed-form estimate: {exc}")
            report = None
        if report is not None:
            self._last_reprojection = report
            detail = dict(detail)
            detail['axxb_transform'] = detail['transform']
            detail['transform'] = report['transform']
            # Report residuals on the final survivor set, not the earlier
            # AX=XB set if pixel refinement rejected more views.
            global_to_training = {v: i for i, v in enumerate(report['training_sample_indices'])}
            detail['kept_indices'] = [global_to_training[i] for i in report['kept_views']]
            detail['rejected_indices'] = [global_to_training[i] for i in report['rejected_views']]
            detail['residuals'] = CalibrationBackend.pairwise_residuals(
                robot, tracking, report['transform'], indices=detail['kept_indices'])
            self.get_logger().info(
                f"Reprojection solve: RMS {report['reprojection_rms_px']:.2f} px over {report['training_views']} views, "
                f"moved the AX=XB estimate by {report['initial_delta_translation_m'] * 1000:.2f} mm / "
                f"{report['initial_delta_rotation_deg']:.3f} deg; board spread "
                f"{report['board_spread']['position_rms_m'] * 1000:.2f} mm RMS")
        self._last_calibration_detail = detail
        if detail['rejected_indices']:
            self.get_logger().warning(
                f"Rejected {len(detail['rejected_indices'])} outlier sample(s) "
                f"(indices {detail['rejected_indices']}) as inconsistent with the rest before fitting."
            )
        refinement = detail['refinement']
        self.get_logger().info(
            f"Hand-eye algorithm: {detail['algorithm_used']} (selected by leave-one-pose-out validation of "
            "Tsai/Park/Horaud/Andreff/Daniilidis); nonlinear refinement moved the estimate by "
            f"{refinement['delta_translation_m'] * 1000:.2f}mm / {refinement['delta_rotation_deg']:.3f}deg"
        )
        return detail['transform']

    def _compute_uncertainty(self, cal):
        """Bootstrap the calibration uncertainty for the current sample set.

        Returns None when disabled or when there is not enough data. Slow by
        design (a full refit per resample), so callers decide when to pay it.
        """
        if self.solver_name == 'intrinsic_pose':
            if cal is None:
                return None
            self._get_intrinsic_calibration(full=True)
            return self._last_uncertainty
        if self.bootstrap_samples <= 0:
            return None
        detail = self._last_calibration_detail
        if cal is None or detail is None:
            return None

        robot, tracking = self._training_samples()
        cache_key = (tuple(cal), detail['algorithm_used'], self.bootstrap_samples,
                     tuple(map(tuple, robot)), tuple(map(tuple, tracking)))
        cached = getattr(self, '_uncertainty_cache', None)
        if cached is not None and cached[0] == cache_key:
            self._last_uncertainty = cached[1]
            return cached[1]
        self.get_logger().info(
            f"Estimating calibration uncertainty ({self.bootstrap_samples} bootstrap resamples), "
            "this takes a few seconds..."
        )
        try:
            state = self._reprojection_state
            if (self._last_reprojection is not None and state is not None
                    and list(cal) == list(self._last_reprojection['transform'])):
                from .reprojection_calibration import ReprojectionSolver
                solver = ReprojectionSolver(state[0], estimate_intrinsics=self.estimate_intrinsics)
                uncertainty = solver.bootstrap(state[1], n=self.bootstrap_samples)
                if uncertainty is not None:
                    uncertainty['guidance'] = CalibrationBackend._uncertainty_guidance(
                        uncertainty['worst_direction_sigma_m'], uncertainty['best_direction_sigma_m'],
                        np.asarray(uncertainty['worst_direction_axis']))
            else:
                uncertainty = CalibrationBackend.bootstrap_uncertainty(
                    samples_robot=robot,
                    samples_tracking=tracking,
                    nominal_transform=cal,
                    algorithm=detail['algorithm_used'],
                    n_bootstrap=self.bootstrap_samples,
                )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            self.get_logger().warning(f"Uncertainty estimation failed: {exc}")
            return None

        if uncertainty is None:
            self.get_logger().warning(
                "Uncertainty estimation did not converge on enough resamples "
                "(too few or too similar samples)."
            )
            return None

        sig = uncertainty['translation_sigma_m']
        rot_sig = uncertainty['rotation_sigma_deg']
        self.get_logger().info(
            "Calibration uncertainty (1 sigma, %d resamples): "
            "translation +/- [%.2f, %.2f, %.2f] mm, rotation +/- [%.3f, %.3f, %.3f] deg"
            % (uncertainty['n_bootstrap'], sig[0] * 1000, sig[1] * 1000, sig[2] * 1000,
               rot_sig[0], rot_sig[1], rot_sig[2])
        )
        self.get_logger().info("Uncertainty guidance: " + uncertainty['guidance'])
        self._last_uncertainty = uncertainty
        self._uncertainty_cache = (cache_key, uncertainty)
        return uncertainty

    def estimate_uncertainty_service_callback(self, req: Trigger.Request, resp: Trigger.Response):
        if (hasattr(self, 'automatic') and self.automatic.active
                and threading.current_thread() is not self.automatic.thread):
            resp.success = False
            resp.message = 'Automatic calibration is running; use Stop first.'
            return resp
        cal = self.get_calibration()
        if cal is None:
            resp.success = False
            resp.message = f"Not enough samples (need at least {CalibrationBackend.MIN_SAMPLES})."
            return resp

        uncertainty = self._compute_uncertainty(cal)
        if uncertainty is None:
            resp.success = False
            resp.message = (
                "Could not estimate uncertainty (disabled via bootstrap_samples, "
                "or not enough distinct samples)."
            )
            self._publish_status(cal, self.sample_metrics[-1] if self.sample_metrics else None)
            return resp

        sig = uncertainty['translation_sigma_m']
        resp.success = True
        resp.message = (
            "1 sigma translation +/- [%.2f, %.2f, %.2f] mm (worst direction %.2f mm). %s"
            % (sig[0] * 1000, sig[1] * 1000, sig[2] * 1000,
               uncertainty['worst_direction_sigma_m'] * 1000, uncertainty['guidance'])
        )
        self._publish_status(cal, self.sample_metrics[-1] if self.sample_metrics else None)
        return resp

    def save_calibration_service_callback(self, req: Trigger.Request, resp: Trigger.Response):
        if (hasattr(self, 'automatic') and self.automatic.active
                and threading.current_thread() is not self.automatic.thread):
            resp.success = False
            resp.message = 'Automatic calibration is running; use Stop first.'
            return resp
        """Save current calibration estimate to YAML file for later publishing.

        The raw dataset is always written. The active calibration file is
        replaced only when the acceptance gate passes (or acceptance_mode is
        'warn'); otherwise the candidate goes to <calibration_file>.rejected.yaml.
        """
        cal = self.get_calibration(full=True)
        if cal is None:
            resp.success = False
            resp.message = "Not enough samples (need at least 4). Capture more points first."
            return resp
        cal_file = os.path.expanduser(str(self.get_parameter('calibration_file').value))
        try:
            residuals = self._calibration_residuals(cal)
            diversity = self._diversity_summary()
            # Record how well-determined the saved numbers actually are, so the
            # YAML carries its own error bars rather than a bare transform.
            uncertainty = self._compute_uncertainty(cal)
            report = self._last_reprojection
            if report is not None:
                report['uncertainty'] = uncertainty
            limits = dict(self.acceptance_limits)
            if self.automatic.active:
                limits['max_position_sigma_m'] = self.automatic.max_position_sigma
            if self.solver_name == 'intrinsic_pose':
                from .intrinsic_solver import evaluate
                verdict = evaluate(report, sigma_limit=limits['max_position_sigma_m'],
                                   min_samples=max(20, limits['min_samples']),
                                   require_validation=self.automatic.active)
            else:
                verdict = acceptance_gate.evaluate(report, self.training_count(), limits)
            if report is None and self.solver_name == 'axxb':
                # Explicit legacy mode: judge on sample count and bootstrap only.
                verdict = acceptance_gate.evaluate(
                    {'uncertainty': uncertainty, 'board_spread': {'position_rms_m': 0.0},
                     'leave_one_out': {'position_rms_m': 0.0, 'reprojection_rms_px': 0.0}},
                    self.training_count(), limits)
                verdict['summary'] += ' (legacy AX=XB mode: no held-out validation)'
            self._last_acceptance = verdict
            detail = self._last_calibration_detail or {}
            data = {
                'calibration_type': self.calibration_type,
                'tracking_base_frame': self.tracking_base_frame,
                'tracking_marker_frame': self.tracking_marker_frame,
                'robot_base_frame': self.robot_base_frame,
                'robot_effector_frame': self.robot_effector_frame,
                'calibrated_child_frame': self.tracking_base_frame,
                'pointcloud_topic': self.pointcloud_topic,
                'pointcloud_frame': self._last_pointcloud_frame,
                'image_topic': self.image_topic,
                'camera_info_topic': self.camera_info_topic,
                'camera_info_frame': self._last_camera_info_frame,
                'marker_size_m': self.marker_size if self.marker_size > 0.0 else None,
                'sample_count': len(self.robot_samples),
                'training_sample_count': self.training_count(),
                'validation_sample_indices': self.validation_indices(),
                'sample_metrics': self.sample_metrics,
                'diversity': diversity,
                'residuals': residuals,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'transform': {
                    'tx': cal[0], 'ty': cal[1], 'tz': cal[2],
                    'qx': cal[3], 'qy': cal[4], 'qz': cal[5], 'qw': cal[6],
                },
                'solver': self.solver_name,
                'algorithm_used': detail.get('algorithm_used'),
                'algorithm_selection': detail.get('algorithm_selection'),
                'axxb_transform': None if self.solver_name == 'intrinsic_pose' else detail.get('axxb_transform', detail.get('transform')),
                'rejected_sample_indices': [self.training_indices()[i] for i in detail.get('rejected_indices', [])],
                'nonlinear_refinement': detail.get('refinement'),
                'reprojection': self._reprojection_summary() if self.solver_name != 'intrinsic_pose' else None,
                'pose_calibration': self._reprojection_summary() if self.solver_name == 'intrinsic_pose' else None,
                'uncertainty': uncertainty,
                'acceptance': verdict,
                'camera_latency': self.latency.stats(),
                'camera_model': self._camera_model,
                'board_spec': self._board_spec_seen,
            }
            if self.automatic.active:
                self.automatic.check()
                data['automatic_validation'] = self.automatic.status.get('validation')
                data['automatic_fit_consistency'] = self.automatic.status.get('fit_consistency')
                data['automatic_sample_plan'] = {
                    'initial': self.automatic.status.get('initial_samples', 0),
                    'targeted': self.automatic.status.get('targeted_samples', 0),
                    'independent_validation': len(self.validation_indices())}
                data['automatic_position_sigma_limit_m'] = self.automatic.max_position_sigma
            run_dir = None
            try:
                run_dir = calibration_dataset.write_run(
                    os.path.expanduser(str(self.get_parameter('dataset_dir').value)), self, data)
                data['dataset_path'] = run_dir
            except OSError as exc:
                self.get_logger().error(f'Could not write the calibration dataset: {exc}')
            accepted = verdict['passed'] or self.acceptance_mode == 'warn'
            target = cal_file if accepted else cal_file + '.rejected.yaml'
            self._write_yaml_atomic(target, data, keep_previous=accepted)
            if run_dir:
                self._write_yaml_atomic(os.path.join(run_dir, 'calibration.yaml'), data, keep_previous=False)
            self._publish_status(cal, self.sample_metrics[-1] if self.sample_metrics else None)
            if not accepted:
                self.get_logger().error(f"Calibration NOT saved as active: {verdict['summary']} Candidate: {target}")
                resp.success = False
                resp.message = f"Acceptance failed; active calibration unchanged. {verdict['summary']} Candidate: {target}"
                return resp
            note = '' if verdict['passed'] else f" WARNING (acceptance_mode=warn): {verdict['summary']}"
            self.get_logger().info("Calibration saved to %s%s" % (cal_file, note))
            resp.success = True
            resp.message = "Saved to " + cal_file + note + (f" (dataset {run_dir})" if run_dir else '')
        except Exception as e:
            self.get_logger().error("Failed to save calibration: %s" % str(e))
            resp.success = False
            resp.message = str(e)
        return resp

    # ------------------------------------------------------------------
    # Touch-off validation (independent of the calibration's own data)
    # ------------------------------------------------------------------
    def _latest_observation(self, max_age_s):
        items = list(self.observations.items.values())
        if not items:
            return None
        latest = items[-1]
        age = (self.get_clock().now().nanoseconds - latest['stamp_ns']) / 1e9
        return latest if -0.1 <= age <= max_age_s else None

    def touchoff_reference_callback(self, req, resp):
        """With the robot still and the board detected, record where the board
        is in the robot base: (a) through the calibration currently in TF
        (URDF/applied) and (b) through the in-memory candidate, if any."""
        if hasattr(self, 'automatic') and self.automatic.active:
            resp.success, resp.message = False, 'Automatic calibration is running; use Stop first.'
            return resp
        if self.calibration_type != 'eye-in-hand':
            resp.success, resp.message = False, 'Touch-off is implemented for eye-in-hand only.'
            return resp
        observation = self._latest_observation(self.freshness_limit(1.0))
        if observation is None:
            resp.success, resp.message = False, 'No fresh ChArUco observation; the board must be detected.'
            return resp
        stamp = rclpy.time.Time(nanoseconds=observation['stamp_ns']).to_msg()
        try:
            robot = get_transform(self._lookup_robot_at(stamp, timeout_s=0.3).transform)
        except TransformException as exc:
            resp.success, resp.message = False, f'Robot pose at the image stamp is unavailable: {exc}'
            return resp
        if not self._robot_was_stationary(stamp, robot):
            resp.success, resp.message = False, 'Robot was moving when the image was taken; hold still and retry.'
            return resp
        camera_board = transform_to_matrix(observation['pose'])
        corners = {int(i): p for i, p in zip(observation['ids'], observation['object_points'])}
        references = {}
        try:
            applied = self.tf_buffer.lookup_transform(self.robot_base_frame, self.tracking_base_frame, stamp,
                                                      Duration(seconds=0.3))
            references['applied_tf'] = transform_to_matrix(get_transform(applied.transform)) @ camera_board
        except TransformException:
            pass
        detail = self._last_calibration_detail
        if detail and detail.get('transform'):
            references['candidate'] = (transform_to_matrix(robot) @ transform_to_matrix(detail['transform'])
                                       @ camera_board)
        if not references:
            resp.success, resp.message = False, 'Neither a camera TF nor an in-memory calibration is available.'
            return resp
        self.touchoff_references = {'boards': references, 'corners': corners, 'stamp_ns': observation['stamp_ns']}
        resp.success = True
        resp.message = (f"Board reference stored from {', '.join(references)} with {len(corners)} corners. "
                        f"Now jog {self.get_parameter('touchoff_tcp_frame').value} onto corner "
                        f"{self.get_parameter('touchoff_corner_id').value} and call touchoff_capture.")
        return resp

    def touchoff_capture_callback(self, req, resp):
        """Compare the physical TCP position on a board corner with where the
        camera said that corner is. The error contains hand-eye, TCP and robot
        kinematic error together, which is what a seam actually experiences."""
        if hasattr(self, 'automatic') and self.automatic.active:
            resp.success, resp.message = False, 'Automatic calibration is running; use Stop first.'
            return resp
        ref = self.touchoff_references
        if not ref:
            resp.success, resp.message = False, 'Call touchoff_reference first (board visible, robot still).'
            return resp
        corner_id = int(self.get_parameter('touchoff_corner_id').value)
        if corner_id not in ref['corners']:
            resp.success, resp.message = False, (f'Corner {corner_id} was not detected in the reference view; '
                                                 f"detected ids: {sorted(ref['corners'])[:20]}...")
            return resp
        tcp_frame = str(self.get_parameter('touchoff_tcp_frame').value)
        try:
            first = self.tf_buffer.lookup_transform(self.robot_base_frame, tcp_frame, rclpy.time.Time())
            time.sleep(0.3)
            second = self.tf_buffer.lookup_transform(self.robot_base_frame, tcp_frame, rclpy.time.Time())
        except TransformException as exc:
            resp.success, resp.message = False, f'TCP frame {tcp_frame} unavailable: {exc}'
            return resp
        p1 = np.array(get_transform(first.transform)[:3])
        tip = np.array(get_transform(second.transform)[:3])
        if np.linalg.norm(tip - p1) > 0.0005:
            resp.success, resp.message = False, 'Robot is moving; hold the tip on the corner and retry.'
            return resp
        corner = np.r_[np.asarray(ref['corners'][corner_id], dtype=float), 1.0]
        result = {'timestamp': datetime.now(timezone.utc).isoformat(), 'corner_id': corner_id,
                  'tcp_frame': tcp_frame, 'tcp_base': [float(v) for v in tip], 'errors': {}}
        parts = []
        for name, board in ref['boards'].items():
            predicted = (board @ corner)[:3]
            error = tip - predicted
            result['errors'][name] = {'vector_m': [float(v) for v in error], 'norm_m': float(np.linalg.norm(error)),
                                      'predicted_base': [float(v) for v in predicted]}
            parts.append(f"{name}: {np.linalg.norm(error) * 1000:.1f} mm "
                         f"(dx {error[0] * 1000:+.1f}, dy {error[1] * 1000:+.1f}, dz {error[2] * 1000:+.1f})")
        self.touchoff_results.append(result)
        try:
            directory = os.path.join(os.path.expanduser(str(self.get_parameter('dataset_dir').value)), 'touchoff')
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, 'touchoff_log.yaml'), 'a') as f:
                yaml.safe_dump([calibration_dataset.plain(result)], f, sort_keys=False)
        except OSError as exc:
            self.get_logger().warning(f'Could not log touch-off result: {exc}')
        self._publish_status(None, None)
        message = f'Touch-off corner {corner_id}: ' + '; '.join(parts)
        self.get_logger().info(message)
        resp.success, resp.message = True, message
        return resp

    def write_failed_dataset(self):
        return calibration_dataset.write_run(
            os.path.expanduser(str(self.get_parameter('dataset_dir').value)), self, None)

    def _write_yaml_atomic(self, path, data, keep_previous):
        import tempfile
        import shutil
        directory = os.path.dirname(os.path.abspath(path)) or '.'
        os.makedirs(directory, exist_ok=True)
        if keep_previous and os.path.exists(path):
            shutil.copy2(path, path + '.previous')
        with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False) as f:
            yaml.safe_dump(calibration_dataset.plain(data), f, default_flow_style=False, sort_keys=False)
            temporary = f.name
        try:
            if self.automatic.active:
                self.automatic.check()
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

def main():
    import signal
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = DataCollector()
    shutdown_requested = threading.Event()

    def request_shutdown(signum, frame):
        node.automatic.stop(None, Trigger.Response())
        shutdown_requested.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    # MultiThreadedExecutor is required, not just nice to have: the capture
    # callback blocks while collecting its burst, and the TF listener has to
    # keep processing /tf on another thread for that burst to see fresh data.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok() and not shutdown_requested.is_set():
            executor.spin_once(timeout_sec=0.1)
        # Keep processing action responses while cancellation is acknowledged.
        deadline = time.monotonic() + 1.0
        while rclpy.ok() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.automatic.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
