# !/usr/bin/env python3
"""
Collect poses and perform calibration
"""

import math
import os
import time
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
        # options are eye-in-hand or eye-on-base
        self.declare_parameter('calibration_type', "eye-on-base")
        self.declare_parameter('calibration_file', os.path.expanduser("~/.ros/hand_eye_calibration.yaml"))
        self.declare_parameter('pointcloud_topic', "/oak/rgbd/points")
        self.declare_parameter('image_topic', "")
        self.declare_parameter('camera_info_topic', "")
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

        self.tracking_base_frame = str(self.get_parameter('tracking_base_frame').value)
        self.tracking_marker_frame = str(self.get_parameter('tracking_marker_frame').value)
        self.robot_base_frame = str(self.get_parameter('robot_base_frame').value)
        self.robot_effector_frame = str(self.get_parameter('robot_effector_frame').value)
        self.calibration_type = str(self.get_parameter('calibration_type').value)
        self.pointcloud_topic = str(self.get_parameter('pointcloud_topic').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.marker_size = float(self.get_parameter('marker_size').value)
        self.capture_burst_duration_s = float(self.get_parameter('capture_burst_duration_s').value)
        self.capture_burst_samples = int(self.get_parameter('capture_burst_samples').value)
        self.bootstrap_samples = int(self.get_parameter('bootstrap_samples').value)

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
        self._preflight_logged = False
        self._last_pointcloud_frame = None
        self._last_camera_info_frame = None
        self._last_calibration_detail = None
        self._last_uncertainty = None

        self.create_timer(2.0, self.preflight_timer_callback)
        self._publish_status(None, None)

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
        )
        msg = String()
        msg.data = status_to_json(status)
        self.status_pub.publish(msg)

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
        burst_deadline = self.get_clock().now() + Duration(seconds=self.capture_burst_duration_s)
        seen_stamps = set()
        tracking_burst = []
        robot_burst = []

        while len(tracking_burst) < self.capture_burst_samples and self.get_clock().now() < burst_deadline:
            time.sleep(0.02)
            try:
                tracking_k = self.tf_buffer.lookup_transform(
                    self.tracking_base_frame, self.tracking_marker_frame, rclpy.time.Time())
            except TransformException:
                continue
            stamp_key = (tracking_k.header.stamp.sec, tracking_k.header.stamp.nanosec)
            if stamp_key in seen_stamps:
                continue  # no new detector frame published yet
            seen_stamps.add(stamp_key)
            try:
                robot_k = self._lookup_robot_at(tracking_k.header.stamp, timeout_s=0.3)
            except TransformException:
                continue
            tracking_burst.append(tracking_k)
            robot_burst.append(robot_k)

        if not tracking_burst:
            # Nothing landed during the burst window — fall back to a single
            # generous-timeout lookup so the diagnostic messages below (frame
            # never published, detector not running, ...) still fire.
            try:
                tracking = self.tf_buffer.lookup_transform(
                    self.tracking_base_frame, self.tracking_marker_frame,
                    rclpy.time.Time(), Duration(seconds=2))
                robot = self._lookup_robot_at(tracking.header.stamp, timeout_s=2.0)
            except TransformException as ex:
                self.get_logger().error("Could not get transforms")
                self.get_logger().error(str(ex))
                if self.tracking_base_frame in str(ex) and "does not exist" in str(ex):
                    self.get_logger().error(
                        f"Frame '{self.tracking_base_frame}' (tracking_base_frame) not in TF. "
                        "It must match the optical frame published by your camera chain — "
                        "e.g. oak_right_camera_optical_frame (Piper + OAK-D SR), or "
                        "camera_optical_frame / <prefix>camera_optical_frame from your URDF/sim. "
                        "Check: ros2 run tf2_ros tf2_monitor"
                    )
                elif self.tracking_marker_frame in str(ex) and "does not exist" in str(ex):
                    self.get_logger().error(
                        f"Frame '{self.tracking_marker_frame}' not in TF. "
                        "Ensure: 1) charuco_detector node is running (started by calibration.launch.py); "
                        "2) camera image + camera_info topics are publishing; "
                        "3) ChArUco board fully visible to the camera; 4) chessboard_visible is true before capture_point."
                    )
                resp.success = False
                resp.message = str(ex)
                return resp
            tracking_burst = [tracking]
            robot_burst = [robot]

        robot_list = [get_transform(t.transform) for t in robot_burst]
        tracking_list = [get_transform(t.transform) for t in tracking_burst]

        robot_avg, robot_spread = CalibrationBackend.average_transforms(robot_list)
        tracking_avg, tracking_spread = CalibrationBackend.average_transforms(tracking_list)

        if robot_spread['max_translation_dev_m'] > 0.003 or robot_spread['max_rotation_dev_deg'] > 0.3:
            self.get_logger().warning(
                f"Robot appears to have moved during the capture burst "
                f"(max deviation {robot_spread['max_translation_dev_m'] * 1000:.2f}mm / "
                f"{robot_spread['max_rotation_dev_deg']:.2f}deg over {robot_spread['count']} frame(s)). "
                "Hold the arm still while capturing for the most precise sample."
            )

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

        self.robot_samples.append(robot_avg)
        self.tracking_samples.append(tracking_avg)
        self.sample_metrics.append(metrics)
        # The cached uncertainty described the previous sample set; it is stale
        # the moment a new sample lands. Re-estimating here would add seconds to
        # every capture, so drop it and let save/estimate_uncertainty redo it.
        self._last_uncertainty = None
        self._log_diversity()

        cal = self.get_calibration()
        if cal is None:
            msg = "Not enough samples yet..."
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

    def get_calibration(self):
        if len(self.robot_samples) < 4:
            self.get_logger().info("Not enough samples yet...")
            return None

        self.get_logger().info("Estimating ...")
        try:
            detail = CalibrationBackend.compute_calibration_detailed(
                samples_robot=self.robot_samples, samples_tracking=self.tracking_samples)
        except (RuntimeError, ValueError) as exc:
            # The backend translates OpenCV's cv2.error into RuntimeError, so
            # a degenerate pose set surfaces here as a failed service call
            # rather than as an exception escaping into the executor.
            self.get_logger().error(f"Calibration failed: {exc}")
            self._last_calibration_detail = None
            return None

        self._last_calibration_detail = detail
        if detail['rejected_indices']:
            self.get_logger().warning(
                f"Rejected {len(detail['rejected_indices'])} outlier sample(s) "
                f"(indices {detail['rejected_indices']}) as inconsistent with the rest before fitting."
            )
        refinement = detail['refinement']
        self.get_logger().info(
            f"Hand-eye algorithm: {detail['algorithm_used']} (auto-selected by cross-validating "
            "Tsai/Park/Horaud/Andreff/Daniilidis); nonlinear refinement moved the estimate by "
            f"{refinement['delta_translation_m'] * 1000:.2f}mm / {refinement['delta_rotation_deg']:.3f}deg"
        )
        return detail['transform']

    def _compute_uncertainty(self, cal):
        """Bootstrap the calibration uncertainty for the current sample set.

        Returns None when disabled or when there is not enough data. Slow by
        design (a full refit per resample), so callers decide when to pay it.
        """
        if self.bootstrap_samples <= 0:
            return None
        detail = self._last_calibration_detail
        if cal is None or detail is None:
            return None

        self.get_logger().info(
            f"Estimating calibration uncertainty ({self.bootstrap_samples} bootstrap resamples), "
            "this takes a few seconds..."
        )
        try:
            uncertainty = CalibrationBackend.bootstrap_uncertainty(
                samples_robot=self.robot_samples,
                samples_tracking=self.tracking_samples,
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
        return uncertainty

    def estimate_uncertainty_service_callback(self, req: Trigger.Request, resp: Trigger.Response):
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
        """Save current calibration estimate to YAML file for later publishing."""
        cal = self.get_calibration()
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
                'sample_metrics': self.sample_metrics,
                'diversity': diversity,
                'residuals': residuals,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'transform': {
                    'tx': cal[0], 'ty': cal[1], 'tz': cal[2],
                    'qx': cal[3], 'qy': cal[4], 'qz': cal[5], 'qw': cal[6],
                },
                'algorithm_used': (self._last_calibration_detail or {}).get('algorithm_used'),
                'rejected_sample_indices': (self._last_calibration_detail or {}).get('rejected_indices'),
                'nonlinear_refinement': (self._last_calibration_detail or {}).get('refinement'),
                'uncertainty': uncertainty,
            }
            os.makedirs(os.path.dirname(os.path.abspath(cal_file)) or '.', exist_ok=True)
            with open(cal_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            self.get_logger().info("Calibration saved to %s" % cal_file)
            resp.success = True
            resp.message = "Saved to " + cal_file
        except Exception as e:
            self.get_logger().error("Failed to save calibration: %s" % str(e))
            resp.success = False
            resp.message = str(e)
        return resp


def main():
    rclpy.init()
    node = DataCollector()

    # MultiThreadedExecutor is required, not just nice to have: the capture
    # callback blocks while collecting its burst, and the TF listener has to
    # keep processing /tf on another thread for that burst to see fresh data.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
