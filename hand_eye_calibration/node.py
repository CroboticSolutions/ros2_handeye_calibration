# !/usr/bin/env python3
"""
Collect poses and perform calibration
"""

import math
import os
from datetime import datetime, timezone
import yaml

import numpy as np
import rclpy
from rclpy.wait_for_message import wait_for_message

from rclpy.node import Node
from rclpy.time import Duration
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from geometry_msgs.msg import TransformStamped, Transform
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

def tf_to_string(tf_stamped_message: TransformStamped):
    tfl = get_transform(tf_stamped_message.transform)
    out = str(tf_stamped_message.child_frame_id) + " -> " + str(tf_stamped_message.header.frame_id) + ":"
    out += "\n\t" + tf_list_to_string(tfl)
    return str(out)

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

        self.tracking_base_frame = str(self.get_parameter('tracking_base_frame').value)
        self.tracking_marker_frame = str(self.get_parameter('tracking_marker_frame').value)
        self.robot_base_frame = str(self.get_parameter('robot_base_frame').value)
        self.robot_effector_frame = str(self.get_parameter('robot_effector_frame').value)
        self.calibration_type = str(self.get_parameter('calibration_type').value)
        self.pointcloud_topic = str(self.get_parameter('pointcloud_topic').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.marker_size = float(self.get_parameter('marker_size').value)

        self.capture_point_service_name = mname + "/capture_point"
        self.capture_point_service = self.create_service(
            Trigger,
            self.capture_point_service_name,
            self.capture_point_service_callback)

        self.save_calibration_service_name = mname + "/save_calibration"
        self.save_calibration_service = self.create_service(
            Trigger,
            self.save_calibration_service_name,
            self.save_calibration_service_callback)

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

    def _sample_metrics(self, robot, tracking):
        robot_tf = get_transform(robot.transform)
        tracking_tf = get_transform(tracking.transform)
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
        if len(self.robot_samples) < 2:
            return None
        x = transform_to_matrix(cal)
        translation_errors = []
        rotation_errors = []
        for i in range(len(self.robot_samples) - 1):
            h_i = transform_to_matrix(self.robot_samples[i])
            h_j = transform_to_matrix(self.robot_samples[i + 1])
            c_i = transform_to_matrix(self.tracking_samples[i])
            c_j = transform_to_matrix(self.tracking_samples[i + 1])
            a = np.linalg.inv(h_j) @ h_i
            b = c_j @ np.linalg.inv(c_i)
            residual = np.linalg.inv(a @ x) @ (x @ b)
            t_err, r_err = matrix_to_residual(residual)
            translation_errors.append(t_err)
            rotation_errors.append(math.degrees(r_err))

        return {
            'mean_translation_m': float(np.mean(translation_errors)),
            'max_translation_m': float(np.max(translation_errors)),
            'mean_rotation_deg': float(np.mean(rotation_errors)),
            'max_rotation_deg': float(np.max(rotation_errors)),
            'pair_count': len(translation_errors),
        }

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

    def capture_point_service_callback(self, req: Trigger.Request, resp: Trigger.Response):
        self.log_preflight()
        # Prefer slightly past time so TF buffer has data; avoid negative time when
        # use_sim_time is true but /clock never publishes (clock stuck at zero).
        now = self.get_clock().now()
        if now.nanoseconds >= 1_000_000_000:
            lookup_time = now - Duration(seconds=1)
        else:
            self.get_logger().warning(
                'Clock near epoch (use_sim_time without /clock?). Using current time for TF lookup.'
            )
            lookup_time = now

        try:
            # here we trick the library (it is actually made for eye_in_hand only). Trust me, I'm an engineer
            if self.calibration_type == "eye-in-hand":
                robot = self.tf_buffer.lookup_transform(self.robot_base_frame,
                                                    self.robot_effector_frame, lookup_time,
                                                    Duration(seconds=2))
            elif self.calibration_type == "eye-on-base":
                robot = self.tf_buffer.lookup_transform(self.robot_effector_frame,
                                                    self.robot_base_frame, lookup_time,
                                                    Duration(seconds=2))
            else:
                msg = "Invalid calibration_type: " + self.calibration_type + ". Options are eye-in-hand or eye-on-base"
                self.get_logger().error(msg)
                resp.success = False
                resp.message = msg
                return resp

            tracking = self.tf_buffer.lookup_transform(self.tracking_base_frame,
                                                    self.tracking_marker_frame, lookup_time,
                                                    Duration(seconds=2))
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

        self.get_logger().info("robot: " + tf_to_string(robot))
        self.get_logger().info("tracking: " + tf_to_string(tracking))
        metrics = self._sample_metrics(robot, tracking)
        self._log_sample_quality(metrics)

        self.robot_samples.append(get_transform(robot.transform))
        self.tracking_samples.append(get_transform(tracking.transform))
        self.sample_metrics.append(metrics)
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
        else:
            self.get_logger().info("Estimating ...")
            cal = CalibrationBackend.compute_calibration(samples_robot=self.robot_samples,
                                                         samples_tracking=self.tracking_samples)
            return cal

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

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
