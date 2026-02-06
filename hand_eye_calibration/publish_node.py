#!/usr/bin/env python3
"""
Load hand-eye calibration from YAML and publish it as a static TF.
"""

import os
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


class CalibrationPublisher(Node):

    def __init__(self):
        super().__init__('hand_eye_calibration_publisher')
        self.declare_parameter('calibration_file', os.path.expanduser('~/.ros/hand_eye_calibration.yaml'))
        self._static_broadcaster = StaticTransformBroadcaster(self)
        self._stamped = None
        self._load_and_publish()

    def _load_and_publish(self):
        cal_file = os.path.expanduser(str(self.get_parameter('calibration_file').value))
        if not os.path.isfile(cal_file):
            self.get_logger().error('Calibration file not found: %s' % cal_file)
            self.get_logger().info('Run calibration, capture at least 4 points, then call save_calibration:')
            self.get_logger().info('  ros2 service call /hand_eye_calibration/save_calibration std_srvs/srv/Trigger {}')
            return
        try:
            with open(cal_file, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error('Failed to load calibration: %s' % str(e))
            return
        t = data['transform']
        parent = data['robot_effector_frame']
        child = data['tracking_base_frame']
        # Calibration result is "tracking_base -> robot_base"; we publish parent=robot_base, child=tracking_base
        # so the TF tree has robot_base -> tracking_base with the saved transform (camera in base frame).
        msg = TransformStamped()
        msg.header.frame_id = parent
        msg.child_frame_id = child
        msg.transform.translation.x = float(t['tx'])
        msg.transform.translation.y = float(t['ty'])
        msg.transform.translation.z = float(t['tz'])
        msg.transform.rotation.x = float(t['qx'])
        msg.transform.rotation.y = float(t['qy'])
        msg.transform.rotation.z = float(t['qz'])
        msg.transform.rotation.w = float(t['qw'])
        self._stamped = msg
        self._publish()

    def _publish(self):
        if self._stamped is None:
            return
        self._stamped.header.stamp = self.get_clock().now().to_msg()
        self._static_broadcaster.sendTransform(self._stamped)

    def timer_callback(self):
        self._publish()


def main():
    rclpy.init()
    node = CalibrationPublisher()
    if node._stamped is not None:
        timer = node.create_timer(1.0, node.timer_callback)
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
