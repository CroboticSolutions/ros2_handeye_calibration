#!/usr/bin/env python3
"""
Load a tool-TCP pivot calibration from YAML and publish it as a static TF
(flange -> tool TCP), for quick visualization/validation right after saving.

This does NOT make the TCP usable by MoveIt: ``arm/set_eelink`` requires the
link to exist in the URDF/SRDF. Use tool_tcp_cli.py --update-xacro to bake the
result into the robot description, rebuild, and restart the stack instead.
"""

import os
import yaml

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_ros import StaticTransformBroadcaster


class PivotCalibrationPublisher(Node):

    def __init__(self):
        super().__init__('tool_tcp_calibration_publisher')
        self.declare_parameter('calibration_file', os.path.expanduser('~/.ros/tool_tcp_calibration.yaml'))
        self._static_broadcaster = StaticTransformBroadcaster(self)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._stamped = None
        self._load_and_publish()

    def _load_and_publish(self):
        cal_file = os.path.expanduser(str(self.get_parameter('calibration_file').value))
        if not os.path.isfile(cal_file):
            self.get_logger().error('Tool TCP calibration file not found: %s' % cal_file)
            self.get_logger().info('Capture at least 4 pivot samples, then call save_calibration:')
            self.get_logger().info('  ros2 service call /tool_tcp_calibration/save_calibration std_srvs/srv/Trigger {}')
            return
        try:
            with open(cal_file, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error('Failed to load TCP calibration: %s' % str(e))
            return

        t = data['transform']
        parent = data.get('robot_flange_frame', data.get('parent_frame', 'link6'))
        child = data.get('tcp_name', 'tool_tcp')

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
        self._warn_if_duplicate_tf(parent, child)
        self._publish()

    def _warn_if_duplicate_tf(self, parent, child):
        try:
            existing = self._tf_buffer.lookup_transform(
                parent, child, rclpy.time.Time(), timeout=Duration(seconds=1.0)
            )
        except TransformException:
            return

        t = existing.transform.translation
        q = existing.transform.rotation
        self.get_logger().warning(
            "TF already exists for %s -> %s before publishing this calibration. "
            "If this comes from a baked xacro joint (tool_tcp_cli --update-xacro), "
            "do not run this publisher at the same time. "
            "Existing transform: xyz=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f]"
            % (parent, child, t.x, t.y, t.z, q.x, q.y, q.z, q.w)
        )

    def _publish(self):
        if self._stamped is None:
            return
        self._stamped.header.stamp = self.get_clock().now().to_msg()
        self._static_broadcaster.sendTransform(self._stamped)

    def timer_callback(self):
        self._publish()


def main():
    rclpy.init()
    node = PivotCalibrationPublisher()
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
