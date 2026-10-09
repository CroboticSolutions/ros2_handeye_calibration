"""Publish which calibration each mount is using, so bags and logs record it.

/calibration/active (std_msgs/String, transient local): JSON
{"robot_id": ..., "mounts": {"<mount>": {"calibration_id", "created_utc", "file",
"source": "store"|"default", ...}}}. Republished whenever a store's
current.yaml is repointed (Apply / rollback); polled once a second.

Parameter `mounts`: ["<mount>=<default yaml>", ...], e.g.
"femto_bolt_handeye=$(find piper_description)/config/calibration/femto_bolt_handeye_default.yaml".
"""
import json
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from hand_eye_calibration import calibration_store


def snapshot(mounts):
    out = {}
    for mount, default in mounts.items():
        data = calibration_store.active(mount, default or None)
        if data is None:
            out[mount] = None
            continue
        current = calibration_store.current_path(mount)
        out[mount] = {
            'calibration_id': data.get('calibration_id'),
            'created_utc': data.get('created_utc'),
            'label': data.get('label'),
            'parent_link': data.get('parent_link'),
            'child_link': data.get('child_link'),
            'file': data.get('_file'),
            'source': 'store' if os.path.exists(current) else 'default',
        }
    return {'robot_id': calibration_store.robot_id(), 'mounts': out}


class ActiveCalibrationPublisher(Node):
    def __init__(self):
        super().__init__('active_calibration_publisher')
        specs = self.declare_parameter('mounts', [calibration_store.FEMTO_HANDEYE + '=']).value
        self._mounts = dict(s.split('=', 1) if '=' in s else (s, '') for s in specs)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(String, '/calibration/active', qos)
        self._last = None
        self.create_timer(1.0, self._poll)
        self._poll()

    def _poll(self):
        try:
            text = json.dumps(snapshot(self._mounts), sort_keys=True)
        except Exception as exc:  # noqa: BLE001 - a bad file must not kill the publisher
            self.get_logger().error(f'Reading calibration store failed: {exc}')
            return
        if text == self._last:
            return
        self._last = text
        self._pub.publish(String(data=text))
        self.get_logger().info(f'Active calibration: {text}')


def main(args=None):
    rclpy.init(args=args)
    node = ActiveCalibrationPublisher()
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
