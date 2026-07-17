#!/usr/bin/env python3
"""
Tool TCP (pivot) calibration.

Collect the robot flange pose while the operator holds the tool tip against a
fixed calibration spike, from several different wrist orientations, and fit
the flange -> tool_tip translation. See pivot_backend.py for the math and
pivot_status.py for the GUI readiness/checklist payload.

Two modes, driven from the GUI (no relaunch needed to switch):

* Position only (single point): one round of touches with the tool tip on the
  spike. Recovers the TCP *translation* only; the TCP orientation defaults to
  the flange orientation (identity offset).

* Position + axis (align to spike): a first round with the tool tip (position),
  then an alignment round where the operator makes the tool's straight tip
  segment collinear with the calibration spike and captures the flange pose.
  Because the spike direction is KNOWN in the base frame (vertical -> [0,0,1]
  when the base is level), the tool axis in the flange frame is R_flangeᵀ @ n.
  Averaging a few alignment poses gives the tool axis, so the saved TCP gets a
  real orientation. This is the path for curved/gooseneck tools (e.g. a welding
  torch). Roll about the axis stays pinned (not observable). The alignment round
  keeps the internal key "axis_ref" for GUI/bridge backward-compat.
"""

import os
from datetime import datetime, timezone

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.time import Duration
from geometry_msgs.msg import Transform
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from .pivot_backend import PivotCalibrationBackend
from .pivot_status import build_tool_tcp_status, status_to_json

TIP_ROUND = "tip"
# The axis-alignment round keeps the historical key "axis_ref" so the GUI
# service names, bridge config and status keys stay unchanged.
ALIGN_ROUND = "axis_ref"


def get_transform(tf: Transform):
    tr = tf.translation
    qt = tf.rotation
    return [tr.x, tr.y, tr.z, qt.x, qt.y, qt.z, qt.w]


def _parse_axis(value) -> list:
    """Parse a spike-direction param: accepts [x,y,z] or a 'x,y,z' string."""
    if isinstance(value, (list, tuple)):
        nums = [float(v) for v in value]
    else:
        nums = [float(v) for v in str(value).replace(";", ",").split(",") if v.strip() != ""]
    if len(nums) != 3:
        return [0.0, 0.0, 1.0]
    return nums


class PivotCollector(Node):

    def __init__(self):
        mname = "tool_tcp_calibration"
        super().__init__(mname)

        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('robot_flange_frame', 'link6')
        self.declare_parameter('tcp_name', 'tool_tcp')
        self.declare_parameter('calibration_file', os.path.expanduser('~/.ros/tool_tcp_calibration.yaml'))
        # Known spike direction in the base frame for the axis-alignment round.
        # Default: vertical (spike perpendicular to a level base).
        self.declare_parameter('spike_axis_base', '0,0,1')

        self.robot_base_frame = str(self.get_parameter('robot_base_frame').value)
        self.robot_flange_frame = str(self.get_parameter('robot_flange_frame').value)
        self.tcp_name = str(self.get_parameter('tcp_name').value)
        self.spike_axis_base = _parse_axis(self.get_parameter('spike_axis_base').value)

        self.capture_point_service = self.create_service(
            Trigger, mname + "/capture_point", self.capture_point_cb)
        self.remove_last_sample_service = self.create_service(
            Trigger, mname + "/remove_last_sample", self.remove_last_sample_cb)
        self.reset_service = self.create_service(
            Trigger, mname + "/reset", self.reset_cb)
        self.select_round_tip_service = self.create_service(
            Trigger, mname + "/select_round_tip", self.select_round_tip_cb)
        self.select_round_axis_ref_service = self.create_service(
            Trigger, mname + "/select_round_axis_ref", self.select_round_axis_ref_cb)
        self.compute_axis_service = self.create_service(
            Trigger, mname + "/compute_axis", self.compute_axis_cb)
        self.save_calibration_service = self.create_service(
            Trigger, mname + "/save_calibration", self.save_calibration_cb)

        status_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.status_pub = self.create_publisher(String, mname + "/status", status_qos)

        self.tf_buffer = Buffer()
        self._listener = TransformListener(self.tf_buffer, self)

        # Tip pivot round (position) + axis-alignment round (poses with the tool
        # held collinear with the spike).
        self.samples = {TIP_ROUND: [], ALIGN_ROUND: []}
        self.active_round = TIP_ROUND
        # Cached axis result (invalidated on any sample change).
        self.axis_result = None

        self._preflight_logged = False
        self.create_timer(2.0, self._preflight_timer_cb)
        self._publish_status()

    def _preflight_timer_cb(self):
        if self._preflight_logged:
            return
        try:
            self.tf_buffer.lookup_transform(
                self.robot_base_frame, self.robot_flange_frame,
                rclpy.time.Time(), Duration(seconds=0.2))
            self.get_logger().info(
                f"Preflight OK: {self.robot_base_frame} -> {self.robot_flange_frame}")
            self._preflight_logged = True
        except TransformException as ex:
            self.get_logger().warning(
                f"Preflight: missing {self.robot_base_frame} -> {self.robot_flange_frame}: {ex}")

    def _compute_round(self, round_key):
        samples = self.samples[round_key]
        if len(samples) < PivotCalibrationBackend.MIN_SAMPLES:
            return None
        try:
            return PivotCalibrationBackend.compute_pivot(samples)
        except ValueError:
            return None

    def _current_mode(self):
        """Derive the mode from state: 'axis' once the operator engages the
        alignment round in any way, otherwise 'position'."""
        if (
            self.samples[ALIGN_ROUND]
            or self.active_round == ALIGN_ROUND
            or self.axis_result is not None
        ):
            return "axis"
        return "position"

    def _publish_status(self):
        tip_pivot = self._compute_round(TIP_ROUND)
        status = build_tool_tcp_status(
            mode=self._current_mode(),
            active_round=self.active_round,
            tip_samples=self.samples[TIP_ROUND],
            tip_pivot=tip_pivot,
            align_samples=self.samples[ALIGN_ROUND],
            axis=self.axis_result,
        )
        msg = String()
        msg.data = status_to_json(status)
        self.status_pub.publish(msg)
        return tip_pivot, status

    def _round_label(self, round_key):
        return "tip" if round_key == TIP_ROUND else "alignment"

    def capture_point_cb(self, req: Trigger.Request, resp: Trigger.Response):
        # Prefer slightly past time so the TF buffer has data; avoid negative
        # time when use_sim_time is true but /clock never publishes.
        now = self.get_clock().now()
        if now.nanoseconds >= 1_000_000_000:
            lookup_time = now - Duration(seconds=1)
        else:
            self.get_logger().warning(
                'Clock near epoch (use_sim_time without /clock?). Using current time for TF lookup.')
            lookup_time = now

        try:
            flange = self.tf_buffer.lookup_transform(
                self.robot_base_frame, self.robot_flange_frame,
                lookup_time, Duration(seconds=2))
        except TransformException as ex:
            self.get_logger().error(f"Could not get flange TF: {ex}")
            resp.success = False
            resp.message = str(ex)
            return resp

        round_key = self.active_round
        self.samples[round_key].append(get_transform(flange.transform))
        # Any sample change invalidates a previously computed axis.
        self.axis_result = None
        tip_pivot, _status = self._publish_status()

        count = len(self.samples[round_key])
        label = self._round_label(round_key)
        if round_key == ALIGN_ROUND:
            resp.success = True
            resp.message = (
                f"Captured {label} pose {count}. Press \"Compute axis\" when done "
                "(tip round must also be ready)."
            )
        elif tip_pivot is None:
            resp.success = True
            resp.message = f"Captured {label} sample {count}."
        else:
            t = tip_pivot["tcp_translation"]
            resp.success = True
            resp.message = (
                f"Captured {label} sample {count}. "
                f"{label} estimate: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m, "
                f"RMS {tip_pivot['rms_residual_m'] * 1000:.2f} mm"
            )
        return resp

    def remove_last_sample_cb(self, req: Trigger.Request, resp: Trigger.Response):
        round_key = self.active_round
        label = self._round_label(round_key)
        if not self.samples[round_key]:
            resp.success = False
            resp.message = f"No {label} samples to remove."
            return resp
        self.samples[round_key].pop()
        self.axis_result = None
        self._publish_status()
        resp.success = True
        resp.message = f"Removed last {label} sample. {len(self.samples[round_key])} remaining."
        return resp

    def reset_cb(self, req: Trigger.Request, resp: Trigger.Response):
        round_key = self.active_round
        label = self._round_label(round_key)
        self.samples[round_key] = []
        self.axis_result = None
        self._publish_status()
        resp.success = True
        resp.message = f"Cleared all {label} samples."
        return resp

    def select_round_tip_cb(self, req: Trigger.Request, resp: Trigger.Response):
        self.active_round = TIP_ROUND
        self._publish_status()
        resp.success = True
        resp.message = "Active round: tool tip."
        return resp

    def select_round_axis_ref_cb(self, req: Trigger.Request, resp: Trigger.Response):
        self.active_round = ALIGN_ROUND
        self._publish_status()
        resp.success = True
        resp.message = "Active round: axis alignment (hold the tool collinear with the spike)."
        return resp

    def compute_axis_cb(self, req: Trigger.Request, resp: Trigger.Response):
        tip_pivot = self._compute_round(TIP_ROUND)
        align_samples = self.samples[ALIGN_ROUND]
        if tip_pivot is None:
            self.axis_result = None
            self._publish_status()
            resp.success = False
            resp.message = (
                f"Tip round needs at least {PivotCalibrationBackend.MIN_SAMPLES} samples "
                "before the axis can be computed."
            )
            return resp
        if len(align_samples) < PivotCalibrationBackend.MIN_ALIGN_SAMPLES:
            self.axis_result = None
            self._publish_status()
            resp.success = False
            resp.message = (
                f"Capture at least {PivotCalibrationBackend.MIN_ALIGN_SAMPLES} alignment "
                "pose (tool collinear with the spike) before computing the axis."
            )
            return resp
        try:
            self.axis_result = PivotCalibrationBackend.compute_axis_from_alignment(
                alignment_samples=align_samples,
                spike_axis_base=self.spike_axis_base,
                tip_translation=tip_pivot["tcp_translation"],
            )
        except ValueError as ex:
            self.axis_result = None
            self._publish_status()
            resp.success = False
            resp.message = str(ex)
            return resp

        self._publish_status()
        a = self.axis_result
        resp.success = True
        resp.message = (
            f"Axis computed from {a['sample_count']} alignment pose(s)"
            + (
                f", spread ±{a['alignment_spread_deg']:.2f}°"
                if a["sample_count"] > 1
                else " (single pose — accuracy = your manual alignment)"
            )
        )
        return resp

    def save_calibration_cb(self, req: Trigger.Request, resp: Trigger.Response):
        tip_pivot = self._compute_round(TIP_ROUND)
        if tip_pivot is None:
            resp.success = False
            resp.message = (
                f"Not enough tip samples (need at least {PivotCalibrationBackend.MIN_SAMPLES})."
            )
            return resp

        # An alignment round was started but the axis was not (re)computed: saving
        # now would silently fall back to identity orientation, which is almost
        # certainly not what an axis calibration wanted.
        if self.samples[ALIGN_ROUND] and self.axis_result is None:
            resp.success = False
            resp.message = (
                "An alignment round is in progress: press \"Compute axis\" (tip ready + at "
                "least one alignment pose) before saving, or clear the alignment round to "
                "save position only."
            )
            return resp

        mode = self._current_mode()

        cal_file = os.path.expanduser(str(self.get_parameter('calibration_file').value))
        try:
            if self.axis_result is not None:
                q = self.axis_result["quaternion"]
                qx, qy, qz, qw = q[0], q[1], q[2], q[3]
            else:
                # Orientation is not observable from a single-point pivot touch;
                # default to the flange orientation until an axis-calibration
                # round (second touch point) or CAD data supplies a real one.
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

            data = {
                'parent_frame': self.robot_flange_frame,
                'tcp_name': self.tcp_name,
                'robot_base_frame': self.robot_base_frame,
                'robot_flange_frame': self.robot_flange_frame,
                'calibration_mode': mode,
                'sample_count': len(self.samples[TIP_ROUND]),
                'condition_number': tip_pivot['condition_number'],
                'rms_residual_m': tip_pivot['rms_residual_m'],
                'max_residual_m': tip_pivot['max_residual_m'],
                'per_sample_residuals_m': tip_pivot['per_sample_residuals_m'],
                'fixed_point_base_frame': {
                    'x': tip_pivot['fixed_point'][0],
                    'y': tip_pivot['fixed_point'][1],
                    'z': tip_pivot['fixed_point'][2],
                },
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'transform': {
                    'tx': tip_pivot['tcp_translation'][0],
                    'ty': tip_pivot['tcp_translation'][1],
                    'tz': tip_pivot['tcp_translation'][2],
                    'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
                },
            }
            if self.axis_result is not None:
                data['axis_calibration'] = {
                    'method': 'align_to_spike',
                    'axis_dir_flange_frame': self.axis_result['axis_dir'],
                    'spike_axis_base_frame': self.axis_result.get('spike_axis_base'),
                    'alignment_spread_deg': self.axis_result.get('alignment_spread_deg'),
                    'alignment_sample_count': self.axis_result.get('sample_count'),
                }

            os.makedirs(os.path.dirname(os.path.abspath(cal_file)) or '.', exist_ok=True)
            with open(cal_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            self.get_logger().info(f"TCP calibration saved to {cal_file}")
            resp.success = True
            orientation_note = (
                "TCP orientation from the fitted tool axis."
                if self.axis_result is not None
                else "TCP orientation defaults to flange orientation."
            )
            resp.message = (
                f"Saved to {cal_file}. {orientation_note} "
                "Apply with tool_tcp_cli --update-xacro, rebuild the description package, "
                "and restart the stack before using arm/set_eelink."
            )
        except Exception as e:
            self.get_logger().error(f"Failed to save TCP calibration: {e}")
            resp.success = False
            resp.message = str(e)
        return resp


def main():
    rclpy.init()
    node = PivotCollector()
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
