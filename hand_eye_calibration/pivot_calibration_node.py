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
import time
from datetime import datetime, timezone

import numpy as np
import rclpy
import yaml
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.time import Duration
from geometry_msgs.msg import Transform
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as Rot

from .pivot_backend import PivotCalibrationBackend
from .pivot_status import build_tool_tcp_status, status_to_json
from . import tcp_quality
from .calibration_dataset import plain

TIP_ROUND = "tip"
# The axis-alignment round keeps the historical key "axis_ref" so the GUI
# service names, bridge config and status keys stay unchanged.
ALIGN_ROUND = "axis_ref"
VALIDATION_ROUND = "validation"


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
        # Metadata only; this node never sends a CAN firmware query.
        self.declare_parameter('robot_firmware_version', 'unknown')
        self.declare_parameter('capture_burst_duration_s', 1.2)
        self.declare_parameter('capture_burst_samples', 50)
        self.declare_parameter('capture_burst_min_samples', 15)
        self.declare_parameter('capture_translation_p95_limit_m', 0.0005)
        self.declare_parameter('capture_rotation_p95_limit_deg', 0.20)
        self.declare_parameter('duplicate_orientation_limit_deg', 5.0)
        self.declare_parameter('max_tf_age_s', 0.25)
        # Acceptance gate (tcp_quality.DEFAULT_LIMITS); 'warn' saves anyway.
        self.declare_parameter('acceptance_mode', 'enforce')
        for key, value in tcp_quality.DEFAULT_LIMITS.items():
            self.declare_parameter('accept_' + key, value)
        self.declare_parameter('dataset_dir', os.path.expanduser('~/.ros/tool_tcp_calibration_runs'))
        # Nominal neck angle (tool axis vs flange Z) from CAD; < 0 disables the check.
        self.declare_parameter('cad_axis_angle_deg', 35.0)
        # 'bend_plane': TCP +X in the torch-neck bend plane; 'legacy': flange-X hint.
        self.declare_parameter('tcp_roll_convention', 'bend_plane')
        # Reorientation check motion.
        self.declare_parameter('reorient_angle_deg', 25.0)
        self.declare_parameter('reorient_spin_deg', 45.0)
        self.declare_parameter('reorient_standoff_m', 0.005)
        self.declare_parameter('reorient_clearance_m', 0.03)
        self.declare_parameter('auto_group', '')
        self.declare_parameter('auto_controller', '')
        self.declare_parameter('auto_check_collisions', True)
        self.declare_parameter('motion_heartbeat_timeout_s', 0.0)

        self.robot_base_frame = str(self.get_parameter('robot_base_frame').value)
        self.robot_flange_frame = str(self.get_parameter('robot_flange_frame').value)
        self.tcp_name = str(self.get_parameter('tcp_name').value)
        self.spike_axis_base = _parse_axis(self.get_parameter('spike_axis_base').value)
        self.robot_firmware_version = str(
            self.get_parameter('robot_firmware_version').value)
        self.capture_burst_duration_s = float(self.get_parameter('capture_burst_duration_s').value)
        self.capture_burst_samples = int(self.get_parameter('capture_burst_samples').value)
        self.capture_burst_min_samples = int(self.get_parameter('capture_burst_min_samples').value)
        self.capture_translation_p95_limit_m = float(
            self.get_parameter('capture_translation_p95_limit_m').value)
        self.capture_rotation_p95_limit_deg = float(
            self.get_parameter('capture_rotation_p95_limit_deg').value)
        self.duplicate_orientation_limit_deg = float(
            self.get_parameter('duplicate_orientation_limit_deg').value)
        self.max_tf_age_s = float(self.get_parameter('max_tf_age_s').value)
        self.acceptance_mode = str(self.get_parameter('acceptance_mode').value)
        if self.acceptance_mode not in ('enforce', 'warn'):
            raise ValueError("acceptance_mode must be 'enforce' or 'warn'")
        self.acceptance_limits = {k: type(v)(self.get_parameter('accept_' + k).value)
                                  for k, v in tcp_quality.DEFAULT_LIMITS.items()}
        self.cad_axis_angle_deg = float(self.get_parameter('cad_axis_angle_deg').value)
        self.roll_convention = str(self.get_parameter('tcp_roll_convention').value)
        from .tcp_reorientation import TcpMotion
        self.motion = TcpMotion(
            self, self.robot_base_frame, self.robot_flange_frame,
            group=str(self.get_parameter('auto_group').value),
            controller=str(self.get_parameter('auto_controller').value),
            check_collisions=bool(self.get_parameter('auto_check_collisions').value),
            heartbeat_timeout=float(self.get_parameter('motion_heartbeat_timeout_s').value))
        self.reorient_index = 0

        # Separate from the TransformListener's default group: TF updates must
        # continue on another executor thread while a capture service waits for
        # its burst, but two captures must never mutate the sample lists at once.
        self._callback_group = MutuallyExclusiveCallbackGroup()

        self.capture_point_service = self.create_service(
            Trigger, mname + "/capture_point", self.capture_point_cb,
            callback_group=self._callback_group)
        self.remove_last_sample_service = self.create_service(
            Trigger, mname + "/remove_last_sample", self.remove_last_sample_cb,
            callback_group=self._callback_group)
        self.reset_service = self.create_service(
            Trigger, mname + "/reset", self.reset_cb,
            callback_group=self._callback_group)
        self.select_round_tip_service = self.create_service(
            Trigger, mname + "/select_round_tip", self.select_round_tip_cb,
            callback_group=self._callback_group)
        self.select_round_axis_ref_service = self.create_service(
            Trigger, mname + "/select_round_axis_ref", self.select_round_axis_ref_cb,
            callback_group=self._callback_group)
        self.select_round_validation_service = self.create_service(
            Trigger, mname + "/select_round_validation", self.select_round_validation_cb,
            callback_group=self._callback_group)
        self.compute_axis_service = self.create_service(
            Trigger, mname + "/compute_axis", self.compute_axis_cb,
            callback_group=self._callback_group)
        self.reorient_next_service = self.create_service(
            Trigger, mname + "/reorient_next", self.reorient_next_cb, callback_group=self._callback_group)
        self.motion_stop_service = self.create_service(
            Trigger, mname + "/motion_stop", self.motion_stop_cb, callback_group=self.motion.cb)
        self.save_calibration_service = self.create_service(
            Trigger, mname + "/save_calibration", self.save_calibration_cb,
            callback_group=self._callback_group)

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
        self.samples = {TIP_ROUND: [], ALIGN_ROUND: [], VALIDATION_ROUND: []}
        self.sample_metadata = {TIP_ROUND: [], ALIGN_ROUND: [], VALIDATION_ROUND: []}
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
        validation_result = (
            PivotCalibrationBackend.validate_pivot(
                self.samples[VALIDATION_ROUND], tip_pivot)
            if tip_pivot is not None and self.samples[VALIDATION_ROUND]
            else None
        )
        status = build_tool_tcp_status(
            mode=self._current_mode(),
            active_round=self.active_round,
            tip_samples=self.samples[TIP_ROUND],
            tip_pivot=tip_pivot,
            validation_samples=self.samples[VALIDATION_ROUND],
            validation_result=validation_result,
            align_samples=self.samples[ALIGN_ROUND],
            axis=self.axis_result,
        )
        acceptance = self._acceptance(tip_pivot, validation_result)
        status['acceptance'] = acceptance
        status['rotation_spans_deg'] = tcp_quality.rotation_spans_deg(self.samples[TIP_ROUND])
        status['cad_axis_deviation_deg'] = self._cad_deviation()
        status['reorientation'] = {**self.motion.state, 'next_index': self.reorient_index,
                                   'targets': 6}
        # Save stays possible with a fit: the node then writes the active file
        # only if acceptance passes, otherwise a .rejected.yaml candidate + dataset.
        msg = String()
        msg.data = status_to_json(status)
        self.status_pub.publish(msg)
        return tip_pivot, status

    def _cad_deviation(self):
        if self.axis_result is None:
            return None
        return tcp_quality.cad_axis_deviation_deg(self.axis_result['axis_dir'], self.cad_axis_angle_deg)

    def _acceptance(self, tip_pivot, validation_result):
        return tcp_quality.evaluate(
            tip_pivot, tcp_quality.rotation_spans_deg(self.samples[TIP_ROUND]), validation_result,
            axis=self.axis_result, axis_mode=self._current_mode() == 'axis',
            cad_deviation=self._cad_deviation(), limits=self.acceptance_limits)

    def _current_joints(self):
        joints = self.motion.joints
        if joints is None or time.monotonic() - self.motion.joint_at > 1.0:
            return None
        return {n: float(p) for n, p in zip(joints.name, joints.position)}

    def _round_label(self, round_key):
        return {
            TIP_ROUND: "tip",
            ALIGN_ROUND: "alignment",
            VALIDATION_ROUND: "validation",
        }[round_key]

    def _capture_burst(self):
        """Collect fresh TF values while the executor services TF callbacks."""
        deadline = time.monotonic() + self.capture_burst_duration_s
        burst = []
        seen_stamps = set()
        while len(burst) < self.capture_burst_samples and time.monotonic() < deadline:
            try:
                flange = self.tf_buffer.lookup_transform(
                    self.robot_base_frame,
                    self.robot_flange_frame,
                    rclpy.time.Time(),
                    Duration(seconds=0.10),
                )
            except TransformException:
                time.sleep(0.01)
                continue

            stamp = flange.header.stamp
            stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            if stamp_ns in seen_stamps:
                time.sleep(0.005)
                continue
            now_ns = self.get_clock().now().nanoseconds
            age_s = (now_ns - stamp_ns) / 1e9
            if stamp_ns > 0 and (age_s < -0.02 or age_s > self.max_tf_age_s):
                time.sleep(0.005)
                continue
            seen_stamps.add(stamp_ns)
            burst.append(get_transform(flange.transform))
            time.sleep(0.005)

        if len(burst) < self.capture_burst_min_samples:
            raise ValueError(
                f"Only {len(burst)} fresh TF samples arrived; need "
                f"{self.capture_burst_min_samples}. Check joint-state/TF publishing."
            )
        aggregate = PivotCalibrationBackend.aggregate_pose_burst(burst)
        if aggregate["translation_p95_m"] > self.capture_translation_p95_limit_m:
            raise ValueError(
                "Robot moved during capture: translation spread "
                f"{aggregate['translation_p95_m'] * 1000:.2f} mm exceeds "
                f"{self.capture_translation_p95_limit_m * 1000:.2f} mm."
            )
        if aggregate["rotation_p95_deg"] > self.capture_rotation_p95_limit_deg:
            raise ValueError(
                "Robot moved during capture: rotation spread "
                f"{aggregate['rotation_p95_deg']:.2f}° exceeds "
                f"{self.capture_rotation_p95_limit_deg:.2f}°."
            )
        return aggregate

    def _nearest_orientation_deg(self, pose, existing_samples):
        if not existing_samples:
            return float("inf")
        candidate = Rot.from_quat(pose[3:])
        return min(
            float(np.degrees((Rot.from_quat(sample[3:]).inv() * candidate).magnitude()))
            for sample in existing_samples
        )

    def capture_point_cb(self, req: Trigger.Request, resp: Trigger.Response):
        try:
            aggregate = self._capture_burst()
        except (TransformException, ValueError) as ex:
            self.get_logger().error(f"Could not capture stable flange pose: {ex}")
            resp.success = False
            resp.message = str(ex)
            return resp

        round_key = self.active_round
        pose = aggregate["pose"]
        comparison_samples = self.samples[round_key]
        if round_key == VALIDATION_ROUND:
            tip_pivot, current_status = self._publish_status()
            if tip_pivot is None or not current_status["tip"]["ready_to_save"]:
                resp.success = False
                resp.message = "Finish a high-quality 20-touch tip fit before validation."
                return resp
            comparison_samples = self.samples[TIP_ROUND] + self.samples[VALIDATION_ROUND]

        nearest_deg = self._nearest_orientation_deg(pose, comparison_samples)
        if nearest_deg < self.duplicate_orientation_limit_deg:
            resp.success = False
            resp.message = (
                f"Pose rejected: nearest captured wrist orientation is only {nearest_deg:.1f}° "
                f"away (minimum {self.duplicate_orientation_limit_deg:.1f}°)."
            )
            return resp

        self.samples[round_key].append(pose)
        self.sample_metadata[round_key].append(
            {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "burst_sample_count": aggregate["sample_count"],
                "translation_p95_m": aggregate["translation_p95_m"],
                "translation_max_m": aggregate["translation_max_m"],
                "rotation_p95_deg": aggregate["rotation_p95_deg"],
                "rotation_max_deg": aggregate["rotation_max_deg"],
                "nearest_orientation_deg": None if not np.isfinite(nearest_deg) else nearest_deg,
                "joints": self._current_joints(),
                "reorientation_target": (self.reorient_index - 1 if round_key == VALIDATION_ROUND
                                         and self.motion.state.get('state') == 'done' else None),
            }
        )
        if round_key == TIP_ROUND:
            # A changed fit must be validated again on untouched poses.
            self.samples[VALIDATION_ROUND] = []
            self.sample_metadata[VALIDATION_ROUND] = []
        if round_key != VALIDATION_ROUND:
            self.axis_result = None
        tip_pivot, _status = self._publish_status()

        count = len(self.samples[round_key])
        label = self._round_label(round_key)
        if round_key == VALIDATION_ROUND:
            validation = PivotCalibrationBackend.validate_pivot(
                self.samples[VALIDATION_ROUND], tip_pivot)
            resp.success = True
            resp.message = (
                f"Captured held-out validation pose {count}. "
                f"RMS {validation['rms_residual_m'] * 1000:.2f} mm, "
                f"max {validation['max_residual_m'] * 1000:.2f} mm."
            )
        elif round_key == ALIGN_ROUND:
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
        if self.sample_metadata[round_key]:
            self.sample_metadata[round_key].pop()
        if round_key == TIP_ROUND:
            self.samples[VALIDATION_ROUND] = []
            self.sample_metadata[VALIDATION_ROUND] = []
        if round_key != VALIDATION_ROUND:
            self.axis_result = None
        self._publish_status()
        resp.success = True
        resp.message = f"Removed last {label} sample. {len(self.samples[round_key])} remaining."
        return resp

    def reset_cb(self, req: Trigger.Request, resp: Trigger.Response):
        round_key = self.active_round
        label = self._round_label(round_key)
        self.samples[round_key] = []
        self.sample_metadata[round_key] = []
        if round_key in (TIP_ROUND, VALIDATION_ROUND):
            self.samples[VALIDATION_ROUND] = []
            self.sample_metadata[VALIDATION_ROUND] = []
            self.reorient_index = 0
        if round_key != VALIDATION_ROUND:
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

    def select_round_validation_cb(self, req: Trigger.Request, resp: Trigger.Response):
        self.active_round = VALIDATION_ROUND
        self._publish_status()
        resp.success = True
        resp.message = "Active round: held-out validation touches."
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

        if self.roll_convention == 'bend_plane':
            frame = tcp_quality.bend_plane_frame(self.axis_result['axis_dir'])
            if frame is not None:
                self.axis_result.update(quaternion=frame['quaternion'],
                                        rotation_matrix=frame['rotation_matrix'],
                                        roll_convention=frame['roll_convention'])
        self._publish_status()
        a = self.axis_result
        cad = self._cad_deviation()
        resp.success = True
        resp.message = (
            f"Axis computed from {a['sample_count']} alignment pose(s)"
            + (
                f", spread ±{a['alignment_spread_deg']:.2f}°"
                if a["sample_count"] > 1
                else " (single pose — accuracy = your manual alignment)"
            )
            + ("" if cad is None else f"; {cad:.1f}° from the CAD neck angle")
        )
        return resp

    # ------------------------------------------------------------------
    # Reorientation check
    # ------------------------------------------------------------------
    def reorient_next_cb(self, req, resp):
        tip_pivot = self._compute_round(TIP_ROUND)
        if tip_pivot is None:
            resp.success, resp.message = False, 'Fit the tip first (tip round).'
            return resp
        if self.motion.active:
            resp.success, resp.message = False, 'A reorientation move is already running.'
            return resp
        if self.reorient_index >= 6:
            resp.success, resp.message = False, 'All 6 reorientation poses done. Reset the validation round to repeat.'
            return resp
        try:
            flange = self.tf_buffer.lookup_transform(self.robot_base_frame, self.robot_flange_frame,
                                                     rclpy.time.Time(), Duration(seconds=0.2))
        except TransformException as exc:
            resp.success, resp.message = False, f'Flange pose unavailable: {exc}'
            return resp
        current = get_transform(flange.transform)
        reference = PivotCalibrationBackend.aggregate_pose_burst(self.samples[TIP_ROUND])['pose'][3:]
        targets = tcp_quality.reorientation_targets(
            reference, float(self.get_parameter('reorient_angle_deg').value),
            float(self.get_parameter('reorient_spin_deg').value), self.spike_axis_base)
        index = self.reorient_index
        self.active_round = VALIDATION_ROUND

        def done(ok):
            if ok:
                self.reorient_index = index + 1
            self._publish_status()

        self.motion.run_step(np.asarray(current), targets[index], tip_pivot['tcp_translation'],
                             tip_pivot['fixed_point'], self.spike_axis_base,
                             float(self.get_parameter('reorient_clearance_m').value),
                             float(self.get_parameter('reorient_standoff_m').value), done)
        self._publish_status()
        resp.success = True
        resp.message = (f'Reorientation pose {index + 1}/6 started: the tool rotates about its fitted tip above the '
                        'spike. Keep the E-stop at hand.')
        return resp

    def motion_stop_cb(self, req, resp):
        self.motion.stop()
        resp.success, resp.message = True, 'Stop requested; cancelling the trajectory.'
        return resp

    def save_calibration_cb(self, req: Trigger.Request, resp: Trigger.Response):
        tip_pivot, status = self._publish_status()
        if tip_pivot is None:
            resp.success = False
            resp.message = (
                f"Not enough tip samples (need at least {PivotCalibrationBackend.MIN_SAMPLES})."
            )
            return resp

        acceptance = status['acceptance']
        if tip_pivot is not None and len(self.samples[TIP_ROUND]) < PivotCalibrationBackend.MIN_SAMPLES:
            resp.success, resp.message = False, 'Not enough tip samples.'
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
                'robot_firmware_version': self.robot_firmware_version,
                'calibration_mode': mode,
                'sample_count': len(self.samples[TIP_ROUND]),
                'condition_number': tip_pivot['condition_number'],
                'rms_residual_m': tip_pivot['rms_residual_m'],
                'max_residual_m': tip_pivot['max_residual_m'],
                'per_sample_residuals_m': tip_pivot['per_sample_residuals_m'],
                'solver': {
                    'method': tip_pivot['method'],
                    'ransac_threshold_m': tip_pivot['ransac_threshold_m'],
                    'inlier_indices': tip_pivot['inlier_indices'],
                    'outlier_indices': tip_pivot['outlier_indices'],
                    'inlier_count': tip_pivot['inlier_count'],
                    'outlier_count': tip_pivot['outlier_count'],
                    'inlier_max_residual_m': tip_pivot['inlier_max_residual_m'],
                },
                'uncertainty': {
                    'bootstrap_count': tip_pivot['bootstrap_count'],
                    'tcp_std_m': tip_pivot['tcp_std_m'],
                    'tcp_ci95_low_m': tip_pivot['tcp_ci95_low_m'],
                    'tcp_ci95_high_m': tip_pivot['tcp_ci95_high_m'],
                    'tcp_ci95_half_width_m': tip_pivot['tcp_ci95_half_width_m'],
                    'max_loo_tcp_shift_m': tip_pivot['max_loo_tcp_shift_m'],
                    'loo_tcp_shifts_m': tip_pivot['loo_tcp_shifts_m'],
                    'influential_sample_indices': tip_pivot['influential_sample_indices'],
                },
                'validation': PivotCalibrationBackend.validate_pivot(
                    self.samples[VALIDATION_ROUND], tip_pivot),
                'raw_samples': {
                    'tip': self.samples[TIP_ROUND],
                    'tip_capture_metadata': self.sample_metadata[TIP_ROUND],
                    'validation': self.samples[VALIDATION_ROUND],
                    'validation_capture_metadata': self.sample_metadata[VALIDATION_ROUND],
                    'axis_alignment': self.samples[ALIGN_ROUND],
                    'axis_alignment_capture_metadata': self.sample_metadata[ALIGN_ROUND],
                },
                'capture_configuration': {
                    'burst_target_samples': self.capture_burst_samples,
                    'burst_min_samples': self.capture_burst_min_samples,
                    'burst_duration_s': self.capture_burst_duration_s,
                    'translation_p95_limit_m': self.capture_translation_p95_limit_m,
                    'rotation_p95_limit_deg': self.capture_rotation_p95_limit_deg,
                    'duplicate_orientation_limit_deg': self.duplicate_orientation_limit_deg,
                    'max_tf_age_s': self.max_tf_age_s,
                },
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

            if self.axis_result is not None:
                data['axis_calibration']['roll_convention'] = self.axis_result.get('roll_convention', 'legacy_flange_x')
                data['axis_calibration']['cad_axis_angle_deg'] = self.cad_axis_angle_deg
                data['axis_calibration']['cad_axis_deviation_deg'] = self._cad_deviation()
            data['rotation_spans_deg'] = status['rotation_spans_deg']
            data['acceptance'] = acceptance
            data['raw_samples']['joint_names'] = list(self.motion.names or [])
            run_dir = self._write_dataset(data)
            data['dataset_path'] = run_dir
            accepted = acceptance['passed'] or self.acceptance_mode == 'warn'
            target = cal_file if accepted else cal_file + '.rejected.yaml'
            self._write_yaml_atomic(target, data, keep_previous=accepted)
            if not accepted:
                self.get_logger().error(f"TCP calibration NOT saved as active: {acceptance['summary']}")
                resp.success = False
                resp.message = (f"Acceptance failed; active TCP unchanged. {acceptance['summary']} "
                                f"Candidate: {target}; dataset: {run_dir}")
                return resp
            self.get_logger().info(f"TCP calibration saved to {cal_file}")
            resp.success = True
            orientation_note = (
                "TCP orientation from the fitted tool axis."
                if self.axis_result is not None
                else "TCP orientation defaults to flange orientation."
            )
            warning = '' if acceptance['passed'] else f" WARNING (acceptance_mode=warn): {acceptance['summary']}"
            resp.message = (
                f"Saved to {cal_file}.{warning} {orientation_note} "
                "Apply with tool_tcp_cli --update-xacro, rebuild the description package, "
                "and restart the stack before using arm/set_eelink."
            )
        except Exception as e:
            self.get_logger().error(f"Failed to save TCP calibration: {e}")
            resp.success = False
            resp.message = str(e)
        return resp

    @staticmethod
    def _write_yaml_atomic(path, data, keep_previous):
        import shutil
        import tempfile
        directory = os.path.dirname(os.path.abspath(path)) or '.'
        os.makedirs(directory, exist_ok=True)
        if keep_previous and os.path.exists(path):
            shutil.copy2(path, path + '.previous')
        with tempfile.NamedTemporaryFile('w', dir=directory, delete=False) as f:
            yaml.safe_dump(plain(data), f, default_flow_style=False, sort_keys=False)
            temporary = f.name
        os.replace(temporary, path)



    def _write_dataset(self, data):
        """Raw TCP run (flange poses, joints, rounds, URDF) for offline re-solving."""
        import json
        root = os.path.expanduser(str(self.get_parameter('dataset_dir').value))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        run = os.path.join(root, stamp)
        suffix = 1
        while os.path.exists(run):
            run = os.path.join(root, f'{stamp}_{suffix}')
            suffix += 1
        try:
            os.makedirs(run)
            urdf = None
            try:
                from rcl_interfaces.srv import GetParameters
                if self.motion.description.wait_for_service(timeout_sec=0.5):
                    future = self.motion.description.call_async(GetParameters.Request(names=['robot_description']))
                    end = time.monotonic() + 2
                    while not future.done() and time.monotonic() < end:
                        time.sleep(0.02)
                    if future.done():
                        urdf = future.result().values[0].string_value
            except Exception:  # noqa: BLE001 - URDF is optional context
                urdf = None
            dataset = {'format_version': 1, 'kind': 'tool_tcp', 'created': datetime.now(timezone.utc).isoformat(),
                       'robot_base_frame': self.robot_base_frame, 'robot_flange_frame': self.robot_flange_frame,
                       'spike_axis_base': self.spike_axis_base, 'robot_description': urdf,
                       'raw_samples': data['raw_samples'], 'online_result': {
                           k: data.get(k) for k in ('transform', 'acceptance', 'rms_residual_m', 'fixed_point_base_frame')}}
            with open(os.path.join(run, 'dataset.json'), 'w') as f:
                json.dump(plain(dataset), f)
            return run
        except OSError as exc:
            self.get_logger().error(f'Could not write the TCP dataset: {exc}')
            return None


def main():
    rclpy.init()
    node = PivotCollector()
    executor = MultiThreadedExecutor(num_threads=3)
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
