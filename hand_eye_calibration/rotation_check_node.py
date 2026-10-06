"""Rotation check of the APPLIED TCP (PivotCollector mixin).

The standard TCP verification (ABB/KUKA/FANUC/Yaskawa, Zivid "touch test"):
the operator brings the wire tip onto a fixed point and captures it; the robot
then rotates the tool about its current TCP (tilt +-angle about two horizontal
axes, spin +-angle about the vertical, then back) and stops `standoff` above the
point. With a correct TCP the tip stays over the point. The operator may lower
the tip onto the point by hand and capture; the deviation of the TCP position
from the start point is then measured in mm.

It always uses the TCP currently in TF (flange -> rotation_check_tcp_frame), so
it validates what MoveIt/welding actually use, independent of calibration rounds.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.time import Duration
from scipy.spatial.transform import Rotation as Rot
from tf2_ros import TransformException

from . import tcp_quality


def _pose(tf):
    t, r = tf.translation, tf.rotation
    return np.array([t.x, t.y, t.z, r.x, r.y, r.z, r.w], float)


def rotation_targets(reference_quat, tilt_deg, spin_deg):
    """Six reoriented poses about the tip, then back to the start orientation."""
    out = tcp_quality.reorientation_targets(reference_quat, tilt_deg, spin_deg, (0.0, 0.0, 1.0))
    labels = [f"tilt +{tilt_deg:.0f}° (A)", f"tilt −{tilt_deg:.0f}° (A)", f"tilt +{tilt_deg:.0f}° (B)",
              f"tilt −{tilt_deg:.0f}° (B)", f"spin +{spin_deg:.0f}°", f"spin −{spin_deg:.0f}°"]
    return out + [np.asarray(reference_quat, float)], labels + ["back to start"]


def tip_deviation(flange_pose, tcp_translation, pivot):
    """Applied-TCP tip position minus the start point: (total, lateral, vertical) in metres."""
    tip = np.asarray(flange_pose[:3]) + Rot.from_quat(flange_pose[3:7]).apply(tcp_translation)
    d = tip - np.asarray(pivot)
    return float(np.linalg.norm(d)), float(np.linalg.norm(d[:2])), float(d[2])


class RotationCheckMixin:

    def _init_rotation_check(self, mname):
        p = self.declare_parameter
        p('rotation_check_tcp_frame', 'arm_tcp')
        p('rotation_check_tilt_deg', 20.0)
        p('rotation_check_spin_deg', 45.0)
        p('rotation_check_standoff_m', 0.002)
        p('rotation_check_clearance_m', 0.02)
        p('rotation_check_limit_mm', 1.0)
        self.rc = dict(pivot=None, reference=None, tcp=None, tcp_frame=None, index=0, targets=None,
                       labels=None, touches=[])
        from std_srvs.srv import Trigger
        for name, cb in (('rotation_check_start', self.rc_start_cb), ('rotation_check_next', self.rc_next_cb),
                         ('rotation_check_capture', self.rc_capture_cb), ('rotation_check_reset', self.rc_reset_cb)):
            self.create_service(Trigger, f'{mname}/{name}', cb, callback_group=self._callback_group)

    def _rc_param(self, name):
        return self.get_parameter('rotation_check_' + name).value

    def _rotation_check_status(self):
        rc = self.rc
        touches = rc['touches']
        limit = float(self._rc_param('limit_mm'))
        summary = None
        rotated = [t for t in touches if t['index'] > 0]
        if rotated:
            tot = np.array([t['total_mm'] for t in rotated])
            summary = dict(max_mm=float(tot.max()), rms_mm=float(np.sqrt(np.mean(tot ** 2))),
                           count=len(rotated), limit_mm=limit, passed=bool(tot.max() <= limit))
        total = 0 if rc['targets'] is None else len(rc['targets'])
        return dict(
            started=rc['pivot'] is not None,
            pivot=None if rc['pivot'] is None else [float(v) for v in rc['pivot']],
            tcp_frame=rc['tcp_frame'] or str(self._rc_param('tcp_frame')),
            tcp_translation_mm=None if rc['tcp'] is None else [float(v) * 1e3 for v in rc['tcp']],
            next_index=rc['index'], total=total,
            next_label=(rc['labels'][rc['index']] if rc['labels'] and rc['index'] < total else None),
            standoff_mm=float(self._rc_param('standoff_m')) * 1e3,
            touches=touches, summary=summary,
            motion=dict(self.motion.state),
        )

    def _rc_tcp(self):
        frame = str(self._rc_param('tcp_frame'))
        tf = self.tf_buffer.lookup_transform(self.robot_flange_frame, frame, rclpy.time.Time(),
                                             Duration(seconds=0.2))
        return frame, _pose(tf.transform)[:3]

    def rc_start_cb(self, req, resp):
        if self.motion.active:
            resp.success, resp.message = False, 'A robot motion is running; stop it first.'
            return resp
        try:
            frame, tcp = self._rc_tcp()
            aggregate = self._capture_burst()
        except (TransformException, ValueError) as ex:
            resp.success, resp.message = False, f'Could not read the applied TCP / flange pose: {ex}'
            return resp
        flange = np.asarray(aggregate['pose'], float)
        pivot = flange[:3] + Rot.from_quat(flange[3:7]).apply(tcp)
        targets, labels = rotation_targets(flange[3:7], float(self._rc_param('tilt_deg')),
                                           float(self._rc_param('spin_deg')))
        self.rc.update(pivot=pivot, reference=flange[3:7], tcp=tcp, tcp_frame=frame, index=0,
                       targets=targets, labels=labels,
                       touches=[dict(index=0, label='start', total_mm=0.0, lateral_mm=0.0, vertical_mm=0.0)])
        self._publish_status()
        resp.success = True
        resp.message = (f'Start point stored at [{pivot[0] * 1e3:.1f}, {pivot[1] * 1e3:.1f}, {pivot[2] * 1e3:.1f}] mm '
                        f'(TCP {frame}). Press "Rotate to next pose"; the tip should stop '
                        f'{float(self._rc_param("standoff_m")) * 1e3:.0f} mm right above this point.')
        return resp

    def rc_next_cb(self, req, resp):
        rc = self.rc
        if rc['pivot'] is None:
            resp.success, resp.message = False, 'Put the tip on a point and press "Set start point" first.'
            return resp
        if self.motion.active:
            resp.success, resp.message = False, 'A motion is already running.'
            return resp
        if rc['index'] >= len(rc['targets']):
            resp.success, resp.message = False, 'All poses done. Reset to repeat.'
            return resp
        try:
            flange = _pose(self.tf_buffer.lookup_transform(self.robot_base_frame, self.robot_flange_frame,
                                                           rclpy.time.Time(), Duration(seconds=0.2)).transform)
            frame, tcp = self._rc_tcp()
        except TransformException as exc:
            resp.success, resp.message = False, f'Flange/TCP pose unavailable: {exc}'
            return resp
        if np.linalg.norm(tcp - rc['tcp']) > 1e-4:
            resp.success, resp.message = False, 'The applied TCP changed since the start point; reset the check.'
            return resp
        index = rc['index']

        def done(ok):
            if ok:
                rc['index'] = index + 1
            self._publish_status()

        self.motion.run_step(flange, rc['targets'][index], tcp, rc['pivot'], (0.0, 0.0, 1.0),
                             float(self._rc_param('clearance_m')), float(self._rc_param('standoff_m')), done)
        self._publish_status()
        resp.success = True
        resp.message = (f'Pose {index + 1}/{len(rc["targets"])} ({rc["labels"][index]}): lift, rotate about the '
                        'tip, descend. Keep the E-stop at hand.')
        return resp

    def rc_capture_cb(self, req, resp):
        rc = self.rc
        if rc['pivot'] is None:
            resp.success, resp.message = False, 'Set the start point first.'
            return resp
        if self.motion.active:
            resp.success, resp.message = False, 'Wait until the motion has finished.'
            return resp
        try:
            aggregate = self._capture_burst()
        except (TransformException, ValueError) as ex:
            resp.success, resp.message = False, str(ex)
            return resp
        total, lateral, vertical = tip_deviation(aggregate['pose'], rc['tcp'], rc['pivot'])
        done = rc['index']
        label = rc['labels'][done - 1] if done > 0 else 'start'
        rc['touches'] = [t for t in rc['touches'] if t['index'] != done] + [dict(
            index=done, label=label, total_mm=total * 1e3, lateral_mm=lateral * 1e3, vertical_mm=vertical * 1e3)]
        rc['touches'].sort(key=lambda t: t['index'])
        self._publish_status()
        resp.success = True
        resp.message = (f'{label}: tip is {total * 1e3:.2f} mm from the start point '
                        f'(sideways {lateral * 1e3:.2f}, vertical {vertical * 1e3:+.2f} mm).')
        return resp

    def rc_reset_cb(self, req, resp):
        if self.motion.active:
            self.motion.stop()
        self.rc.update(pivot=None, reference=None, tcp=None, tcp_frame=None, index=0, targets=None,
                       labels=None, touches=[])
        self._publish_status()
        resp.success, resp.message = True, 'Rotation check cleared.'
        return resp

