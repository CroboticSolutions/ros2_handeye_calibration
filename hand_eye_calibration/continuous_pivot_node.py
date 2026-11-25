"""Continuous pivot recording for the tool TCP collector (mixin of PivotCollector).

The operator seats the tool tip in a conical cup, presses Start and turns the
tool by hand (Piper teach mode) while the tip stays in the cup. The collector
reads the flange pose from TF at ``continuous_rate_hz`` and keeps a pose for the
tip round when

* the wrist turns slower than ``continuous_max_rot_speed_deg_s`` (joint states
  and TF of a fast-moving arm disagree in time, which shows up as tip error), and
* its orientation is at least ``continuous_min_step_deg`` from every kept pose,
  so lingering in one attitude does not weight the fit.

While recording the status only carries counts, rotation spans and a quick
linear estimate; the full robust fit (RANSAC + Huber + bootstrap, ~1 s per 50
poses) runs once on Stop. The kept poses are ordinary tip samples: Save, Apply
and the dataset work as for touches on the spike.
"""

import time
from datetime import datetime, timezone

import numpy as np
import rclpy
from rclpy.time import Duration
from scipy.spatial.transform import Rotation as Rot
from std_srvs.srv import Trigger
from tf2_ros import TransformException

from .pivot_backend import PivotCalibrationBackend
from . import tcp_quality

TIP_ROUND = "tip"


def rotation_step_deg(pose, kept_quats):
    """Smallest rotation (deg) between ``pose`` and the kept orientations; inf if none."""
    if len(kept_quats) == 0:
        return float("inf")
    candidate = Rot.from_quat(pose[3:])
    return float(np.degrees((Rot.from_quat(np.asarray(kept_quats)).inv() * candidate).magnitude()).min())


def quick_estimate(samples):
    """Linear pivot fit for the live readout: tip (m, flange frame) and RMS (m)."""
    if len(samples) < PivotCalibrationBackend.MIN_SAMPLES:
        return None
    solution, _rank, condition = PivotCalibrationBackend._linear_fit(samples)
    residuals = PivotCalibrationBackend._residual_vectors(samples, solution)
    rms = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    return {"tip": [float(v) for v in solution[:3]], "rms_m": rms, "condition_number": condition}


class ContinuousPivotMixin:
    def _init_continuous(self, mname):
        p = self.declare_parameter
        p('continuous_rate_hz', 20.0)
        p('continuous_min_step_deg', 3.0)
        p('continuous_max_samples', 120)
        p('continuous_max_rot_speed_deg_s', 30.0)
        p('continuous_status_period_s', 0.5)
        self.cp = self._cp_empty(active=False)
        self.create_service(Trigger, f'{mname}/continuous_start', self.cp_start_cb,
                            callback_group=self._callback_group)
        self.create_service(Trigger, f'{mname}/continuous_stop', self.cp_stop_cb,
                            callback_group=self._callback_group)
        rate = max(1.0, float(self._cp_param('rate_hz')))
        # Same mutually exclusive group as captures/saves: the tick never races a
        # service that changes the sample lists.
        self.create_timer(1.0 / rate, self._cp_tick, callback_group=self._callback_group)

    def _cp_param(self, name):
        return self.get_parameter('continuous_' + name).value

    @staticmethod
    def _cp_empty(active):
        return dict(active=active, started=time.monotonic() if active else None, duration_s=0.0,
                    seen=0, kept=0, too_fast=0, too_similar=0, stale=0, prev=None,
                    last_status=0.0, message=None)

    def _continuous_status(self):
        cp = self.cp
        samples = self.samples[TIP_ROUND]
        duration = (time.monotonic() - cp['started']) if cp['active'] else cp['duration_s']
        status = {k: cp[k] for k in ('active', 'seen', 'kept', 'too_fast', 'too_similar', 'stale', 'message')}
        status.update(
            duration_s=round(float(duration or 0.0), 1),
            max_samples=int(self._cp_param('max_samples')),
            min_step_deg=float(self._cp_param('min_step_deg')),
            max_rot_speed_deg_s=float(self._cp_param('max_rot_speed_deg_s')),
        )
        if cp['active']:
            status['rotation_spans_deg'] = tcp_quality.rotation_spans_deg(samples)
            try:
                status['estimate'] = quick_estimate(samples)
            except (ValueError, np.linalg.LinAlgError):
                status['estimate'] = None
        return status

    def cp_start_cb(self, req, resp):
        if self.active_round != TIP_ROUND:
            resp.success, resp.message = False, 'Continuous recording works in the tool tip round.'
            return resp
        if self.cp['active']:
            resp.success, resp.message = True, 'Already recording.'
            return resp
        # A fresh run: the cup is one fixed point, so earlier touches elsewhere
        # (e.g. on the spike) must not mix into the fit.
        for key in (TIP_ROUND, 'validation'):
            self.samples[key] = []
            self.sample_metadata[key] = []
        self.axis_result = None
        self.reorient_index = 0
        self.cp = self._cp_empty(active=True)
        self._publish_status()
        resp.success = True
        resp.message = ('Recording: keep the tip in the cup and turn the tool slowly in all directions '
                        '(tilt both ways, then spin). Press Stop when the spans are green.')
        return resp

    def cp_stop_cb(self, req, resp):
        if not self.cp['active']:
            resp.success, resp.message = True, 'Not recording.'
            return resp
        self._cp_stop('Stopped.')
        tip_pivot, _status = self._publish_status()
        n = len(self.samples[TIP_ROUND])
        resp.success = True
        if tip_pivot is None:
            resp.message = (f'Stopped with {n} poses; need at least {PivotCalibrationBackend.MIN_SAMPLES} '
                            'for a fit.')
        else:
            t = np.asarray(tip_pivot['tcp_translation']) * 1000
            resp.message = (f'Stopped: {n} poses, tip [{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}] mm, '
                            f'RMS {tip_pivot["rms_residual_m"] * 1000:.2f} mm, '
                            f'{tip_pivot.get("outlier_count", 0)} outliers.')
        return resp

    def _cp_stop(self, message):
        cp = self.cp
        cp['duration_s'] = time.monotonic() - cp['started']
        cp['active'] = False
        cp['message'] = message

    def _cp_tick(self):
        cp = self.cp
        if not cp['active']:
            return
        if self.active_round != TIP_ROUND:
            self._cp_stop('Stopped: the active round changed.')
            self._publish_status()
            return
        try:
            flange = self.tf_buffer.lookup_transform(
                self.robot_base_frame, self.robot_flange_frame, rclpy.time.Time(), Duration(seconds=0.0))
        except TransformException:
            cp['stale'] += 1
            return
        stamp = flange.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if cp['prev'] is not None and stamp_ns == cp['prev'][0]:
            return
        age_s = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
        if stamp_ns > 0 and (age_s < -0.02 or age_s > self.max_tf_age_s):
            cp['stale'] += 1
            return
        t = flange.transform
        pose = [t.translation.x, t.translation.y, t.translation.z,
                t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
        cp['seen'] += 1
        prev, cp['prev'] = cp['prev'], (stamp_ns, pose)
        if prev is None:
            return
        dt = (stamp_ns - prev[0]) / 1e9
        if dt <= 0:
            return
        speed = float(np.degrees((Rot.from_quat(prev[1][3:]).inv() * Rot.from_quat(pose[3:])).magnitude())) / dt
        if speed > float(self._cp_param('max_rot_speed_deg_s')):
            cp['too_fast'] += 1
        else:
            kept_quats = [s[3:] for s in self.samples[TIP_ROUND]]
            step = rotation_step_deg(pose, kept_quats)
            if step < float(self._cp_param('min_step_deg')):
                cp['too_similar'] += 1
            else:
                self.samples[TIP_ROUND].append(pose)
                self.sample_metadata[TIP_ROUND].append({
                    'captured_at': datetime.now(timezone.utc).isoformat(),
                    'source': 'continuous',
                    'rotation_speed_deg_s': speed,
                    'nearest_orientation_deg': None if not np.isfinite(step) else step,
                    'joints': self._current_joints(),
                })
                cp['kept'] += 1
        if len(self.samples[TIP_ROUND]) >= int(self._cp_param('max_samples')):
            self._cp_stop(f'Stopped: reached {len(self.samples[TIP_ROUND])} poses.')
            self._publish_status()
            return
        now = time.monotonic()
        if now - cp['last_status'] >= float(self._cp_param('status_period_s')):
            cp['last_status'] = now
            self._publish_status()
