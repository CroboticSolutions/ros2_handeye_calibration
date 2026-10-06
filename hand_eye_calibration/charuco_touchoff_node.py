"""ChArUco touch-off round for the tool TCP collector (PivotCollector mixin).

Flow: detect the board once from a pose where the camera sees it (the board
must not move afterwards), select a corner ID, put the wire tip on that corner
with the torch perpendicular to the board, capture; repeat for >= 4 IDs; save.
Math lives in charuco_touchoff.py.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Duration
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Int32
from std_srvs.srv import Trigger
from tf2_ros import TransformException

from . import charuco_touchoff as cto
from . import tcp_quality
from .pivot_backend import PivotCalibrationBackend

CHARUCO_ROUND = "charuco"


def _matrix(tf):
    T = np.eye(4)
    r, t = tf.rotation, tf.translation
    T[:3, :3] = Rot.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def _gray(msg):
    enc = msg.encoding.lower()
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8uc1": 1}.get(enc)
    if ch is None:
        raise ValueError(f"unsupported image encoding {msg.encoding}")
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    if ch == 1:
        return a.reshape(msg.height, msg.width).copy()
    a = a.reshape(msg.height, msg.width, ch)[..., :3].astype(np.float32)
    w = [0.299, 0.587, 0.114] if enc.startswith("rgb") else [0.114, 0.587, 0.299]
    return (a @ np.asarray(w, np.float32)).astype(np.uint8)


class CharucoTouchoffMixin:

    def _init_charuco(self, mname):
        p = self.declare_parameter
        p('charuco_squares_x', 6)
        p('charuco_squares_y', 8)
        p('charuco_square_m', 0.025)
        p('charuco_marker_m', 0.018)
        p('charuco_dictionary', 'DICT_4X4_100')
        p('charuco_image_topic', '/camera/color/image_raw')
        p('charuco_info_topic', '/camera/color/camera_info')
        p('charuco_detect_frames', 15)
        # Frame of the currently applied TCP, for live perpendicularity/distance hints.
        p('charuco_live_tcp_frame', 'arm_tcp')
        for key, value in cto.DEFAULT_LIMITS.items():
            p('charuco_accept_' + key, value)
        g = self.get_parameter
        self._cto_board = cto.make_board(g('charuco_squares_x').value, g('charuco_squares_y').value,
                                         g('charuco_square_m').value, g('charuco_marker_m').value,
                                         g('charuco_dictionary').value)
        self._cto_board_spec = dict(squares_x=int(g('charuco_squares_x').value),
                                    squares_y=int(g('charuco_squares_y').value),
                                    square_m=float(g('charuco_square_m').value),
                                    marker_m=float(g('charuco_marker_m').value),
                                    dictionary=str(g('charuco_dictionary').value))
        self._cto_live_frame = str(g('charuco_live_tcp_frame').value)
        self._cto_limits = {k: type(v)(g('charuco_accept_' + k).value) for k, v in cto.DEFAULT_LIMITS.items()}
        self._cto_frames_needed = int(g('charuco_detect_frames').value)
        self._cto_lock = threading.Lock()
        self._cto_frames = deque(maxlen=self._cto_frames_needed)
        self._cto_collect = False
        self._cto_info = None
        self.cto = dict(detection=None, targets=None, normal=None, selected=None,
                        samples=[], metadata=[], live=None)
        sensor_cb = ReentrantCallbackGroup()
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(Image, str(g('charuco_image_topic').value), self._cto_image_cb,
                                 qos_profile_sensor_data, callback_group=sensor_cb)
        self.create_subscription(CameraInfo, str(g('charuco_info_topic').value),
                                 lambda m: setattr(self, '_cto_info', m), qos_profile_sensor_data,
                                 callback_group=sensor_cb)
        self.create_subscription(Int32, mname + '/from_gui/charuco_select', self._cto_select_cb, 10,
                                 callback_group=sensor_cb)
        for name, cb in (('select_round_charuco', self.cto_select_round_cb), ('charuco_detect', self.cto_detect_cb),
                         ('charuco_capture', self.cto_capture_cb), ('charuco_remove_last', self.cto_remove_last_cb),
                         ('charuco_reset', self.cto_reset_cb), ('charuco_save', self.cto_save_cb)):
            self.create_service(Trigger, f'{mname}/{name}', cb, callback_group=self._callback_group)
        self.create_timer(0.5, self._cto_live_timer, callback_group=sensor_cb)

    # ---------------------------------------------------------------- inputs
    def _cto_image_cb(self, msg):
        if self._cto_collect:
            with self._cto_lock:
                self._cto_frames.append(msg)

    def _cto_select_cb(self, msg):
        tg = self.cto['targets']
        if tg is None or not 0 <= msg.data < len(tg['ids']):
            return
        self.cto['selected'] = int(msg.data)
        self._publish_status()

    # ---------------------------------------------------------------- status
    def _cto_result(self):
        samples = self.cto['samples']
        if not samples:
            return None, None
        result = cto.solve(samples, normal_up=self.cto['normal'])
        return result, cto.acceptance(result, self._cto_limits)

    def _charuco_status(self):
        tg = self.cto['targets']
        result, gate = self._cto_result()
        captured = {s['id']: i for i, s in enumerate(self.cto['samples'])}
        return dict(
            board=self._cto_board_spec,
            detection=self.cto['detection'],
            corners=None if tg is None else [
                dict(id=i, x_mm=float(xy[0]), y_mm=float(xy[1]),
                     base=[float(v) for v in p], captured=i in captured,
                     residual_mm=(None if result is None or i not in captured
                                  else result['residuals_m'][captured[i]] * 1e3))
                for i, xy, p in zip(tg['ids'], tg['points_board_mm'], tg['points_base'])],
            selected_id=self.cto['selected'],
            sample_count=len(self.cto['samples']),
            distinct_ids=len(captured),
            min_points=self._cto_limits['min_points'],
            result=result,
            acceptance=gate,
            live=self.cto['live'],
        )

    def _cto_live_timer(self):
        if self.active_round != CHARUCO_ROUND or self.cto['targets'] is None:
            return
        live = None
        try:
            T = _matrix(self.tf_buffer.lookup_transform(
                self.robot_base_frame, self._cto_live_frame, rclpy.time.Time(),
                Duration(seconds=0.05)).transform)
            n = self.cto['normal']
            F = _matrix(self.tf_buffer.lookup_transform(
                self.robot_base_frame, self.robot_flange_frame, rclpy.time.Time(),
                Duration(seconds=0.05)).transform)
            # Tool axis = the TCP frame axis closest to the flange->tip direction
            # (the model xacro uses +X along the wire, pivot/axis saves use +Z).
            out = T[:3, 3] - F[:3, 3]
            axes = np.concatenate([T[:3, :3], -T[:3, :3]], axis=1)
            axis = axes[:, int(np.argmax(axes.T @ out))]
            tilt = float(np.degrees(np.arccos(np.clip(axis @ (-n), -1, 1))))
            live = dict(frame=self._cto_live_frame, tilt_deg=tilt)
            sel = self.cto['selected']
            if sel is not None:
                d = T[:3, 3] - self.cto['targets']['points_base'][sel]
                h = float(d @ n)
                live.update(height_mm=h * 1e3, lateral_mm=float(np.linalg.norm(d - h * n) * 1e3))
        except TransformException:
            live = dict(frame=self._cto_live_frame, error='no TF for the live TCP frame')
        self.cto['live'] = live
        self._publish_status()

    # ---------------------------------------------------------------- services
    def cto_select_round_cb(self, req, resp):
        self.active_round = CHARUCO_ROUND
        self._publish_status()
        resp.success, resp.message = True, "Active round: ChArUco touch-off."
        return resp

    def cto_detect_cb(self, req, resp):
        info = self._cto_info
        if info is None:
            resp.success, resp.message = False, "No camera_info yet; is the camera running?"
            return resp
        with self._cto_lock:
            self._cto_frames.clear()
        self._cto_collect = True
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            with self._cto_lock:
                if len(self._cto_frames) >= self._cto_frames_needed:
                    break
            time.sleep(0.02)
        self._cto_collect = False
        with self._cto_lock:
            frames = list(self._cto_frames)
        if not frames:
            resp.success, resp.message = False, "No camera images arrived."
            return resp
        try:
            K = np.asarray(info.k, float).reshape(3, 3)
            D = np.asarray(info.d, float)
            T_cam_board, det = cto.board_pose_from_images(self._cto_board, [_gray(m) for m in frames], K, D)
            last = frames[-1]
            T_base_cam = _matrix(self.tf_buffer.lookup_transform(
                self.robot_base_frame, last.header.frame_id, rclpy.time.Time.from_msg(last.header.stamp),
                Duration(seconds=0.5)).transform)
        except (ValueError, TransformException) as ex:
            resp.success, resp.message = False, f"Board detection failed: {ex}"
            return resp
        if det['translation_spread_mm'] > 2.0:
            resp.success = False
            resp.message = (f"Board pose moved {det['translation_spread_mm']:.1f} mm between frames; "
                            "keep the robot still and detect again.")
            return resp
        T_base_board = T_base_cam @ T_cam_board
        tg = cto.board_targets(self._cto_board, T_base_board)
        n = cto.orient_normal_up(tg['normal_base'], T_base_cam[:3, 3], T_base_board[:3, 3])
        tilt = float(np.degrees(np.arccos(abs(n @ np.array([0.0, 0.0, 1.0])))))
        self.cto.update(targets=tg, normal=n, selected=None, samples=[], metadata=[], detection=dict(
            det, camera_frame=last.header.frame_id, board_pose_base=T_base_board.tolist(),
            normal_base=[float(v) for v in n], board_tilt_vs_base_z_deg=tilt,
            distance_m=float(np.linalg.norm(T_cam_board[:3, 3])),
            detected_at=datetime.now(timezone.utc).isoformat()))
        self._publish_status()
        resp.success = True
        resp.message = (f"Board found in {det['frames']} frames, {len(det['corner_ids_seen'])} corners, "
                        f"reprojection {det['reprojection_px']:.2f} px, tilt vs base {tilt:.1f}°. "
                        "Do not move the board from now on; previous touches were cleared.")
        return resp

    def cto_capture_cb(self, req, resp):
        sel = self.cto['selected']
        if self.cto['targets'] is None:
            resp.success, resp.message = False, "Detect the board first."
            return resp
        if sel is None:
            resp.success, resp.message = False, "Select the corner ID you are touching."
            return resp
        try:
            aggregate = self._capture_burst()
        except (TransformException, ValueError) as ex:
            resp.success, resp.message = False, str(ex)
            return resp
        target = [float(v) for v in self.cto['targets']['points_base'][sel]]
        sample = dict(id=sel, pose=aggregate['pose'], target=target)
        meta = dict(captured_at=datetime.now(timezone.utc).isoformat(), id=sel,
                    burst_sample_count=aggregate['sample_count'],
                    translation_p95_m=aggregate['translation_p95_m'],
                    rotation_p95_deg=aggregate['rotation_p95_deg'], joints=self._current_joints(),
                    live=self.cto['live'])
        replaced = [i for i, s in enumerate(self.cto['samples']) if s['id'] == sel]
        for i in reversed(replaced):
            del self.cto['samples'][i]
            del self.cto['metadata'][i]
        self.cto['samples'].append(sample)
        self.cto['metadata'].append(meta)
        _, status = self._publish_status()
        c = status['charuco']
        r = c['result']
        resp.success = True
        resp.message = (f"{'Replaced' if replaced else 'Captured'} ID {sel}: {c['distinct_ids']} IDs. "
                        f"TCP [{r['tcp_translation'][0] * 1e3:.1f}, {r['tcp_translation'][1] * 1e3:.1f}, "
                        f"{r['tcp_translation'][2] * 1e3:.1f}] mm, RMS {r['rms_residual_m'] * 1e3:.2f} mm"
                        + (f", axis spread {r['axis_spread_deg']:.2f}°" if 'axis_spread_deg' in r else ''))
        return resp

    def cto_remove_last_cb(self, req, resp):
        if not self.cto['samples']:
            resp.success, resp.message = False, "No touches to remove."
            return resp
        s = self.cto['samples'].pop()
        self.cto['metadata'].pop()
        self._publish_status()
        resp.success, resp.message = True, f"Removed touch on ID {s['id']}."
        return resp

    def cto_reset_cb(self, req, resp):
        self.cto.update(samples=[], metadata=[])
        self._publish_status()
        resp.success, resp.message = True, "Cleared all ChArUco touches (board detection kept)."
        return resp

    def cto_save_cb(self, req, resp):
        result, gate = self._cto_result()
        if result is None:
            resp.success, resp.message = False, "No touches captured."
            return resp
        axis = result.get('axis_dir')
        frame = tcp_quality.bend_plane_frame(axis) if (axis and self.roll_convention == 'bend_plane') else None
        if frame is None:
            frame = PivotCalibrationBackend.frame_from_axis(np.asarray(axis)) if axis else None
        qx, qy, qz, qw = frame['quaternion'] if frame else (0.0, 0.0, 0.0, 1.0)
        t = result['tcp_translation']
        data = {
            'parent_frame': self.robot_flange_frame,
            'tcp_name': self.tcp_name,
            'robot_base_frame': self.robot_base_frame,
            'robot_flange_frame': self.robot_flange_frame,
            'robot_firmware_version': self.robot_firmware_version,
            'calibration_mode': 'charuco_touchoff',
            'sample_count': len(self.cto['samples']),
            'rms_residual_m': result['rms_residual_m'],
            'max_residual_m': result['max_residual_m'],
            'per_sample_residuals_m': result['residuals_m'],
            'charuco_touchoff': {
                'board': self._cto_board_spec,
                'detection': self.cto['detection'],
                'result': result,
                'roll_convention': frame.get('roll_convention', 'frame_from_axis') if frame else None,
                'cad_axis_angle_deg': self.cad_axis_angle_deg,
                'cad_axis_deviation_deg': (tcp_quality.cad_axis_deviation_deg(axis, self.cad_axis_angle_deg)
                                           if axis else None),
            },
            'raw_samples': {'charuco_touches': self.cto['samples'],
                            'charuco_capture_metadata': self.cto['metadata'],
                            'joint_names': list(self.motion.names or [])},
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'transform': {'tx': t[0], 'ty': t[1], 'tz': t[2], 'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw},
            'acceptance': gate,
        }
        cal_file = os.path.expanduser(str(self.get_parameter('calibration_file').value))
        try:
            data['dataset_path'] = self._write_dataset(data)
            accepted = gate['passed'] or self.acceptance_mode == 'warn'
            target = cal_file if accepted else cal_file + '.rejected.yaml'
            self._write_yaml_atomic(target, data, keep_previous=accepted)
        except Exception as ex:  # noqa: BLE001 - report any write failure to the GUI
            resp.success, resp.message = False, f"Save failed: {ex}"
            return resp
        if not accepted:
            resp.success = False
            resp.message = f"Acceptance failed; active TCP unchanged. {gate['summary']} Candidate: {target}"
            return resp
        resp.success = True
        resp.message = (f"Saved to {cal_file}: TCP [{t[0] * 1e3:.1f}, {t[1] * 1e3:.1f}, {t[2] * 1e3:.1f}] mm, "
                        f"RMS {result['rms_residual_m'] * 1e3:.2f} mm from {result['distinct_ids']} IDs"
                        + ('' if gate['passed'] else f" (WARNING: {gate['summary']})") + ". Press Apply.")
        return resp
