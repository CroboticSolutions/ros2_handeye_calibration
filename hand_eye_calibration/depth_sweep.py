"""Automatic depth calibration sweep: the robot views the still ChArUco board
from several incidence angles and distances and stores, per view, the median
depth image with the board pose seen in colour.

Runs inside the hand-eye calibration node and reuses AutomaticCalibration for
motion (MoveIt-planned, collision-checked trajectories, operator heartbeat,
Stop). Needs the applied hand-eye calibration (camera TF) to aim the camera;
the depth model itself is fitted by the GUI backend from the stored views
(arm_api2_perception.depth_board_analysis).
"""

from __future__ import annotations

import json
import os
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

INCIDENCES_DEG = (15.0, 25.0, 35.0, 45.0, 55.0)
AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0)
DISTANCES_M = (0.32, 0.40, 0.48)
FRAMES_PER_VIEW = 15
IK_POSITION_TOL_M = 0.015
IK_AIM_TOL_DEG = 3.0


# ------------------------------------------------------------------ planning
def look_at_targets(base_board: np.ndarray, board_center: np.ndarray, camera_now: np.ndarray,
                    incidences=INCIDENCES_DEG, azimuths=AZIMUTHS_DEG, distances=DISTANCES_M):
    """Camera positions around the board centre (base frame).

    The azimuth is measured about the board normal from the current viewing
    direction, so azimuth 0 keeps the camera on its present side.
    """
    center = (base_board @ np.r_[board_center, 1.0])[:3]
    normal = base_board[:3, 2].copy()
    if normal @ (camera_now[:3, 3] - center) < 0:
        normal = -normal
    view = camera_now[:3, 3] - center
    side = view - normal * (view @ normal)
    if np.linalg.norm(side) < 1e-6:
        side = base_board[:3, 0].copy()
    side /= np.linalg.norm(side)
    other = np.cross(normal, side)
    targets = []
    for inc in incidences:
        for az in azimuths if inc > 0 else (0.0,):
            a = np.radians(az)
            lateral = np.cos(a) * side + np.sin(a) * other
            direction = np.cos(np.radians(inc)) * normal + np.sin(np.radians(inc)) * lateral
            for d in distances:
                targets.append({"incidence_deg": float(inc), "azimuth_deg": float(az), "distance_m": float(d),
                                "position": center + d * direction, "look_at": center})
    return targets


def solve_camera_ik(camera_fk, q0, position, look_at, lower, upper, seeds=()):
    """Joints that put the camera at ``position`` looking at ``look_at``.

    Roll about the optical axis is free (the board is framed separately).
    Returns (q, position_error_m, aim_error_deg) of the best solution or None.
    """
    lo = np.asarray(lower) + 0.02
    hi = np.asarray(upper) - 0.02

    def residual(q):
        cam = camera_fk(q)
        want = look_at - cam[:3, 3]
        want /= max(np.linalg.norm(want), 1e-9)
        return np.r_[cam[:3, 3] - position, 0.3 * np.cross(cam[:3, 2], want), 1e-3 * (q - q0)]

    best = None
    for seed in (q0, *seeds):
        x0 = np.clip(seed, lo, hi)
        fit = least_squares(residual, x0, bounds=(lo, hi), diff_step=1e-4, xtol=1e-8, max_nfev=150)
        cam = camera_fk(fit.x)
        perr = float(np.linalg.norm(cam[:3, 3] - position))
        aim = look_at - cam[:3, 3]
        aerr = float(np.degrees(np.arccos(np.clip(cam[:3, 2] @ aim / np.linalg.norm(aim), -1, 1))))
        if perr <= IK_POSITION_TOL_M and aerr <= IK_AIM_TOL_DEG:
            cost = float(np.max(np.abs(fit.x - q0)))
            if best is None or cost < best[3]:
                best = (fit.x, perr, aerr, cost)
    return None if best is None else best[:3]


def select_views(candidates, count):
    """Spread the views over incidence first, then azimuth and distance."""
    chosen, used = [], set()
    by_inc = {}
    for c in candidates:
        by_inc.setdefault(c["incidence_deg"], []).append(c)
    rounds = 0
    while len(chosen) < count and rounds < 50:
        rounds += 1
        added = False
        for inc in sorted(by_inc):
            pool = [c for c in by_inc[inc] if id(c) not in used]
            if not pool:
                continue
            # Prefer an azimuth and distance not yet used at this incidence.
            seen = [(c["azimuth_deg"], c["distance_m"]) for c in chosen if c["incidence_deg"] == inc]
            pool.sort(key=lambda c: (sum(c["azimuth_deg"] == a for a, _ in seen),
                                     sum(c["distance_m"] == d for _, d in seen), c["cost"]))
            chosen.append(pool[0])
            used.add(id(pool[0]))
            added = True
            if len(chosen) >= count:
                break
        if not added:
            break
    return chosen


def order_by_travel(views, q_start):
    """Greedy nearest-neighbour order in joint space."""
    rest, out, q = list(views), [], np.asarray(q_start)
    while rest:
        nxt = min(rest, key=lambda v: float(np.max(np.abs(v["q"] - q))))
        rest.remove(nxt)
        out.append(nxt)
        q = nxt["q"]
    return out


# ------------------------------------------------------------------ session
class DepthSweepSession:
    def __init__(self, runner, root):
        from .visibility import BoardFraming, CameraKinematics
        self.r = runner
        self.n = runner.node
        info = runner.camera_info
        if info is None:
            raise ValueError('Camera info is missing; no motion was sent.')
        self.framing = BoardFraming(runner.board_spec, info)
        # observed_board() checks framing in the optical frame of the detector.
        runner.framing = self.framing
        runner.optical_tracking = np.eye(4)
        n = self.n
        mount = n.tf_buffer.lookup_transform(n.robot_effector_frame, n.tracking_base_frame,
                                             __import__('rclpy').time.Time())
        from .automatic_calibration import matrix
        effector_camera = matrix(mount.transform)
        self.fk = CameraKinematics(root, n.robot_base_frame, n.robot_effector_frame, runner.names,
                                   np.linalg.inv(effector_camera))
        self.depth_topic = str(n.get_parameter('depth_topic').value)
        self.native_info_topic = str(n.get_parameter('depth_info_topic').value)
        self.views_wanted = int(n.get_parameter('depth_sweep_views').value)
        self.frames = []
        self.native_info = None
        self.lock = threading.Lock()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.directory = os.path.join(os.path.expanduser(str(n.get_parameter('depth_dataset_dir').value)), stamp)

    # depth subscription only while sweeping
    def _depth(self, msg):
        with self.lock:
            self.frames.append(msg)
            del self.frames[:-60]

    def _native_info(self, msg):
        self.native_info = msg

    def plan(self, q0):
        board_cam = self.r.observed_board()
        base_board = self.fk.camera(q0) @ board_cam
        targets = look_at_targets(base_board, self.framing.center, self.fk.camera(q0))
        rng = np.random.default_rng(0)
        candidates = []
        for t in targets:
            self.r.check()
            seeds = [np.clip(q0 + rng.normal(0, 0.35, q0.size), self.r.lower, self.r.upper) for _ in range(2)]
            sol = solve_camera_ik(self.fk.camera, q0, t["position"], t["look_at"], self.r.lower, self.r.upper, seeds)
            if sol is None:
                continue
            q, perr, aerr = sol
            if not self.framing.contains(np.linalg.inv(self.fk.camera(q)) @ base_board, margin=.05):
                continue
            candidates.append({**t, "q": q, "cost": float(np.max(np.abs(q - q0))),
                               "position": t["position"].tolist(), "look_at": t["look_at"].tolist()})
        views = order_by_travel(select_views(candidates, self.views_wanted), q0)
        return views, base_board, len(targets), len(candidates)

    def capture(self, view_index, view):
        """Median depth of FRAMES_PER_VIEW frames exposed after the robot stopped."""
        after = (self.r.stationary_ns or self.n.get_clock().now().nanoseconds) + 50_000_000
        with self.lock:
            self.frames.clear()
        end = time.monotonic() + 8.0
        depth = []
        while time.monotonic() < end and len(depth) < FRAMES_PER_VIEW:
            self.r.check()
            self.r.stop_event.wait(0.05)
            with self.lock:
                fresh = [m for m in self.frames
                         if m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec > after]
                self.frames.clear()
            depth.extend(fresh)
        if len(depth) < 3:
            raise RuntimeError(f'Only {len(depth)} depth frames after the stop on {self.depth_topic}.')
        observations = [o for o in list(self.n.observations.items.values()) if o['stamp_ns'] > after]
        if not observations:
            raise RuntimeError('Board not detected at the view; sequence stopped.')
        poses = []
        for o in observations:
            p = np.eye(4)
            p[:3, :3] = Rotation.from_quat(o['pose'][3:7]).as_matrix()
            p[:3, 3] = o['pose'][:3]
            poses.append(p)
        rotation = Rotation.from_matrix([p[:3, :3] for p in poses]).mean().as_matrix()
        board = np.eye(4)
        board[:3, :3] = rotation
        board[:3, 3] = np.mean([p[:3, 3] for p in poses], axis=0)
        best = max(observations, key=lambda o: len(o['ids']))
        images = np.stack([_decode_depth(m) for m in depth])
        with np.errstate(all='ignore'):
            median = np.nanmedian(np.where(images > 0, images, np.nan), axis=0).astype(np.float32)
        os.makedirs(self.directory, exist_ok=True)
        np.savez_compressed(os.path.join(self.directory, f'view_{view_index:02d}.npz'),
                            depth_m=median, board_pose=board, image_points=np.asarray(best['image_points'], float),
                            ids=np.asarray(best['ids'], int), joints=np.asarray(self.r.joint_positions(), float),
                            frames=len(depth), detections=len(observations))
        return {'index': view_index, 'frames': len(depth), 'detections': len(observations),
                'incidence_deg': view['incidence_deg'], 'azimuth_deg': view['azimuth_deg'],
                'distance_m': view['distance_m']}

    def write_meta(self, views, captured):
        info = self.r.camera_info
        meta = {
            'created': datetime.now(timezone.utc).isoformat(),
            'depth_topic': self.depth_topic,
            'native_depth': bool(self.native_info_topic),
            'camera_frame': self.n.tracking_base_frame,
            'color': {'k': list(map(float, info.k)), 'd': list(map(float, info.d)),
                      'width': int(info.width), 'height': int(info.height)},
            'board': {k: float(v) if isinstance(v, float) else v for k, v in self.r.board_spec.items()},
            'marker_length_m': float(self.n.get_parameter('marker_length_m').value)
            if self.n.has_parameter('marker_length_m') else None,
            'planned_views': len(views),
            'views': captured,
        }
        if self.native_info is not None:
            meta['native'] = {'k': list(map(float, self.native_info.k)), 'd': list(map(float, self.native_info.d)),
                              'frame_id': self.native_info.header.frame_id}
            try:
                tf = self.n.tf_buffer.lookup_transform(self.n.tracking_base_frame, self.native_info.header.frame_id,
                                                       __import__('rclpy').time.Time())
                from .automatic_calibration import matrix
                meta['native']['color_from_native'] = matrix(tf.transform).tolist()
            except Exception as exc:  # noqa: BLE001
                meta['native']['tf_error'] = str(exc)
        with open(os.path.join(self.directory, 'run.json'), 'w') as f:
            json.dump(meta, f, indent=1)

    def run(self):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        if not self.depth_topic:
            raise ValueError('No depth_topic configured for this camera; depth calibration is unavailable.')
        n, r = self.n, self.r
        sub = n.create_subscription(Image, self.depth_topic, self._depth, qos_profile_sensor_data,
                                    callback_group=r.group)
        info_sub = None
        if self.native_info_topic:
            info_sub = n.create_subscription(CameraInfo, self.native_info_topic, self._native_info,
                                             qos_profile_sensor_data, callback_group=r.group)
        try:
            q0 = r.joint_positions()
            r.update('planning', 'Planning views around the board (uses the applied hand-eye calibration).',
                     phase='depth', mode='depth_sweep')
            views, _, tried, reachable = self.plan(q0)
            if len(views) < 3:
                raise RuntimeError(f'Only {len(views)} of {tried} views are reachable with the board framed; '
                                   'move the board closer to the robot or the camera to the board.')
            r.update('moving', f'{len(views)} views planned ({reachable}/{tried} reachable).',
                     total=len(views), pose=0, accepted=0, depth_dataset=self.directory)
            captured, skipped = [], 0
            for i, view in enumerate(views):
                r.check()
                r.update('moving', f'View {i + 1}/{len(views)}: incidence {view["incidence_deg"]:.0f}°, '
                         f'{view["distance_m"] * 1000:.0f} mm.', pose=i + 1)
                r.planned_trajectory = None
                path = r.plan_joint_path(r.joint_positions(), view['q'])
                trajectory = r.planned_trajectory
                r.planned_trajectory = None
                if path is None or trajectory is None:
                    skipped += 1
                    r.update('moving', f'View {i + 1} skipped: no collision-free path.', skipped=skipped)
                    continue
                r.move_planned(trajectory)
                r.settle(np.asarray(trajectory.points[-1].positions, float))
                r.update('capturing', f'View {i + 1}/{len(views)}: capturing depth.')
                try:
                    captured.append(self.capture(i, view))
                except RuntimeError as exc:
                    skipped += 1
                    r.update('capturing', f'View {i + 1} skipped: {exc}', skipped=skipped)
                    continue
                r.update('capturing', f'View {i + 1} captured.', accepted=len(captured))
                self.write_meta(views, captured)
            if len(captured) < 3:
                raise RuntimeError(f'Only {len(captured)} views captured; nothing to fit.')
            # Back to the start pose so the operator finds the robot where it was.
            r.planned_trajectory = None
            if r.plan_joint_path(r.joint_positions(), q0) is not None and r.planned_trajectory is not None:
                trajectory, r.planned_trajectory = r.planned_trajectory, None
                r.move_planned(trajectory)
            return captured
        finally:
            n.destroy_subscription(sub)
            if info_sub is not None:
                n.destroy_subscription(info_sub)


def _decode_depth(msg):
    dtype = np.uint16 if msg.encoding in ('16UC1', 'mono16') else np.float32
    a = np.frombuffer(bytes(msg.data), dtype).reshape(msg.height, msg.step // np.dtype(dtype).itemsize)[:, :msg.width]
    return a.astype(np.float32) * (1e-3 if dtype == np.uint16 else 1.0)


# ------------------------------------------------------------------ service
def start(runner, req, resp):
    """``hand_eye_calibration/depth_sweep_start``: same guards as automatic calibration."""
    with runner.lock:
        if not runner.node.get_parameter('auto_enabled').value:
            resp.message = 'Automatic motion is not enabled for this robot profile.'
        elif runner.active or (runner.thread is not None and runner.thread.is_alive()):
            resp.message = 'Automatic motion is already running.'
        elif runner.node.calibration_type != 'eye-in-hand':
            resp.message = 'Depth calibration sweep requires an eye-in-hand camera.'
        elif not str(runner.node.get_parameter('depth_topic').value):
            resp.message = 'No depth_topic configured for this camera.'
        elif runner.motion_block_reason():
            resp.message = runner.motion_block_reason()
        elif not runner.board_ready():
            resp.message = 'Position the camera so the whole board is detected, then start.'
        elif not runner.heartbeat_ok():
            resp.message = 'No operator heartbeat; start from the open GUI page.'
        else:
            runner.stop_event.clear()
            runner.status = {**runner.status, 'active': True, 'state': 'preparing', 'mode': 'depth_sweep',
                             'phase': 'depth', 'pose': 0, 'total': 0, 'accepted': 0, 'skipped': 0,
                             'depth_dataset': None, 'message': 'Checking robot and controller.'}
            runner.thread = threading.Thread(target=run, args=(runner,), daemon=True)
            runner.thread.start()
            resp.success, resp.message = True, 'Depth calibration sweep started.'
    return resp


def run(runner):
    from rcl_interfaces.srv import GetParameters
    from .automatic_calibration import Stopped
    try:
        if not runner.description.wait_for_service(timeout_sec=2):
            raise RuntimeError('Robot description is unavailable.')
        urdf = runner.wait(runner.description.call_async(GetParameters.Request(names=['robot_description'])), 3)
        root = ET.fromstring(urdf.values[0].string_value)
        runner.configure_robot(root)
        if not runner.action.wait_for_server(timeout_sec=2):
            raise RuntimeError('Trajectory controller is unavailable.')
        initial = runner.joint_positions()
        runner.require_joint_limits(initial)
        runner.settle(initial)
        session = DepthSweepSession(runner, root)
        captured = session.run()
        runner.update('completed', f'{len(captured)} views stored in {session.directory}. Fit them on the depth page.',
                      active=False, accepted=len(captured), depth_dataset=session.directory)
    except Stopped as exc:
        runner.update('stopped', str(exc), active=False)
    except Exception as exc:  # noqa: BLE001 - shown in the GUI
        runner.node.get_logger().error(f'Depth sweep: {exc}')
        runner.update('failed', str(exc), active=False)
    finally:
        with runner.lock:
            if runner.goal is not None:
                runner.goal.cancel_goal_async()
                runner.goal = None
