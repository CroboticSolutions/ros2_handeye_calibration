"""Store hand-eye and welding-TCP results for a FANUC CRX (crx10ia_l / crx10ia).

Same scheme as welding_gun_apply_cli (PiPER): nothing is baked into a xacro. Each
Apply writes an immutable version in the robot's calibration store
(~/.ros/calibration/<robot>/<mount>/, calibration_store.py) and repoints
current.yaml; crx_description's crx_calibration_args.xacro (real robot,
fanuc_control) and seam_ros2_pkg's crx10ia_gz.urdf.xacro (Gazebo) load that file.

Both URDFs share the `flange` link (identical FK; link_6 vs J6_link differ by
160 mm), so calibrations are always relative to flange.

camera  femto_bolt:     flange -> camera_base_link (mount femto_bolt_handeye)
        realsense_d435: flange -> camera_bottom_screw_frame (mount realsense_d435_handeye)
        origin = T_cal * inv(camera root -> optical). Real Femto: that inner chain
        from the running driver's TF (publish_tf:=true); everything else (sim, and the
        D435, whose driver runs with publish_tf:=false) from the nominal xacro chain.
        --target real | nominal | both; the other origin is carried over.
tcp     flange -> TCP link (mount welding_tcp). real: crx_description `tcp`
        (fanuc_control MoveIt tip), sim: seam_ros2_pkg `welding_tcp`. Each target
        keeps its own axis convention (`wire_axis_tcp`): the new orientation is the
        smallest rotation of the active one that puts that axis on the calibrated
        tool +Z (wire). No limit on how far it may move (first calibration of an
        unknown tool). `tcp-offset --along-wire-mm` stores a manual shift.

Both refuse a YAML whose acceptance check failed unless --force. The caller pushes
the new robot_description and restarts MoveIt (GUI Apply does both).
"""
import argparse
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from hand_eye_calibration import calibration_store
from hand_eye_calibration.welding_gun_apply_cli import (
    _delta,
    _fmt,
    _metrics,
    _origin,
    _origin_matrix,
    chain_from_urdf,
    load_calibration,
)

ROBOTS = ('crx10ia_l', 'crx10ia')
EFFECTOR = 'flange'
CAMERAS = {
    'femto_bolt': {'mount': calibration_store.CRX_FEMTO_HANDEYE, 'root': 'camera_base_link'},
    'realsense_d435': {'mount': calibration_store.CRX_D435_HANDEYE, 'root': 'camera_bottom_screw_frame'},
}
# Gazebo keeps the old FANUC sim optical frame name as an identity alias.
OPTICAL_ALIASES = {'oak_rgb_camera_optical_frame': 'camera_color_optical_frame'}
TCP_TARGETS = ('real', 'sim')


def crx_share():
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('crx_description')
    except Exception:  # noqa: BLE001 - source tree fallback (tests without an overlay)
        return '/root/arms_ws/src/robots/ros2_fanuc_interface/crx_description'


def camera_default(camera):
    return f'{crx_share()}/config/calibration/{camera}_handeye_default.yaml'


def tcp_default():
    return f'{crx_share()}/config/calibration/welding_tcp_default.yaml'


def _check_robot(robot):
    if robot not in ROBOTS:
        raise SystemExit(f'Unknown CRX robot {robot!r}; expected one of {ROBOTS}.')


def nominal_inner_chain(camera, optical_frame):
    """camera root -> optical from the crx_wrist_camera macro (nominal extrinsics)."""
    with tempfile.TemporaryDirectory() as tmp:
        cal = os.path.join(tmp, 'zero.yaml')
        with open(cal, 'w') as f:
            yaml.safe_dump({'real': {'xyz': '0 0 0', 'rpy': '0 0 0'},
                            'nominal': {'xyz': '0 0 0', 'rpy': '0 0 0'}}, f)
        wrapper = os.path.join(tmp, 'w.xacro')
        with open(wrapper, 'w') as f:
            f.write('<?xml version="1.0"?><robot xmlns:xacro="http://ros.org/wiki/xacro" name="w">'
                    '<link name="p"/><xacro:include filename="$(find crx_description)/urdf/include/'
                    'crx_wrist_camera.xacro"/>'
                    f'<xacro:crx_wrist_camera camera="{camera}" parent_link="p" '
                    f'calibration_file="{cal}" nominal="true"/></robot>')
        urdf = subprocess.run(['xacro', wrapper], check=True, capture_output=True, text=True,
                              timeout=30).stdout
    return chain_from_urdf(urdf, optical_frame, CAMERAS[camera]['root'])


def live_inner_chain(root, optical_frame, timeout_s=5.0):
    """camera root -> optical as the running system publishes it (Femto driver TF)."""
    import time

    import rclpy
    from rclpy.duration import Duration
    from tf2_ros import Buffer, TransformListener
    rclpy.init()
    node = rclpy.create_node('crx_apply_tf')
    buffer = Buffer()
    TransformListener(buffer, node)
    try:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                t = buffer.lookup_transform(root, optical_frame, rclpy.time.Time(), Duration(seconds=0.1))
            except Exception:  # noqa: BLE001 - keep waiting for TF
                continue
            q, p = t.transform.rotation, t.transform.translation
            m = np.eye(4)
            m[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            m[:3, 3] = [p.x, p.y, p.z]
            return m
        raise SystemExit(f'TF {root} -> {optical_frame} not available; is the camera driver running?')
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _camera_base(robot, camera):
    mount = CAMERAS[camera]['mount']
    calibration_store.ensure_seeded(mount, camera_default(camera), robot)
    base = calibration_store.active(mount, camera_default(camera), robot)
    if base is None:
        raise SystemExit(f'No active {camera} calibration and no default at {camera_default(camera)}.')
    return base


def apply_camera(robot, camera, calibration_yaml, target='real', inner_real=None, force=False,
                 label=None, activate=True):
    """Write (and by default activate) a camera mount version. Returns (calibration_id, record)."""
    _check_robot(robot)
    if camera not in CAMERAS:
        raise SystemExit(f'Unsupported CRX wrist camera {camera!r}; expected one of {tuple(CAMERAS)}.')
    data, T = load_calibration(calibration_yaml, force)
    if data.get('robot_effector_frame') != EFFECTOR or data.get('calibration_type') != 'eye-in-hand':
        raise SystemExit(f'Expected an eye-in-hand calibration relative to {EFFECTOR!r} (CRX mounts '
                         f'live on the flange), got {data.get("robot_effector_frame")!r}.')
    optical = OPTICAL_ALIASES.get(data['tracking_base_frame'], data['tracking_base_frame'])
    root = CAMERAS[camera]['root']
    inner = {}
    if target in ('real', 'both'):
        if inner_real is not None:
            inner['real'] = inner_real
        elif camera == 'femto_bolt':
            inner['real'] = live_inner_chain(root, optical)
        else:
            inner['real'] = nominal_inner_chain(camera, optical)
    if target in ('nominal', 'both'):
        inner['nominal'] = nominal_inner_chain(camera, optical)
    base = _camera_base(robot, camera)
    record = {
        'label': label,
        'kind': 'camera_mount',
        'camera': camera,
        'parent_link': EFFECTOR,
        'child_link': root,
        'real': dict(base['real']),
        'nominal': dict(base['nominal']),
        'updated': [],
        'delta_from_previous': {},
    }
    for name, chain in inner.items():
        record[name] = _origin(T @ np.linalg.inv(chain))
        record['updated'].append(name)
        mm, deg = _delta(base[name], record[name])
        record['delta_from_previous'][name] = {'translation_mm': round(mm, 3), 'rotation_deg': round(deg, 4)}
        record[f'inner_chain_{name}'] = {'xyz': _fmt(chain[:3, 3]),
                                         'quat_xyzw': _fmt(Rotation.from_matrix(chain[:3, :3]).as_quat())}
    record['source'] = {
        'calibration_yaml': os.path.realpath(os.path.expanduser(calibration_yaml)),
        'sha256': calibration_store.sha256(calibration_yaml),
        'timestamp': data.get('timestamp'),
        'calibration_type': data.get('calibration_type'),
        'robot_effector_frame': data.get('robot_effector_frame'),
        'tracking_base_frame': data.get('tracking_base_frame'),
        'dataset_path': data.get('dataset_path'),
        'transform': data.get('transform'),
    }
    record['metrics'] = _metrics(data)
    mount = CAMERAS[camera]['mount']
    calibration_id, _path, record = calibration_store.save_version(mount, record, robot=robot)
    if activate:
        calibration_store.activate(mount, calibration_id, robot)
    return calibration_id, record


def _tcp_base(robot):
    mount = calibration_store.CRX_WELDING_TCP
    calibration_store.ensure_seeded(mount, tcp_default(), robot)
    base = calibration_store.active(mount, tcp_default(), robot)
    if base is None:
        raise SystemExit(f'No active TCP calibration and no default at {tcp_default()}.')
    return base


def tcp_rotation(current_rpy, wire_axis_tcp, tool_z):
    """Smallest rotation of the current TCP orientation that puts its wire axis on tool_z."""
    current = Rotation.from_euler('xyz', current_rpy)
    wire = current.apply(np.asarray(wire_axis_tcp, float) / np.linalg.norm(wire_axis_tcp))
    target = np.asarray(tool_z, float) / np.linalg.norm(tool_z)
    axis = np.cross(wire, target)
    angle = np.arctan2(np.linalg.norm(axis), np.dot(wire, target))
    if np.linalg.norm(axis) < 1e-12:
        if angle < 1e-6:
            return current
        # Antiparallel: any axis perpendicular to the wire.
        axis = np.cross(wire, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(wire, [0.0, 1.0, 0.0])
    align = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle)
    return align * current


def _rpy(rotation):
    """URDF rpy; the real CRX tcp default (pitch 90 deg) is exactly at gimbal lock,
    where scipy warns but still returns an equivalent rotation."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return rotation.as_euler('xyz')


def _tcp_version(robot, base, target, origin, label, extra):
    record = {
        'label': label,
        'kind': 'tool_tcp',
        'parent_link': EFFECTOR,
        'child_link': base.get('child_link', 'tcp'),
        'real': dict(base['real']),
        'sim': dict(base['sim']),
        'updated': [target],
    }
    record[target] = {**base[target], **origin}
    a, b = _origin_matrix(base[target]), _origin_matrix(record[target])
    w = np.asarray(base[target]['wire_axis_tcp'], float)
    cos = np.clip(np.dot(a[:3, :3] @ w, b[:3, :3] @ w) / np.dot(w, w), -1.0, 1.0)
    record['delta_from_previous'] = {target: {
        'translation_mm': round(float(np.linalg.norm(b[:3, 3] - a[:3, 3]) * 1000.0), 3),
        'wire_axis_deg': round(float(np.degrees(np.arccos(cos))), 4)}}
    record.update(extra)
    mount = calibration_store.CRX_WELDING_TCP
    return calibration_store.save_version(mount, record, robot=robot)


def apply_tcp(robot, calibration_yaml, target='real', force=False, label=None, activate=True):
    """Pivot/touch-off result (flange -> tool, +Z = wire) -> new TCP store version."""
    _check_robot(robot)
    data, T = load_calibration(calibration_yaml, force)
    parent = data.get('parent_frame') or data.get('robot_flange_frame')
    if parent not in (None, EFFECTOR):
        raise SystemExit(f'Expected a TCP relative to {EFFECTOR!r}, got {parent!r}.')
    base = _tcp_base(robot)
    rotation = tcp_rotation(np.fromstring(base[target]['rpy'], sep=' '),
                            base[target]['wire_axis_tcp'], T[:3, 2])
    origin = {'xyz': _fmt(T[:3, 3]), 'rpy': _fmt(_rpy(rotation))}
    extra = {
        'source': {
            'calibration_yaml': os.path.realpath(os.path.expanduser(calibration_yaml)),
            'sha256': calibration_store.sha256(calibration_yaml),
            'timestamp': data.get('timestamp'),
            'calibration_mode': data.get('calibration_mode'),
            'parent_frame': parent,
            'transform': data.get('transform'),
            'source_note': data.get('source'),
        },
        'metrics': {
            'acceptance_passed': (data.get('acceptance') or {}).get('passed'),
            'acceptance_summary': (data.get('acceptance') or {}).get('summary'),
            'sample_count': data.get('sample_count'),
            'validation_status': data.get('validation_status'),
            'rms_m': data.get('rms_m') or (data.get('quality') or {}).get('rms_m'),
        },
    }
    calibration_id, _path, record = _tcp_version(robot, base, target, origin, label, extra)
    if activate:
        calibration_store.activate(calibration_store.CRX_WELDING_TCP, calibration_id, robot)
    return calibration_id, record


def offset_tcp(robot, along_wire_mm, target='real', label=None, activate=True):
    """Manual shift of the active TCP along its wire axis as a new version."""
    _check_robot(robot)
    base = _tcp_base(robot)
    T = _origin_matrix(base[target])
    w = np.asarray(base[target]['wire_axis_tcp'], float)
    T[:3, 3] += T[:3, :3] @ (w / np.linalg.norm(w)) * (along_wire_mm / 1000.0)
    origin = {'xyz': _fmt(T[:3, 3]), 'rpy': base[target]['rpy']}
    label = label or f'{along_wire_mm:+.2f} mm along the wire from {base.get("calibration_id")}'
    calibration_id, _path, record = _tcp_version(
        robot, base, target, origin, label,
        {'manual_offset': {'along_wire_mm': along_wire_mm, 'from_id': base.get('calibration_id')}})
    if activate:
        calibration_store.activate(calibration_store.CRX_WELDING_TCP, calibration_id, robot)
    return calibration_id, record


def mount_name(kind, camera=None):
    if kind == 'tcp':
        return calibration_store.CRX_WELDING_TCP
    if camera not in CAMERAS:
        raise SystemExit(f'--camera is required for camera mounts ({"|".join(CAMERAS)}).')
    return CAMERAS[camera]['mount']


def _print_version(robot, calibration_id, record, mount, activated):
    print(f'Calibration {calibration_id} ({", ".join(record["updated"]) or "-"}) written to '
          f'{calibration_store.mount_dir(mount, robot)}' + (' and activated' if activated else ''))
    for name, d in record.get('delta_from_previous', {}).items():
        change = ', '.join(f'{k} {v}' for k, v in d.items())
        child = record[name].get('child_link', record['child_link'])
        print(f'  {name}: {record["parent_link"]} -> {child} xyz {record[name]["xyz"]} '
              f'rpy {record[name]["rpy"]} (change {change})')


def _parse_inner(text):
    v = [float(x) for x in text.split()]
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(v[3:]).as_matrix()
    m[:3, 3] = v[:3]
    return m


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot', choices=ROBOTS, default='crx10ia_l')
    sub = parser.add_subparsers(dest='kind', required=True)
    cam = sub.add_parser('camera')
    cam.add_argument('calibration_yaml', nargs='?', default='~/.ros/hand_eye_calibration.yaml')
    cam.add_argument('--camera', choices=tuple(CAMERAS), required=True)
    cam.add_argument('--target', choices=('real', 'nominal', 'both'), default='real',
                     help='origin to recompute: real (robot) or nominal (simulation)')
    cam.add_argument('--inner-real', help='camera root->optical as "x y z qx qy qz qw" (default: '
                                          'live driver TF for the Femto, nominal for the D435)')
    tcp = sub.add_parser('tcp')
    tcp.add_argument('calibration_yaml', nargs='?', default='~/.ros/tool_tcp_calibration.yaml')
    tcp.add_argument('--target', choices=TCP_TARGETS, default='real')
    off = sub.add_parser('tcp-offset', help='shift the active TCP along the wire (new version)')
    off.add_argument('--along-wire-mm', type=float, required=True)
    off.add_argument('--target', choices=TCP_TARGETS, default='real')
    for p in (cam, tcp, off):
        p.add_argument('--label', help='free-text label stored with the version')
        p.add_argument('--no-activate', action='store_true', help='only write the version')
    for p in (cam, tcp):
        p.add_argument('--force', action='store_true')
    lst = sub.add_parser('list', help='list calibration versions')
    act = sub.add_parser('activate', help='point current.yaml at an existing version (rollback)')
    act.add_argument('calibration_id')
    for p in (lst, act):
        p.add_argument('--mount', choices=('camera', 'tcp'), default='camera')
        p.add_argument('--camera', choices=tuple(CAMERAS))
    args = parser.parse_args(argv)
    robot = args.robot

    if args.kind == 'camera':
        inner = _parse_inner(args.inner_real) if args.inner_real else None
        calibration_id, record = apply_camera(robot, args.camera, args.calibration_yaml, args.target,
                                              inner, args.force, args.label, not args.no_activate)
        _print_version(robot, calibration_id, record, CAMERAS[args.camera]['mount'], not args.no_activate)
        return 0
    if args.kind == 'tcp':
        calibration_id, record = apply_tcp(robot, args.calibration_yaml, args.target, args.force,
                                           args.label, not args.no_activate)
        _print_version(robot, calibration_id, record, calibration_store.CRX_WELDING_TCP, not args.no_activate)
        return 0
    if args.kind == 'tcp-offset':
        calibration_id, record = offset_tcp(robot, args.along_wire_mm, args.target, args.label,
                                            not args.no_activate)
        _print_version(robot, calibration_id, record, calibration_store.CRX_WELDING_TCP, not args.no_activate)
        return 0
    mount = mount_name(args.mount, args.camera)
    if args.kind == 'list':
        for v in calibration_store.versions(mount, robot):
            print(f'{"*" if v["active"] else " "} {v["calibrationId"]}  {v["createdUtc"]}  '
                  f'{",".join(v["updated"] or [])}  {v["label"] or ""}')
        return 0
    path = calibration_store.activate(mount, args.calibration_id, robot)
    print(f'Activated {args.calibration_id} ({path})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
