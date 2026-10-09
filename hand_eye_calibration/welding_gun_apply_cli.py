"""Bake hand-eye and tool-TCP results into the Piper welding-gun xacro files.

The production stack (piper_orbbec_femto_bolt_welding_gun) does not use the
generic mount joints the other Apply paths rewrite:

camera  The Femto origin (femto_parent_link -> camera_base_link, link5 for the
        real mount since 2026-10-05) is not in any xacro: it is a new immutable
        version in the robot's calibration store
        (~/.ros/calibration/<robot_id>/femto_bolt_handeye/, calibration_store.py)
        and current.yaml is repointed at it; the xacro loads that file. Each
        version holds a real and a nominal (simulation) origin:
        origin = T_cal * inv(camera_base_link -> color_optical).
        Real: that inner chain comes from the running driver's TF (its own
        depth->colour extrinsics). Nominal: from the femto_bolt xacro itself.
        --target picks which origin is recomputed (real by default; a sim stack
        uses nominal); the other one is carried over from the active version.
        `list` and `activate <id>` show and roll back versions.

tcp     arm_tcp_joint (piper_welding_gun -> arm_tcp, identity mount on link6)
        is not in the xacro either: a version in the store's welding_gun_tcp
        mount, with a real and a sim origin (sim = gun model tip). --target
        picks which one (real by default). The calibrated tool +Z (wire
        direction) becomes arm_tcp +X, the axis the welding planner uses; of
        the two equivalent rolls the one closest to the active origin is kept
        so downstream conventions do not flip. There is deliberately no limit
        on how far it may move: a first calibration on an unknown tool can
        change anything. `tcp-offset --along-wire-mm` stores a manual shift
        along the wire as its own version.

Both refuse a YAML whose acceptance check failed. The caller pushes the new
robot_description and restarts MoveIt (GUI Apply does both).
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from hand_eye_calibration import calibration_store

PIPER_URDF = '/root/arms_ws/src/robots/piper_ros/src/robot_description/piper_description/urdf'
FEMTO_DEFAULT_YAML = f'{os.path.dirname(PIPER_URDF)}/config/calibration/femto_bolt_handeye_default.yaml'
TCP_DEFAULT_YAML = f'{os.path.dirname(PIPER_URDF)}/config/calibration/welding_gun_tcp_default.yaml'
GUN_XACRO = f'{PIPER_URDF}/piper_welding_gun.urdf.xacro'
CAMERA_BASE = 'camera_base_link'


def femto_parent_link(gun_xacro=GUN_XACRO):
    """Real-robot Femto parent link declared in the welding-gun xacro (link6 before 2026-10-05)."""
    with open(gun_xacro) as f:
        match = re.search(r'<xacro:arg name="femto_parent_link" default="([^"]+)"', f.read())
    return match.group(1) if match else 'link6'


def _matrix(xyz, rpy):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    m[:3, 3] = xyz
    return m


def _fmt(values):
    return ' '.join(f'{v:.12f}' for v in values)


def load_calibration(path, force=False):
    with open(os.path.expanduser(path)) as f:
        data = yaml.safe_load(f)
    acceptance = data.get('acceptance')
    if acceptance is not None and not acceptance.get('passed') and not force:
        raise SystemExit(f'{path}: acceptance failed ({acceptance.get("summary")}); refusing to apply. '
                         'Use --force only if you know why.')
    t = data['transform']
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([t['qx'], t['qy'], t['qz'], t['qw']]).as_matrix()
    m[:3, 3] = [t['tx'], t['ty'], t['tz']]
    return data, m


def chain_from_urdf(urdf_text, child, parent):
    joints = {j.find('child').attrib['link']: j for j in ET.fromstring(urdf_text).findall('joint')}
    out = np.eye(4)
    link = child
    while link != parent:
        if link not in joints:
            raise ValueError(f'No fixed chain {parent} -> {child} in URDF.')
        j = joints[link]
        if j.attrib['type'] != 'fixed':
            raise ValueError(f'Joint {j.attrib["name"]} is not fixed.')
        o = j.find('origin')
        o = {} if o is None else o.attrib
        out = _matrix(np.fromstring(o.get('xyz', '0 0 0'), sep=' '),
                      np.fromstring(o.get('rpy', '0 0 0'), sep=' ')) @ out
        link = j.find('parent').attrib['link']
    return out


def nominal_inner_chain(optical_frame):
    """camera_base_link -> optical from orbbec_description's femto_bolt macro."""
    wrapper = ('<?xml version="1.0"?><robot xmlns:xacro="http://ros.org/wiki/xacro" name="w">'
               '<link name="p"/><xacro:include filename="$(find orbbec_description)/urdf/femto_bolt.urdf.xacro"/>'
               '<xacro:femto_bolt prefix="camera" parent="p" use_nominal_extrinsics="true">'
               '<origin xyz="0 0 0" rpy="0 0 0"/></xacro:femto_bolt></robot>')
    with tempfile.NamedTemporaryFile('w', suffix='.xacro', delete=False) as f:
        f.write(wrapper)
        path = f.name
    try:
        urdf = subprocess.run(['xacro', path], check=True, capture_output=True, text=True, timeout=30).stdout
    finally:
        os.unlink(path)
    return chain_from_urdf(urdf, optical_frame, CAMERA_BASE)


def live_inner_chain(optical_frame, timeout_s=5.0):
    """camera_base_link -> optical as the running system publishes it."""
    import rclpy
    from rclpy.duration import Duration
    from tf2_ros import Buffer, TransformListener
    rclpy.init()
    node = rclpy.create_node('welding_gun_apply_tf')
    buffer = Buffer()
    TransformListener(buffer, node)
    try:
        import time
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                t = buffer.lookup_transform(CAMERA_BASE, optical_frame, rclpy.time.Time(), Duration(seconds=0.1))
            except Exception:  # noqa: BLE001 - keep waiting for TF
                continue
            q, p = t.transform.rotation, t.transform.translation
            m = np.eye(4)
            m[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            m[:3, 3] = [p.x, p.y, p.z]
            return m
        raise SystemExit(f'TF {CAMERA_BASE} -> {optical_frame} not available; is the Femto driver running?')
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _origin(T):
    return {'xyz': _fmt(T[:3, 3]), 'rpy': _fmt(Rotation.from_matrix(T[:3, :3]).as_euler('xyz'))}


def _origin_matrix(origin):
    return _matrix(np.fromstring(origin['xyz'], sep=' '), np.fromstring(origin['rpy'], sep=' '))


def _delta(old, new):
    """(translation mm, rotation deg) between two origins."""
    a, b = _origin_matrix(old), _origin_matrix(new)
    rot = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()
    return float(np.linalg.norm(b[:3, 3] - a[:3, 3]) * 1000.0), float(np.degrees(rot))


def _metrics(data):
    pose = (data.get('pose_calibration') or {})
    acceptance = data.get('acceptance') or {}
    return {
        'acceptance_passed': acceptance.get('passed'),
        'acceptance_summary': acceptance.get('summary'),
        'checks': {c['name']: c.get('value') for c in acceptance.get('checks', []) if 'name' in c},
        'sample_count': data.get('sample_count'),
        'position_rms_m': (pose.get('pose_metrics') or {}).get('position_rms_m'),
        'validation_position_max_m': (pose.get('validation_views') or {}).get('position_max_m'),
        'algorithm': data.get('algorithm_used'),
    }


def camera_record(data, T_cal, calibration_yaml, base, inner_real=None, inner_nominal=None, label=None):
    """New store version: recompute the requested origins, carry the rest over from base."""
    parent = data.get('robot_effector_frame')
    record = {
        'label': label,
        'kind': 'camera_mount',
        'parent_link': parent,
        'child_link': CAMERA_BASE,
        'real': dict(base['real']),
        'nominal': dict(base['nominal']),
        'updated': [],
        'delta_from_previous': {},
    }
    for name, inner in (('real', inner_real), ('nominal', inner_nominal)):
        if inner is None:
            continue
        record[name] = _origin(T_cal @ np.linalg.inv(inner))
        record['updated'].append(name)
        mm, deg = _delta(base[name], record[name])
        record['delta_from_previous'][name] = {'translation_mm': round(mm, 3), 'rotation_deg': round(deg, 4)}
        record[f'inner_chain_{name}'] = {'xyz': _fmt(inner[:3, 3]), 'quat_xyzw': _fmt(Rotation.from_matrix(inner[:3, :3]).as_quat())}
    if base.get('parent_link') not in (None, parent):
        raise SystemExit(f'Calibration parent {parent!r} differs from the active mount parent '
                         f'{base.get("parent_link")!r}; recalibrate both origins or change the mount.')
    record['source'] = {
        'calibration_yaml': os.path.realpath(os.path.expanduser(calibration_yaml)),
        'sha256': calibration_store.sha256(calibration_yaml),
        'timestamp': data.get('timestamp'),
        'calibration_type': data.get('calibration_type'),
        'robot_effector_frame': parent,
        'tracking_base_frame': data.get('tracking_base_frame'),
        'dataset_path': data.get('dataset_path'),
        'transform': data.get('transform'),
    }
    record['metrics'] = _metrics(data)
    return record


def apply_camera(calibration_yaml, target='real', inner_real=None, force=False, label=None, activate=True):
    """Write (and by default activate) a store version. Returns (calibration_id, record)."""
    data, T = load_calibration(calibration_yaml, force)
    parent = femto_parent_link()
    if data.get('robot_effector_frame') != parent or data.get('calibration_type') != 'eye-in-hand':
        raise SystemExit(f'Expected an eye-in-hand calibration relative to {parent} (the Femto '
                         f'parent in {GUN_XACRO}), got {data.get("robot_effector_frame")!r}.')
    optical = data['tracking_base_frame']
    real = nominal = None
    if target in ('real', 'both'):
        real = inner_real if inner_real is not None else live_inner_chain(optical)
    if target in ('nominal', 'both'):
        nominal = nominal_inner_chain(optical)
    calibration_store.ensure_seeded(calibration_store.FEMTO_HANDEYE, FEMTO_DEFAULT_YAML)
    base = calibration_store.active(calibration_store.FEMTO_HANDEYE, FEMTO_DEFAULT_YAML)
    if base is None:
        raise SystemExit(f'No active Femto calibration and no default at {FEMTO_DEFAULT_YAML}.')
    record = camera_record(data, T, calibration_yaml, base, real, nominal, label)
    calibration_id, _path, record = calibration_store.save_version(calibration_store.FEMTO_HANDEYE, record)
    if activate:
        calibration_store.activate(calibration_store.FEMTO_HANDEYE, calibration_id)
    return calibration_id, record


def tcp_origin(T_flange_tcp, current_rpy):
    """Tool +Z -> arm_tcp +X, keeping the roll closest to the current origin."""
    base = Rotation.from_matrix(T_flange_tcp[:3, :3]) * Rotation.from_euler('y', -np.pi / 2)
    candidates = [base, base * Rotation.from_euler('x', np.pi)]
    current = Rotation.from_euler('xyz', current_rpy)
    best = min(candidates, key=lambda r: (current.inv() * r).magnitude())
    return T_flange_tcp[:3, 3], best.as_euler('xyz')


def _tcp_delta(old, new):
    """(translation mm, wire-axis (arm_tcp +X) change deg); information only."""
    a, b = _origin_matrix(old), _origin_matrix(new)
    cos = np.clip(np.dot(a[:3, 0], b[:3, 0]), -1.0, 1.0)
    return float(np.linalg.norm(b[:3, 3] - a[:3, 3]) * 1000.0), float(np.degrees(np.arccos(cos)))


def _tcp_version(base, target, origin, label, extra):
    record = {
        'label': label,
        'kind': 'tool_tcp',
        'parent_link': base.get('parent_link', 'piper_welding_gun'),
        'child_link': base.get('child_link', 'arm_tcp'),
        'real': dict(base['real']),
        'sim': dict(base['sim']),
        'updated': [target],
    }
    record[target] = origin
    mm, deg = _tcp_delta(base[target], origin)
    record['delta_from_previous'] = {target: {'translation_mm': round(mm, 3), 'wire_axis_deg': round(deg, 4)}}
    record.update(extra)
    mount = calibration_store.WELDING_GUN_TCP
    calibration_id, _path, record = calibration_store.save_version(mount, record)
    return calibration_id, record


def _tcp_base():
    mount = calibration_store.WELDING_GUN_TCP
    calibration_store.ensure_seeded(mount, TCP_DEFAULT_YAML)
    base = calibration_store.active(mount, TCP_DEFAULT_YAML)
    if base is None:
        raise SystemExit(f'No active TCP calibration and no default at {TCP_DEFAULT_YAML}.')
    return base


def apply_tcp(calibration_yaml, target='real', force=False, label=None, activate=True):
    """Pivot/touch-off result -> new TCP store version (activated by default)."""
    data, T = load_calibration(calibration_yaml, force)
    parent = data.get('parent_frame') or data.get('robot_flange_frame')
    if parent not in (None, 'link6'):
        raise SystemExit(f'Expected a TCP relative to link6, got {parent!r}.')
    base = _tcp_base()
    xyz, rpy = tcp_origin(T, np.fromstring(base[target]['rpy'], sep=' '))
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
    calibration_id, record = _tcp_version(base, target, {'xyz': _fmt(xyz), 'rpy': _fmt(rpy)}, label, extra)
    if activate:
        calibration_store.activate(calibration_store.WELDING_GUN_TCP, calibration_id)
    return calibration_id, record


def offset_tcp(along_wire_mm, target='real', label=None, activate=True):
    """Manual shift of the active TCP along the wire (arm_tcp +X) as a new version."""
    base = _tcp_base()
    T = _origin_matrix(base[target])
    T[:3, 3] += T[:3, 0] * (along_wire_mm / 1000.0)
    origin = {'xyz': _fmt(T[:3, 3]), 'rpy': base[target]['rpy']}
    label = label or f'{along_wire_mm:+.2f} mm along the wire from {base.get("calibration_id")}'
    calibration_id, record = _tcp_version(base, target, origin, label,
                                          {'manual_offset': {'along_wire_mm': along_wire_mm,
                                                             'from_id': base.get('calibration_id')}})
    if activate:
        calibration_store.activate(calibration_store.WELDING_GUN_TCP, calibration_id)
    return calibration_id, record


_MOUNTS = {'camera': calibration_store.FEMTO_HANDEYE, 'tcp': calibration_store.WELDING_GUN_TCP}


def _print_version(calibration_id, record, mount, activated):
    print(f'Calibration {calibration_id} ({", ".join(record["updated"]) or "-"}) written to '
          f'{calibration_store.mount_dir(mount)}' + (' and activated' if activated else ''))
    for name, d in record.get('delta_from_previous', {}).items():
        change = ', '.join(f'{k} {v}' for k, v in d.items())
        print(f'  {name}: {record["parent_link"]} -> {record["child_link"]} xyz {record[name]["xyz"]} '
              f'rpy {record[name]["rpy"]} (change {change})')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='kind', required=True)
    cam = sub.add_parser('camera')
    cam.add_argument('calibration_yaml', nargs='?', default='~/.ros/hand_eye_calibration.yaml')
    cam.add_argument('--target', choices=('real', 'nominal', 'both'), default='real',
                     help='origin to recompute: real (driver TF), nominal (simulation) or both')
    cam.add_argument('--inner-real', help='camera_base_link->optical as "x y z qx qy qz qw" (default: live TF)')
    tcp = sub.add_parser('tcp')
    tcp.add_argument('calibration_yaml', nargs='?', default='~/.ros/tool_tcp_calibration.yaml')
    tcp.add_argument('--target', choices=('real', 'sim'), default='real')
    off = sub.add_parser('tcp-offset', help='shift the active TCP along the wire (new version)')
    off.add_argument('--along-wire-mm', type=float, required=True)
    off.add_argument('--target', choices=('real', 'sim'), default='real')
    for p in (cam, tcp, off):
        p.add_argument('--label', help='free-text label stored with the version')
        p.add_argument('--no-activate', action='store_true', help='only write the version')
    for p in (cam, tcp):
        p.add_argument('--force', action='store_true')
    lst = sub.add_parser('list', help='list calibration versions')
    act = sub.add_parser('activate', help='point current.yaml at an existing version (rollback)')
    act.add_argument('calibration_id')
    for p in (lst, act):
        p.add_argument('--mount', choices=tuple(_MOUNTS), default='camera')
    args = parser.parse_args(argv)

    if args.kind == 'camera':
        inner = None
        if args.inner_real:
            v = [float(x) for x in args.inner_real.split()]
            inner = np.eye(4)
            inner[:3, :3] = Rotation.from_quat(v[3:]).as_matrix()
            inner[:3, 3] = v[:3]
        calibration_id, record = apply_camera(args.calibration_yaml, args.target, inner, args.force,
                                              args.label, not args.no_activate)
        _print_version(calibration_id, record, calibration_store.FEMTO_HANDEYE, not args.no_activate)
        return 0
    if args.kind in ('tcp', 'tcp-offset'):
        if args.kind == 'tcp':
            calibration_id, record = apply_tcp(args.calibration_yaml, args.target, args.force,
                                               args.label, not args.no_activate)
        else:
            calibration_id, record = offset_tcp(args.along_wire_mm, args.target, args.label,
                                                not args.no_activate)
        _print_version(calibration_id, record, calibration_store.WELDING_GUN_TCP, not args.no_activate)
        return 0
    mount = _MOUNTS[args.mount]
    if args.kind == 'list':
        for v in calibration_store.versions(mount):
            print(f'{"*" if v["active"] else " "} {v["calibrationId"]}  {v["createdUtc"]}  '
                  f'{",".join(v["updated"] or [])}  {v["label"] or ""}')
        return 0
    path = calibration_store.activate(mount, args.calibration_id)
    print(f'Activated {args.calibration_id} ({path})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
