"""Bake hand-eye and tool-TCP results into the Piper welding-gun xacro files.

The production stack (piper_orbbec_femto_bolt_welding_gun) does not use the
generic mount joints the other Apply paths rewrite:

camera  piper_femto_bolt_handeye_macros.xacro passes one <origin> to
        orbbec_description's femto_bolt macro, with a nominal (simulation)
        and a real value selected by use_nominal_extrinsics. Calibration gives
        parent -> camera_color_optical_frame, parent = femto_parent_link in the
        welding-gun xacro (link5 for the real mount since 2026-10-05); the origin
        is parent -> camera_base_link = T_cal * inv(camera_base_link -> color_optical).
        Real: that inner chain comes from the running driver's TF (its own
        depth->colour extrinsics). Nominal: from the femto_bolt xacro itself.
        By default only the real value is changed; the simulation keeps its
        pose unless --nominal is given.

tcp     arm_tcp_joint in piper_welding_gun.urdf.xacro (parent
        piper_welding_gun, identity mount on link6). Only the real-robot
        branch (xacro:unless simulation) is rewritten. The calibrated tool +Z
        (wire direction) becomes arm_tcp +X, the axis the welding planner
        uses; of the two equivalent rolls the one closest to the current
        origin is kept so downstream conventions do not flip.

Both refuse a YAML whose acceptance check failed. MoveIt keeps its model until
the hardware stack is restarted.
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

PIPER_URDF = '/root/arms_ws/src/robots/piper_ros/src/robot_description/piper_description/urdf'
FEMTO_XACRO = f'{PIPER_URDF}/include/piper_femto_bolt_handeye_macros.xacro'
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


_FEMTO_ORIGIN = re.compile(
    r"""xyz="\$\{'(?P<xn>[^']+)' if use_nominal_extrinsics else '(?P<xr>[^']+)'\}"\s*"""
    r"""rpy="\$\{'(?P<rn>[^']+)' if use_nominal_extrinsics else '(?P<rr>[^']+)'\}\"""")


def bake_camera(xacro_path, T_cal, inner_real, inner_nominal=None):
    text = open(xacro_path).read()
    m = _FEMTO_ORIGIN.search(text)
    if not m:
        raise SystemExit(f'{xacro_path}: nominal/real femto origin expression not found.')
    values = {k: m.group(k) for k in ('xn', 'xr', 'rn', 'rr')}
    mount = T_cal @ np.linalg.inv(inner_real)
    values['xr'] = _fmt(mount[:3, 3])
    values['rr'] = _fmt(Rotation.from_matrix(mount[:3, :3]).as_euler('xyz'))
    if inner_nominal is not None:
        nominal = T_cal @ np.linalg.inv(inner_nominal)
        values['xn'] = _fmt(nominal[:3, 3])
        values['rn'] = _fmt(Rotation.from_matrix(nominal[:3, :3]).as_euler('xyz'))
    new = (f"""xyz="${{'{values['xn']}' if use_nominal_extrinsics else '{values['xr']}'}}"\n"""
           f"""              rpy="${{'{values['rn']}' if use_nominal_extrinsics else '{values['rr']}'}}\"""")
    text = text[:m.start()] + new + text[m.end():]
    _write(xacro_path, text)
    return mount


_TCP_REAL = re.compile(
    r'(<joint\s+name="arm_tcp_joint".*?<xacro:unless\s+value="\$\(arg simulation\)">\s*<origin\s+)'
    r'xyz="(?P<xyz>[^"]+)"\s+rpy="(?P<rpy>[^"]+)"', re.DOTALL)


def tcp_origin(T_flange_tcp, current_rpy):
    """Tool +Z -> arm_tcp +X, keeping the roll closest to the current origin."""
    base = Rotation.from_matrix(T_flange_tcp[:3, :3]) * Rotation.from_euler('y', -np.pi / 2)
    candidates = [base, base * Rotation.from_euler('x', np.pi)]
    current = Rotation.from_euler('xyz', current_rpy)
    best = min(candidates, key=lambda r: (current.inv() * r).magnitude())
    return T_flange_tcp[:3, 3], best.as_euler('xyz')


def bake_tcp(xacro_path, T_flange_tcp):
    text = open(xacro_path).read()
    m = _TCP_REAL.search(text)
    if not m:
        raise SystemExit(f'{xacro_path}: real-robot origin of arm_tcp_joint not found.')
    xyz, rpy = tcp_origin(T_flange_tcp, np.fromstring(m.group('rpy'), sep=' '))
    text = text[:m.start('xyz')] + _fmt(xyz) + text[m.end('xyz'):m.start('rpy')] + _fmt(rpy) + text[m.end('rpy'):]
    _write(xacro_path, text)
    return xyz, rpy


def _write(path, text):
    if os.path.exists(path):
        with open(path) as f, open(path + '.before_apply', 'w') as backup:
            backup.write(f.read())
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='kind', required=True)
    cam = sub.add_parser('camera')
    cam.add_argument('calibration_yaml', nargs='?', default='~/.ros/hand_eye_calibration.yaml')
    cam.add_argument('--xacro', default=FEMTO_XACRO)
    cam.add_argument('--inner-real', help='camera_base_link->optical as "x y z qx qy qz qw" (default: live TF)')
    cam.add_argument('--nominal', action='store_true', help='also update the simulation (nominal) origin')
    cam.add_argument('--force', action='store_true')
    tcp = sub.add_parser('tcp')
    tcp.add_argument('calibration_yaml', nargs='?', default='~/.ros/tool_tcp_calibration.yaml')
    tcp.add_argument('--xacro', default=GUN_XACRO)
    tcp.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    if args.kind == 'camera':
        data, T = load_calibration(args.calibration_yaml, args.force)
        parent = femto_parent_link()
        if data.get('robot_effector_frame') != parent or data.get('calibration_type') != 'eye-in-hand':
            raise SystemExit(f'Expected an eye-in-hand calibration relative to {parent} (the Femto '
                             f'parent in {GUN_XACRO}), got {data.get("robot_effector_frame")!r}.')
        optical = data['tracking_base_frame']
        if args.inner_real:
            v = [float(x) for x in args.inner_real.split()]
            inner = np.eye(4)
            inner[:3, :3] = Rotation.from_quat(v[3:]).as_matrix()
            inner[:3, 3] = v[:3]
        else:
            inner = live_inner_chain(optical)
        nominal = nominal_inner_chain(optical) if args.nominal else None
        mount = bake_camera(os.path.expanduser(args.xacro), T, inner, nominal)
        print(f'Updated {args.xacro}: {parent} -> {CAMERA_BASE} xyz {_fmt(mount[:3, 3])}'
              + (' (and nominal)' if nominal is not None else ''))
    else:
        data, T = load_calibration(args.calibration_yaml, args.force)
        parent = data.get('parent_frame') or data.get('robot_flange_frame')
        if parent not in (None, 'link6'):
            raise SystemExit(f'Expected a TCP relative to link6, got {parent!r}.')
        xyz, rpy = bake_tcp(os.path.expanduser(args.xacro), T)
        print(f'Updated {args.xacro}: arm_tcp_joint (real) xyz {_fmt(xyz)} rpy {_fmt(rpy)}')
    print('MoveIt keeps the previous model until the hardware stack is restarted.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
