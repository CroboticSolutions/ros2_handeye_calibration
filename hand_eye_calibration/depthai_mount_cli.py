#!/usr/bin/env python3
"""
Convert a saved hand-eye YAML (eye-in-hand, optical frame as tracking_base_frame)
into depthai URDF mount arguments — and optionally write them into a xacro.

During calibration, OpenCV returns T_effector_optical (T_cal). The depthai
description publishes T_parent_optical_nominal (T_nom) when the center joint uses
identity cam_pos (parent → oak-d_frame → <socket>_camera_frame → optical). You want

    T_parent_optical = T_mount @ T_nom = T_cal   =>   T_mount = T_cal @ inv(T_nom)

where T_mount is the fixed joint robot_effector_frame → oak-d-base-frame.

Supported camera models (nominal chains match depthai_descriptions_v3
depthai_macro.urdf.xacro):
  OAK-D-PRO-W / OAK-D-PRO / OAK-D : baseline 0.075, rgb centered
  OAK-D-SR                        : baseline 0.02, no rgb socket

The optical socket (rgb / left / right) is inferred from tracking_base_frame in the
YAML ("oak_rgb_camera_optical_frame" → rgb), or forced with --socket.

Typical use after saving a new calibration from the GUI:

    python3 -m hand_eye_calibration.depthai_mount_cli \
        --update-xacro ~/arms_ws/src/piper_ros/src/robot_description/piper_description/urdf/include/piper_oak_d_pro_w_handeye_macros.xacro

then rebuild piper_description and restart the robot stack.

Assumes calibration was captured while the depthai center joint used identity
cam_pos (true when the camera TF chain comes from the Piper URDF mount macro).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

# socket y-offsets from oak-d_frame, per model (depthai_macro.urdf.xacro).
CAMERA_MODELS: dict[str, dict[str, float]] = {
    "OAK-D-PRO-W": {"rgb": 0.0, "left": 0.075 / 2, "right": -0.075 / 2},
    "OAK-D-PRO": {"rgb": 0.0, "left": 0.075 / 2, "right": -0.075 / 2},
    "OAK-D": {"rgb": 0.0, "left": 0.075 / 2, "right": -0.075 / 2},
    "OAK-D-SR": {"left": 0.02 / 2, "right": -0.02 / 2},  # no rgb socket
}

DEFAULT_XACRO_JOINT = "piper_hand_eye_link6_mount"


def _T_mat(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = t
    return T


def _T_from_urdf_rpy_xyz(xyz: list[float], rpy: list[float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw about X, Y, Z (same as scipy euler 'xyz')."""
    return _T_mat(Rot.from_euler("xyz", rpy).as_matrix(), np.array(xyz, dtype=float))


def nominal_parent_to_optical(camera_model: str, socket: str) -> np.ndarray:
    """T oak-d-base-frame → <socket> optical frame with identity cam_pos.

    Chain per depthai_macro.urdf.xacro: parent → oak-d_frame (identity) →
    camera_frame (y = socket offset) → optical (rpy -pi/2 0 -pi/2).
    """
    offsets = CAMERA_MODELS[camera_model]
    if socket not in offsets:
        raise ValueError(
            f"{camera_model} has no '{socket}' socket; available: {sorted(offsets)}"
        )
    T_base_cam = _T_from_urdf_rpy_xyz([0.0, offsets[socket], 0.0], [0.0, 0.0, 0.0])
    T_cam_opt = _T_from_urdf_rpy_xyz([0.0, 0.0, 0.0], [-math.pi / 2, 0.0, -math.pi / 2])
    return T_base_cam @ T_cam_opt


def infer_socket(tracking_base_frame: str) -> str | None:
    frame = tracking_base_frame.lower()
    for socket in ("rgb", "right", "left"):
        if socket in frame:
            return socket
    return None


def _rotation_to_urdf_rpy(Rm: np.ndarray) -> tuple[float, float, float]:
    """Match hand_eye_calibration.node.tf_to_urdf_tf convention."""
    e = list(Rot.from_matrix(Rm).as_euler(seq="ZYX"))
    return float(e[2]), float(e[1]), float(e[0])


def compute_mount(T_cal: np.ndarray, camera_model: str, socket: str) -> np.ndarray:
    return T_cal @ np.linalg.inv(nominal_parent_to_optical(camera_model, socket))


def _load_yaml(path: str) -> dict:
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def update_xacro_origin(
    xacro_path: str,
    xyz: np.ndarray,
    rpy: tuple[float, float, float],
    joint_name: str = DEFAULT_XACRO_JOINT,
) -> None:
    """Rewrite the <origin> of the named fixed joint in place."""
    path = os.path.expanduser(xacro_path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Non-greedy .*? alone already stops at the nearest "/>" — excluding "/"
    # from the group (as an earlier version did) breaks on any xacro
    # expression containing "/" in the existing origin (e.g. rpy="0 ${pi/2}
    # 0"): the "/" can't be consumed, so the match silently fails and re
    # falls through to the *next* <origin> tag in the file instead, corrupting
    # a different joint. Reproduced live 2026-08-28 against a real xacro file.
    joint_pattern = re.compile(
        r'(<joint\s+name="%s"[^>]*>.*?<origin\b)(.*?)(/>)' % re.escape(joint_name),
        re.DOTALL,
    )
    m = joint_pattern.search(content)
    if not m:
        raise SystemExit(f"Joint '{joint_name}' with an <origin .../> not found in {path}")

    new_attrs = (
        f'\n        xyz="{xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f}"'
        f'\n        rpy="{rpy[0]:.12f} {rpy[1]:.12f} {rpy[2]:.12f}"'
    )
    content = content[: m.start(2)] + new_attrs + content[m.end(2):]
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "calibration_yaml",
        nargs="?",
        default=os.path.expanduser("~/.ros/hand_eye_calibration.yaml"),
        help="Path to hand_eye_calibration YAML (default: ~/.ros/hand_eye_calibration.yaml)",
    )
    p.add_argument(
        "--camera-model",
        default="OAK-D-PRO-W",
        choices=sorted(CAMERA_MODELS),
        help="depthai_descriptions model used for the nominal chain (default: OAK-D-PRO-W)",
    )
    p.add_argument(
        "--socket",
        choices=("rgb", "left", "right"),
        help="Optical socket the calibration targets; default: inferred from tracking_base_frame",
    )
    p.add_argument(
        "--update-xacro",
        metavar="XACRO_PATH",
        help="Rewrite the mount joint <origin> in this xacro file in place",
    )
    p.add_argument(
        "--joint-name",
        default=DEFAULT_XACRO_JOINT,
        help=f"Mount joint name inside the xacro (default: {DEFAULT_XACRO_JOINT})",
    )
    args = p.parse_args(argv)

    data = _load_yaml(args.calibration_yaml)
    ctype = data.get("calibration_type", "")
    if ctype != "eye-in-hand":
        print(
            "Warning: calibration_type is not eye-in-hand; mount composition still uses "
            "effector → optical from YAML.",
            file=sys.stderr,
        )

    eff = data.get("robot_effector_frame", "")
    optical = data.get("tracking_base_frame", "")
    socket = args.socket or infer_socket(optical)
    if socket is None:
        raise SystemExit(
            f"Cannot infer optical socket from tracking_base_frame={optical!r}; pass --socket"
        )

    t = data["transform"]
    T_cal = _T_mat(
        Rot.from_quat([t["qx"], t["qy"], t["qz"], t["qw"]]).as_matrix(),
        np.array([t["tx"], t["ty"], t["tz"]], dtype=float),
    )

    T_mount = compute_mount(T_cal, args.camera_model, socket)
    xyz = T_mount[:3, 3]
    roll, pitch, yaw = _rotation_to_urdf_rpy(T_mount[:3, :3])

    print(f"# {args.camera_model} ({socket} socket), calibration: {eff} -> {optical}")
    print(f"# Mount joint ({eff} -> oak-d-base-frame):")
    print(f'#   xyz="{xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f}"')
    print(f'#   rpy="{roll:.12f} {pitch:.12f} {yaw:.12f}"')
    print("# As depthai driver.launch.py args (parent_frame:=%s):" % eff)
    print(f"    cam_pos_x:={xyz[0]:.12f} \\")
    print(f"    cam_pos_y:={xyz[1]:.12f} \\")
    print(f"    cam_pos_z:={xyz[2]:.12f} \\")
    print(f"    cam_roll:={roll:.12f} \\")
    print(f"    cam_pitch:={pitch:.12f} \\")
    print(f"    cam_yaw:={yaw:.12f}")
    print("# Expected TF tree: %s -> oak-d-base-frame -> … -> %s" % (eff, optical))

    if args.update_xacro:
        update_xacro_origin(args.update_xacro, xyz, (roll, pitch, yaw), args.joint_name)
        print(f"# Updated {args.update_xacro} (joint '{args.joint_name}').")
        print("# Rebuild piper_description and restart the robot stack to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
