#!/usr/bin/env python3
"""
Convert a saved eye-in-hand hand-eye YAML (effector → camera optical) into the
D435 mount joint origin (parent → camera_bottom_screw_frame) and optionally
write it into a xacro.

Nominal chain matches realsense2_description ``sensor_d435`` with
``use_nominal_extrinsics:=true``:

  bottom_screw → camera_link → color_frame → color_optical_frame

Typical FANUC site flow after GUI save:

    python3 -m hand_eye_calibration.realsense_mount_cli \\
      --update-xacro \\
      ~/arms_ws/src/seam_ros2_pkg/urdf/include/fanuc_realsense_d435_handeye_macros.xacro \\
      --joint-name fanuc_j6_realsense_d435_mount

then rebuild ``seam_ros2_pkg`` and restart the robot stack.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .depthai_mount_cli import (
    _T_from_urdf_rpy_xyz,
    _T_mat,
    _load_yaml,
    _rotation_to_urdf_rpy,
    update_xacro_origin,
)

# From realsense2_description urdf/_d435.urdf.xacro (sensor_d435).
_D435_MESH_X = 0.0149 - 0.1e-3 - 4.2e-3  # mount_from_center - glass - zero_depth
_D435_DEPTH_PY = 0.0175
_D435_DEPTH_PZ = 0.025 / 2
_D435_COLOR_Y = 0.015

DEFAULT_JOINT_NAME = "fanuc_j6_realsense_d435_mount"


def nominal_bottom_screw_to_color_optical() -> np.ndarray:
    """T bottom_screw_frame → camera_color_optical_frame (nominal extrinsics)."""
    T_bs_link = _T_from_urdf_rpy_xyz(
        [_D435_MESH_X, _D435_DEPTH_PY, _D435_DEPTH_PZ], [0.0, 0.0, 0.0]
    )
    T_link_color = _T_from_urdf_rpy_xyz([0.0, _D435_COLOR_Y, 0.0], [0.0, 0.0, 0.0])
    T_color_opt = _T_from_urdf_rpy_xyz(
        [0.0, 0.0, 0.0], [-math.pi / 2, 0.0, -math.pi / 2]
    )
    return T_bs_link @ T_link_color @ T_color_opt


def compute_mount(T_cal: np.ndarray) -> np.ndarray:
    """T parent → bottom_screw so that parent→optical equals T_cal."""
    return T_cal @ np.linalg.inv(nominal_bottom_screw_to_color_optical())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "calibration_yaml",
        nargs="?",
        default=os.path.expanduser("~/.ros/hand_eye_calibration.yaml"),
        help="Path to hand_eye_calibration YAML",
    )
    p.add_argument(
        "--update-xacro",
        metavar="XACRO_PATH",
        help="Rewrite the mount joint <origin> in this xacro in place",
    )
    p.add_argument(
        "--joint-name",
        default=DEFAULT_JOINT_NAME,
        help=f"Mount joint name (default: {DEFAULT_JOINT_NAME})",
    )
    args = p.parse_args(argv)

    data = _load_yaml(args.calibration_yaml)
    ctype = data.get("calibration_type", "")
    if ctype and ctype != "eye-in-hand":
        print(
            "Warning: calibration_type is not eye-in-hand; still using YAML transform.",
            file=sys.stderr,
        )

    eff = data.get("robot_effector_frame", "")
    optical = data.get("tracking_base_frame", "")
    t = data["transform"]
    T_cal = _T_mat(
        Rot.from_quat([t["qx"], t["qy"], t["qz"], t["qw"]]).as_matrix(),
        np.array([t["tx"], t["ty"], t["tz"]], dtype=float),
    )
    T_mount = compute_mount(T_cal)
    xyz = T_mount[:3, 3]
    roll, pitch, yaw = _rotation_to_urdf_rpy(T_mount[:3, :3])

    print(f"# D435 mount from calibration: {eff} -> {optical}")
    print(f"# Joint {args.joint_name} (parent -> camera_bottom_screw_frame):")
    print(f'#   xyz="{xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f}"')
    print(f'#   rpy="{roll:.12f} {pitch:.12f} {yaw:.12f}"')

    if args.update_xacro:
        update_xacro_origin(args.update_xacro, xyz, (roll, pitch, yaw), args.joint_name)
        print(f"# Updated {args.update_xacro} (joint '{args.joint_name}').")
        print("# Rebuild seam_ros2_pkg (or piper_description) and restart the stack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
