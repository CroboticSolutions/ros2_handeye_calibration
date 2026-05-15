#!/usr/bin/env python3
"""
Convert a saved hand-eye YAML (eye-in-hand, optical frame as tracking_base_frame)
into depthai_ros_driver URDF mount arguments.

During calibration, OpenCV returns T_effector_optical. depthai_descriptions publishes
T_parent_optical_nominal when cam_pos on the center joint is zero (parent → oak-d_frame → …).

You want T_parent_optical_calibrated = T_mount @ T_nom, with parent attached to link6, so:

    T_mount = T_cal @ inv(T_nom)

Pass the printed cam_pos_* / cam_roll / cam_pitch / cam_yaw into oak_depthai_sr_rgbd.launch.py
with parent_frame:=link6 (or your robot_effector_frame).

Limitations:
- Supported nominal chains: OAK-D-SR stereo layout matching depthai_descriptions (baseline 0.02).
- Assumes calibration was captured while the depthai center joint used identity cam_pos (typical if you
  used a separate static TF into oak-d-base-frame during calibration).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot


def _T_mat(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = t
    return T


def _T_from_urdf_rpy_xyz(xyz: list[float], rpy: list[float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw about X, Y, Z (same as scipy euler 'xyz')."""
    return _T_mat(Rot.from_euler("xyz", rpy).as_matrix(), np.array(xyz, dtype=float))


def _nominal_parent_to_optical_oak_d_sr() -> np.ndarray:
    """Matches depthai_macro.urdf.xacro for OAK-D-SR (baseline 0.02 m)."""
    baseline = 0.02
    # oak-d_frame -> oak_right_camera_frame
    T_base_rcam = _T_from_urdf_rpy_xyz([0.0, -baseline / 2.0, 0.0], [0.0, 0.0, 0.0])
    # camera_frame -> optical: rpy="-pi/2 0 -pi/2"
    T_rcam_opt = _T_from_urdf_rpy_xyz([0.0, 0.0, 0.0], [-math.pi / 2, 0.0, -math.pi / 2])
    return T_base_rcam @ T_rcam_opt


def _rotation_to_urdf_rpy(Rm: np.ndarray) -> tuple[float, float, float]:
    """Match hand_eye_calibration.node.tf_to_urdf_tf convention."""
    qx, qy, qz, qw = Rot.from_matrix(Rm).as_quat()
    e = list(Rot.from_quat([qx, qy, qz, qw]).as_euler(seq="ZYX"))
    roll, pitch, yaw = e[2], e[1], e[0]
    return float(roll), float(pitch), float(yaw)


def _load_yaml(path: str) -> dict:
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "calibration_yaml",
        nargs="?",
        default=os.path.expanduser("~/.ros/hand_eye_calibration.yaml"),
        help="Path to hand_eye_calibration YAML (default: ~/.ros/hand_eye_calibration.yaml)",
    )
    p.add_argument(
        "--camera-model",
        default="OAK-D-SR",
        choices=("OAK-D-SR",),
        help="depthai_descriptions model used for nominal chain (default: OAK-D-SR)",
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
    t = data["transform"]
    T_cal = _T_mat(
        Rot.from_quat([t["qx"], t["qy"], t["qz"], t["qw"]]).as_matrix(),
        np.array([t["tx"], t["ty"], t["tz"]], dtype=float),
    )

    if args.camera_model == "OAK-D-SR":
        T_nom = _nominal_parent_to_optical_oak_d_sr()
    else:
        raise SystemExit("Unsupported camera model")

    T_mount = T_cal @ np.linalg.inv(T_nom)
    xyz = T_mount[:3, 3]
    roll, pitch, yaw = _rotation_to_urdf_rpy(T_mount[:3, :3])

    print("# Paste after: ros2 launch arm_api2 oak_depthai_sr_rgbd.launch.py \\\n#   parent_frame:=%s \\" % eff)
    print(f"    cam_pos_x:={xyz[0]:.12f} \\")
    print(f"    cam_pos_y:={xyz[1]:.12f} \\")
    print(f"    cam_pos_z:={xyz[2]:.12f} \\")
    print(f"    cam_roll:={roll:.12f} \\")
    print(f"    cam_pitch:={pitch:.12f} \\")
    print(f"    cam_yaw:={yaw:.12f}")
    print("# Expected TF tree: %s -> (oak URDF parent link) -> … -> %s" % (eff, optical))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
