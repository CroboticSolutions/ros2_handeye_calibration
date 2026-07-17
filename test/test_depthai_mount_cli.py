"""
Tests for depthai_mount_cli: nominal chains, mount composition, xacro rewrite.

Run offline (no ROS):
  cd arms_ws/src/ros2_handeye_calibration
  python3 -m pytest test/test_depthai_mount_cli.py -v
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from hand_eye_calibration.depthai_mount_cli import (
    _T_from_urdf_rpy_xyz,
    _T_mat,
    _rotation_to_urdf_rpy,
    compute_mount,
    infer_socket,
    nominal_parent_to_optical,
    update_xacro_origin,
)

# The values hand-derived for the Piper mount from the 2026-05 calibration
# (link6 -> oak_right_camera_optical_frame) with the PRO-W 75 mm baseline.
LEGACY_CAL = {
    "t": [-0.0916, -0.0015, 0.0632],
    "q": [0.1157, -0.1322, 0.7206, -0.6708],
}
EXPECTED_MOUNT_XYZ = [-0.088848448438, 0.035896799479, 0.063597935575]
EXPECTED_MOUNT_RPY = [0.030685050721, -1.217622058280, -0.102237086738]


def _cal_matrix(t, q):
    return _T_mat(Rot.from_quat(q).as_matrix(), np.array(t, dtype=float))


class NominalChainTest(unittest.TestCase):
    def test_pro_w_rgb_is_pure_optical_rotation(self) -> None:
        T = nominal_parent_to_optical("OAK-D-PRO-W", "rgb")
        np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, 0.0], atol=1e-12)
        expected_rot = Rot.from_euler(
            "xyz", [-math.pi / 2, 0.0, -math.pi / 2]
        ).as_matrix()
        np.testing.assert_allclose(T[:3, :3], expected_rot, atol=1e-12)

    def test_pro_w_stereo_offsets(self) -> None:
        T_left = nominal_parent_to_optical("OAK-D-PRO-W", "left")
        T_right = nominal_parent_to_optical("OAK-D-PRO-W", "right")
        np.testing.assert_allclose(T_left[:3, 3], [0.0, 0.0375, 0.0], atol=1e-12)
        np.testing.assert_allclose(T_right[:3, 3], [0.0, -0.0375, 0.0], atol=1e-12)

    def test_sr_has_no_rgb_socket(self) -> None:
        with self.assertRaises(ValueError):
            nominal_parent_to_optical("OAK-D-SR", "rgb")
        T_right = nominal_parent_to_optical("OAK-D-SR", "right")
        np.testing.assert_allclose(T_right[:3, 3], [0.0, -0.01, 0.0], atol=1e-12)

    def test_infer_socket(self) -> None:
        self.assertEqual(infer_socket("oak_rgb_camera_optical_frame"), "rgb")
        self.assertEqual(infer_socket("oak_right_camera_optical_frame"), "right")
        self.assertEqual(infer_socket("oak_left_camera_optical_frame"), "left")
        self.assertIsNone(infer_socket("camera_optical_frame"))


class MountCompositionTest(unittest.TestCase):
    def test_reproduces_piper_xacro_mount_from_legacy_calibration(self) -> None:
        T_cal = _cal_matrix(LEGACY_CAL["t"], LEGACY_CAL["q"])
        T_mount = compute_mount(T_cal, "OAK-D-PRO-W", "right")
        np.testing.assert_allclose(T_mount[:3, 3], EXPECTED_MOUNT_XYZ, atol=1e-9)
        rpy = _rotation_to_urdf_rpy(T_mount[:3, :3])
        np.testing.assert_allclose(rpy, EXPECTED_MOUNT_RPY, atol=1e-9)

    def test_mount_times_nominal_equals_calibration(self) -> None:
        """Round trip: T_mount @ T_nom must reproduce T_cal for every socket."""
        T_cal = _cal_matrix([0.05, -0.02, 0.1], Rot.random(random_state=7).as_quat())
        for model, socket in (
            ("OAK-D-PRO-W", "rgb"),
            ("OAK-D-PRO-W", "right"),
            ("OAK-D-PRO-W", "left"),
            ("OAK-D-SR", "right"),
        ):
            T_mount = compute_mount(T_cal, model, socket)
            T_round = T_mount @ nominal_parent_to_optical(model, socket)
            np.testing.assert_allclose(T_round, T_cal, atol=1e-12)

    def test_urdf_rpy_round_trip(self) -> None:
        Rm = Rot.random(random_state=3).as_matrix()
        rpy = _rotation_to_urdf_rpy(Rm)
        np.testing.assert_allclose(
            _T_from_urdf_rpy_xyz([0, 0, 0], list(rpy))[:3, :3], Rm, atol=1e-12
        )


XACRO_SNIPPET = """<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="test">
  <xacro:macro name="mount" params="parent_link">
    <link name="oak-d-base-frame"/>
    <joint name="piper_hand_eye_link6_mount" type="fixed">
      <origin
        xyz="-0.088848448438 0.035896799479 0.063597935575"
        rpy="0.030685050721 -1.217622058280 -0.102237086738"/>
      <parent link="${parent_link}"/>
      <child link="oak-d-base-frame"/>
    </joint>
  </xacro:macro>
</robot>
"""


class XacroUpdateTest(unittest.TestCase):
    def test_rewrites_origin_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mount.xacro"
            path.write_text(XACRO_SNIPPET, encoding="utf-8")
            update_xacro_origin(
                str(path), np.array([0.1, 0.2, 0.3]), (0.4, 0.5, 0.6)
            )
            content = path.read_text(encoding="utf-8")
            self.assertIn('xyz="0.100000000000 0.200000000000 0.300000000000"', content)
            self.assertIn('rpy="0.400000000000 0.500000000000 0.600000000000"', content)
            self.assertNotIn("-0.088848448438", content)
            # still parseable XML
            import xml.dom.minidom

            xml.dom.minidom.parseString(content)

    def test_missing_joint_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mount.xacro"
            path.write_text(XACRO_SNIPPET, encoding="utf-8")
            with self.assertRaises(SystemExit):
                update_xacro_origin(
                    str(path), np.zeros(3), (0.0, 0.0, 0.0), joint_name="nope"
                )


if __name__ == "__main__":
    unittest.main()
