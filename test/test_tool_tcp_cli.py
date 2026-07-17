"""
Tests for tool_tcp_cli: YAML -> xacro joint (flange -> tool TCP), both creating
a fresh xacro file and updating an existing one.

Run offline (no ROS):
  cd arms_ws/src/ros2_handeye_calibration
  python3 -m pytest test/test_tool_tcp_cli.py -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

from hand_eye_calibration.tool_tcp_cli import main as cli_main

CAL_YAML = {
    "parent_frame": "link6",
    "tcp_name": "welding_tcp",
    "robot_base_frame": "base_link",
    "robot_flange_frame": "link6",
    "sample_count": 10,
    "condition_number": 4.2,
    "rms_residual_m": 0.0008,
    "max_residual_m": 0.0015,
    "transform": {
        "tx": 0.012, "ty": -0.006, "tz": 0.148,
        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
    },
}


class ToolTcpCliTest(unittest.TestCase):
    def test_creates_fresh_xacro_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "tool_tcp_calibration.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(CAL_YAML, f)
            xacro_path = Path(tmp) / "piper_tool_tcp_macros.xacro"

            rc = cli_main([str(yaml_path), "--update-xacro", str(xacro_path)])
            self.assertEqual(rc, 0)
            self.assertTrue(xacro_path.is_file())

            content = xacro_path.read_text()
            self.assertIn('name="piper_tool_tcp_mount"', content)
            self.assertIn('child link="welding_tcp"', content)
            self.assertIn('xyz="0.012000000000 -0.006000000000 0.148000000000"', content)
            # Identity rotation -> rpy all zero.
            self.assertIn('rpy="0.000000000000 0.000000000000 0.000000000000"', content)

    def test_updates_existing_xacro_origin_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "tool_tcp_calibration.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(CAL_YAML, f)
            xacro_path = Path(tmp) / "piper_tool_tcp_macros.xacro"

            # First run creates the file with a zero-ish placeholder origin.
            cli_main([str(yaml_path), "--update-xacro", str(xacro_path)])
            first_content = xacro_path.read_text()

            # Re-calibrate with a different translation and re-run: origin must change,
            # joint/macro scaffolding must be preserved.
            recalibrated = dict(CAL_YAML)
            recalibrated["transform"] = dict(CAL_YAML["transform"])
            recalibrated["transform"]["tx"] = 0.02
            with open(yaml_path, "w") as f:
                yaml.dump(recalibrated, f)

            rc = cli_main([str(yaml_path), "--update-xacro", str(xacro_path)])
            self.assertEqual(rc, 0)
            second_content = xacro_path.read_text()

            self.assertNotEqual(first_content, second_content)
            self.assertIn('xyz="0.020000000000 -0.006000000000 0.148000000000"', second_content)
            # Macro/joint scaffolding untouched by the origin-only rewrite.
            self.assertIn('name="piper_tool_tcp_mount"', second_content)
            self.assertIn('child link="welding_tcp"', second_content)

    def test_orientation_round_trips_through_urdf_rpy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = dict(CAL_YAML)
            quat = Rot.from_euler("xyz", [10, 20, 30], degrees=True).as_quat()
            data["transform"] = {
                "tx": 0.01, "ty": 0.02, "tz": 0.03,
                "qx": float(quat[0]), "qy": float(quat[1]),
                "qz": float(quat[2]), "qw": float(quat[3]),
            }
            yaml_path = Path(tmp) / "cal.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(data, f)
            xacro_path = Path(tmp) / "mount.xacro"

            cli_main([str(yaml_path), "--update-xacro", str(xacro_path)])
            content = xacro_path.read_text()

            import re
            m = re.search(r'rpy="([\d.eE+-]+) ([\d.eE+-]+) ([\d.eE+-]+)"', content)
            self.assertIsNotNone(m)
            rpy = [float(v) for v in m.groups()]
            rebuilt = Rot.from_euler("xyz", rpy).as_matrix()
            expected = Rot.from_quat(quat).as_matrix()
            np.testing.assert_allclose(rebuilt, expected, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
