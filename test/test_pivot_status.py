"""Unit tests for pivot calibration readiness status JSON."""

from __future__ import annotations

import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from hand_eye_calibration.pivot_backend import PivotCalibrationBackend
from hand_eye_calibration.pivot_status import (
    build_pivot_status,
    build_tool_tcp_status,
    status_to_json,
)

SPIKE = np.array([0.0, 0.0, 1.0])


def _alignment_samples(n_poses):
    """Flange poses holding a fixed tool axis collinear with SPIKE, differing
    only by roll about the spike."""
    r0 = Rot.from_euler("xyz", [15, -25, 35], degrees=True)
    out = []
    for i in range(n_poses):
        rz = Rot.from_rotvec(np.radians(30.0 * i) * SPIKE)
        q = (rz * r0).as_quat()
        out.append([0.0, 0.0, 0.0, q[0], q[1], q[2], q[3]])
    return out

DIVERSE_RPYS = [
    (0, 0, 0), (35, 0, 0), (-35, 20, 0), (0, 40, 15),
    (20, -30, 40), (-25, 25, -25), (45, 10, -20), (-10, -40, 30),
    (15, 15, 15), (-15, -15, -15),
]

TRUE_TCP = np.array([0.01, 0.0, 0.15])
TRUE_FIXED_POINT = np.array([0.3, 0.0, 0.4])


def _make_sample(rpy_deg, tcp=TRUE_TCP, fixed_point=TRUE_FIXED_POINT):
    rot = Rot.from_euler("xyz", rpy_deg, degrees=True)
    r = rot.as_matrix()
    q = fixed_point - r @ tcp
    quat = rot.as_quat()
    return [q[0], q[1], q[2], quat[0], quat[1], quat[2], quat[3]]


class PivotStatusTest(unittest.TestCase):
    def test_not_ready_with_few_samples(self) -> None:
        samples = [_make_sample(rpy) for rpy in [(0, 0, 0), (10, 0, 0)]]
        status = build_pivot_status(flange_samples=samples, pivot=None)
        self.assertEqual(status["readiness"], "not_ready")
        self.assertFalse(status["ready_to_save"])
        self.assertEqual(len(status["samples"]), 2)
        self.assertIsNone(status["samples"][0]["residual_m"])
        self.assertNotIn("fixed_point", status)

    def test_viz_geometry_with_pivot(self) -> None:
        samples = [_make_sample(rpy) for rpy in DIVERSE_RPYS]
        pivot = PivotCalibrationBackend.compute_pivot(samples)
        status = build_pivot_status(flange_samples=samples, pivot=pivot)
        self.assertEqual(len(status["samples"]), len(samples))
        self.assertAlmostEqual(status["samples"][0]["x"], samples[0][0], places=9)
        self.assertIsNotNone(status["samples"][0]["residual_m"])
        self.assertIn("fixed_point", status)
        self.assertAlmostEqual(
            status["fixed_point"]["x"], TRUE_FIXED_POINT[0], places=5
        )

    def test_collecting_when_orientations_too_similar(self) -> None:
        rpys = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
        samples = [_make_sample(rpy) for rpy in rpys]
        pivot = PivotCalibrationBackend.compute_pivot(samples)
        status = build_pivot_status(flange_samples=samples, pivot=pivot)
        self.assertEqual(status["readiness"], "collecting")
        self.assertTrue(any("too similar" in g or "orientation" in g.lower() for g in status["guidance"]))

    def test_excellent_with_diverse_low_residual_samples(self) -> None:
        rpys = [
            (0, 0, 0), (35, 0, 0), (-35, 20, 0), (0, 40, 15),
            (20, -30, 40), (-25, 25, -25), (45, 10, -20), (-10, -40, 30),
            (15, 15, 15), (-15, -15, -15),
        ]
        samples = [_make_sample(rpy) for rpy in rpys]
        pivot = PivotCalibrationBackend.compute_pivot(samples)
        status = build_pivot_status(flange_samples=samples, pivot=pivot)
        self.assertEqual(status["readiness"], "excellent")
        self.assertTrue(status["ready_to_save"])
        parsed = json.loads(status_to_json(status))
        self.assertEqual(parsed["sample_count"], 10)
        self.assertAlmostEqual(parsed["estimate"]["tx"], TRUE_TCP[0], places=6)

    def test_last_sample_residual_flagged_when_touch_slipped(self) -> None:
        rpys = [
            (0, 0, 0), (35, 0, 0), (-35, 20, 0), (0, 40, 15),
            (20, -30, 40), (-25, 25, -25),
        ]
        samples = [_make_sample(rpy) for rpy in rpys]
        bad = list(samples[-1])
        bad[1] += 0.01
        samples[-1] = bad
        pivot = PivotCalibrationBackend.compute_pivot(samples)
        status = build_pivot_status(flange_samples=samples, pivot=pivot)
        self.assertIsNotNone(status["last_sample_residual_m"])
        self.assertGreater(status["last_sample_residual_m"], 0.003)


class ToolTcpStatusTest(unittest.TestCase):
    def _tip_round(self, tcp=TRUE_TCP):
        samples = [_make_sample(rpy, tcp=tcp) for rpy in DIVERSE_RPYS]
        pivot = PivotCalibrationBackend.compute_pivot(samples)
        return samples, pivot

    def test_position_mode_save_gates_on_tip_round(self) -> None:
        tip_samples, tip_pivot = self._tip_round()
        status = build_tool_tcp_status(
            mode="position",
            active_round="tip",
            tip_samples=tip_samples,
            tip_pivot=tip_pivot,
            align_samples=[],
            axis=None,
        )
        self.assertEqual(status["mode"], "position")
        self.assertTrue(status["ready_to_save"])
        self.assertFalse(status["axis"]["computed"])

    def test_axis_mode_not_ready_until_axis_computed(self) -> None:
        tip_samples, tip_pivot = self._tip_round()
        align_samples = _alignment_samples(3)

        # Tip ready and alignment poses captured, but axis not computed yet.
        status = build_tool_tcp_status(
            mode="axis",
            active_round="axis_ref",
            tip_samples=tip_samples,
            tip_pivot=tip_pivot,
            align_samples=align_samples,
            axis=None,
        )
        self.assertFalse(status["ready_to_save"])
        self.assertFalse(status["axis"]["computed"])
        self.assertFalse(status["axis"]["ok"])

        axis = PivotCalibrationBackend.compute_axis_from_alignment(
            alignment_samples=align_samples,
            spike_axis_base=SPIKE,
            tip_translation=tip_pivot["tcp_translation"],
        )
        status = build_tool_tcp_status(
            mode="axis",
            active_round="axis_ref",
            tip_samples=tip_samples,
            tip_pivot=tip_pivot,
            align_samples=align_samples,
            axis=axis,
        )
        self.assertTrue(status["axis"]["computed"])
        self.assertTrue(status["axis"]["ok"])
        self.assertTrue(status["ready_to_save"])
        parsed = json.loads(status_to_json(status))
        self.assertIn("quaternion", parsed["axis"])
        self.assertIn("spike_axis_base", parsed["axis"])
        self.assertEqual(parsed["axis"]["sample_count"], 3)
        self.assertIn("tip_fixed_point", parsed["axis"])
        self.assertEqual(len(parsed["tip"]["samples"]), len(tip_samples))
        self.assertEqual(len(parsed["axis_ref"]["samples"]), len(align_samples))

    def test_single_alignment_pose_still_ready(self) -> None:
        tip_samples, tip_pivot = self._tip_round()
        align_samples = _alignment_samples(1)
        axis = PivotCalibrationBackend.compute_axis_from_alignment(
            alignment_samples=align_samples,
            spike_axis_base=SPIKE,
            tip_translation=tip_pivot["tcp_translation"],
        )
        status = build_tool_tcp_status(
            mode="axis",
            active_round="axis_ref",
            tip_samples=tip_samples,
            tip_pivot=tip_pivot,
            align_samples=align_samples,
            axis=axis,
        )
        self.assertTrue(status["axis"]["ok"])
        self.assertTrue(status["ready_to_save"])


if __name__ == "__main__":
    unittest.main()
