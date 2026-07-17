"""
Unit tests for pivot (tool-tip) calibration, using synthetic flange poses with
a known ground-truth TCP translation and pivot point.

Run offline (no ROS):
  cd arms_ws/src/ros2_handeye_calibration
  python3 -m pytest test/test_pivot_backend.py -v
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from hand_eye_calibration.pivot_backend import PivotCalibrationBackend

# Ground truth: flange -> tool_tip vector, and the fixed point the tip touches
# (both arbitrary but non-trivial so a bug in axis handling would show up).
TRUE_TCP = np.array([0.012, -0.006, 0.148])
TRUE_FIXED_POINT = np.array([0.35, 0.05, 0.42])

# A deterministic, reasonably diverse set of wrist orientations (roll/pitch/yaw
# in degrees). Fixed rather than random so the test is not flaky.
ORIENTATIONS_DEG = [
    (0, 0, 0),
    (35, 0, 0),
    (-35, 20, 0),
    (0, 40, 15),
    (20, -30, 40),
    (-25, 25, -25),
    (45, 10, -20),
    (-10, -40, 30),
]


def _make_sample(rpy_deg, tcp=TRUE_TCP, fixed_point=TRUE_FIXED_POINT):
    rot = Rot.from_euler("xyz", rpy_deg, degrees=True)
    r = rot.as_matrix()
    # p_tip = R @ t + q  =>  q = p_tip - R @ t
    q = fixed_point - r @ tcp
    quat = rot.as_quat()  # scalar-last, matches node samples
    return [q[0], q[1], q[2], quat[0], quat[1], quat[2], quat[3]]


class PivotBackendTest(unittest.TestCase):
    def test_recovers_known_tcp_and_fixed_point(self) -> None:
        samples = [_make_sample(rpy) for rpy in ORIENTATIONS_DEG]
        result = PivotCalibrationBackend.compute_pivot(samples)

        np.testing.assert_allclose(
            result["tcp_translation"], TRUE_TCP, atol=1e-9
        )
        np.testing.assert_allclose(
            result["fixed_point"], TRUE_FIXED_POINT, atol=1e-9
        )
        self.assertLess(result["rms_residual_m"], 1e-9)
        self.assertLess(result["max_residual_m"], 1e-9)
        self.assertEqual(len(result["per_sample_residuals_m"]), len(samples))

    def test_flags_one_bad_sample_via_per_sample_residual(self) -> None:
        samples = [_make_sample(rpy) for rpy in ORIENTATIONS_DEG]
        # Corrupt one sample as if the tip slipped 5 mm off the pivot point.
        bad = list(samples[3])
        bad[1] += 0.005
        samples[3] = bad

        result = PivotCalibrationBackend.compute_pivot(samples)
        residuals = result["per_sample_residuals_m"]
        worst_idx = int(np.argmax(residuals))
        self.assertEqual(worst_idx, 3)
        self.assertGreater(residuals[3], 0.001)

    def test_raises_on_too_few_samples(self) -> None:
        samples = [_make_sample(rpy) for rpy in ORIENTATIONS_DEG[:3]]
        with self.assertRaises(ValueError):
            PivotCalibrationBackend.compute_pivot(samples)

    def test_condition_number_is_worse_with_similar_orientations(self) -> None:
        diverse = [_make_sample(rpy) for rpy in ORIENTATIONS_DEG]
        near_identical = [
            _make_sample(rpy)
            for rpy in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        ]

        diverse_result = PivotCalibrationBackend.compute_pivot(diverse)
        similar_result = PivotCalibrationBackend.compute_pivot(near_identical)

        self.assertGreater(
            similar_result["condition_number"], diverse_result["condition_number"]
        )


class FrameFromAxisTest(unittest.TestCase):
    def test_quaternion_plus_z_is_the_axis(self) -> None:
        axis = np.array([0.1, -0.2, 0.97])
        axis = axis / np.linalg.norm(axis)
        result = PivotCalibrationBackend.frame_from_axis(axis)
        np.testing.assert_allclose(result["axis_dir"], axis, atol=1e-12)
        rot = Rot.from_quat(result["quaternion"]).as_matrix()
        np.testing.assert_allclose(rot[:, 2], axis, atol=1e-9)
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=9)

    def test_raises_on_zero_axis(self) -> None:
        with self.assertRaises(ValueError):
            PivotCalibrationBackend.frame_from_axis([0.0, 0.0, 0.0])

    def test_axis_parallel_to_flange_x_still_orthonormal(self) -> None:
        # Axis along flange X exercises the Gram-Schmidt fallback hint.
        result = PivotCalibrationBackend.frame_from_axis([1.0, 0.0, 0.0])
        rot = Rot.from_quat(result["quaternion"]).as_matrix()
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=9)
        np.testing.assert_allclose(rot[:, 2], [1.0, 0.0, 0.0], atol=1e-9)
        # Convention: TCP +X/+Y flipped 180° about the tool axis.
        np.testing.assert_allclose(rot[:, 1], [0.0, 1.0, 0.0], atol=1e-9)

    def test_axis_along_flange_z_flips_xy_about_tool_axis(self) -> None:
        result = PivotCalibrationBackend.frame_from_axis([0.0, 0.0, 1.0])
        rot = Rot.from_quat(result["quaternion"]).as_matrix()
        np.testing.assert_allclose(rot[:, 2], [0.0, 0.0, 1.0], atol=1e-9)
        # TCP +X/+Y point to the opposite side of the flange's X/Y.
        np.testing.assert_allclose(rot[:, 0], [-1.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(rot[:, 1], [0.0, -1.0, 0.0], atol=1e-9)


# Known spike direction in the base frame (vertical).
SPIKE = np.array([0.0, 0.0, 1.0])


def _align_sample(rot: Rot):
    """Flange pose (base frame) as a 7-tuple; translation is irrelevant to the axis."""
    q = rot.as_quat()
    return [0.0, 0.0, 0.0, q[0], q[1], q[2], q[3]]


def _alignment_poses(a_true, roll_degs):
    """Flange poses that all hold tool axis ``a_true`` (flange frame) collinear
    with SPIKE, differing only by roll about the spike (physically free)."""
    # R0 maps a_true -> SPIKE, i.e. R0ᵀ @ SPIKE == a_true.
    r0 = Rot.from_euler("xyz", [20, -35, 40], degrees=True)
    # a_true is defined from r0 so the construction is exactly consistent.
    for roll in roll_degs:
        rz = Rot.from_rotvec(np.radians(roll) * SPIKE)
        yield _align_sample(rz * r0)


class AlignAxisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.r0 = Rot.from_euler("xyz", [20, -35, 40], degrees=True)
        # Tool axis in the flange frame that maps onto the vertical spike.
        self.a_true = self.r0.as_matrix().T @ SPIKE

    def test_recovers_axis_from_aligned_poses(self) -> None:
        samples = list(_alignment_poses(self.a_true, [0, 30, -45, 90]))
        result = PivotCalibrationBackend.compute_axis_from_alignment(
            alignment_samples=samples,
            spike_axis_base=SPIKE,
            tip_translation=self.a_true * 0.15,
        )
        np.testing.assert_allclose(result["axis_dir"], self.a_true, atol=1e-9)
        self.assertLess(result["alignment_spread_deg"], 1e-6)
        self.assertEqual(result["sample_count"], 4)
        rot = Rot.from_quat(result["quaternion"]).as_matrix()
        np.testing.assert_allclose(rot[:, 2], self.a_true, atol=1e-9)

    def test_tip_translation_fixes_the_sign(self) -> None:
        samples = list(_alignment_poses(self.a_true, [0, 45]))
        pos = PivotCalibrationBackend.compute_axis_from_alignment(
            samples, spike_axis_base=SPIKE, tip_translation=self.a_true * 0.15
        )
        neg = PivotCalibrationBackend.compute_axis_from_alignment(
            samples, spike_axis_base=SPIKE, tip_translation=-self.a_true * 0.15
        )
        np.testing.assert_allclose(pos["axis_dir"], self.a_true, atol=1e-9)
        np.testing.assert_allclose(neg["axis_dir"], -self.a_true, atol=1e-9)

    def test_spread_reflects_misalignment(self) -> None:
        good = list(_alignment_poses(self.a_true, [0, 30, 60]))
        # Perturb one pose by ~2° about a spike-orthogonal axis so its recovered
        # direction tilts off the mean.
        tilt = Rot.from_rotvec(np.radians(2.0) * np.array([1.0, 0.0, 0.0]))
        perturbed = Rot.from_euler("xyz", [20, -35, 40], degrees=True)
        bad_sample = _align_sample(tilt * perturbed)
        samples = good + [bad_sample]
        result = PivotCalibrationBackend.compute_axis_from_alignment(
            samples, spike_axis_base=SPIKE, tip_translation=self.a_true * 0.15
        )
        self.assertGreater(result["alignment_spread_deg"], 0.5)

    def test_single_pose_is_enough(self) -> None:
        samples = list(_alignment_poses(self.a_true, [0]))
        result = PivotCalibrationBackend.compute_axis_from_alignment(
            samples, spike_axis_base=SPIKE, tip_translation=self.a_true * 0.15
        )
        self.assertEqual(result["sample_count"], 1)
        np.testing.assert_allclose(result["axis_dir"], self.a_true, atol=1e-9)

    def test_raises_on_no_poses_or_zero_spike(self) -> None:
        with self.assertRaises(ValueError):
            PivotCalibrationBackend.compute_axis_from_alignment([], spike_axis_base=SPIKE)
        samples = list(_alignment_poses(self.a_true, [0]))
        with self.assertRaises(ValueError):
            PivotCalibrationBackend.compute_axis_from_alignment(
                samples, spike_axis_base=[0.0, 0.0, 0.0]
            )


if __name__ == "__main__":
    unittest.main()
