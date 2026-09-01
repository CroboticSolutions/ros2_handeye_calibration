"""
Unit tests for the hand-eye math in CalibrationBackend, using synthetic data
generated from a known ground-truth transform so the tests do not depend on
ROS, real hardware, or a physical ChArUco board.

Run offline (no ROS):
  cd arms_ws/src/ros2_handeye_calibration
  python3 -m pytest test/test_calibration_backend.py -v
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from hand_eye_calibration.calibration_backend import CalibrationBackend


def _rt_to_matrix(R, t):
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


def _rt_to_list(R, t):
    q = Rot.from_matrix(R).as_quat()
    return [float(t[0]), float(t[1]), float(t[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _random_rt(rng, translation_scale=0.5):
    R = Rot.random(random_state=rng).as_matrix()
    t = rng.uniform(-translation_scale, translation_scale, size=3)
    return R, t


def _generate_eye_in_hand_dataset(n, rng, translation_scale=0.5):
    """
    Build a perfectly consistent eye-in-hand dataset for a known X = gripper2cam
    and a fixed board pose W = base2board:
        target2cam_i = inv(X) @ inv(base2gripper_i) @ W
    Returns (samples_robot, samples_tracking, X_R, X_t) where samples are
    [tx,ty,tz,qx,qy,qz,qw] lists (samples_robot = base2gripper_i,
    samples_tracking = target2cam_i), matching what node.py collects.
    """
    R_X, t_X = _random_rt(rng, translation_scale=0.15)
    R_W, t_W = _random_rt(rng, translation_scale=1.0)
    T_X = _rt_to_matrix(R_X, t_X)
    T_W = _rt_to_matrix(R_W, t_W)
    T_X_inv = np.linalg.inv(T_X)

    samples_robot = []
    samples_tracking = []
    for _ in range(n):
        R_g, t_g = _random_rt(rng, translation_scale=translation_scale)
        T_Gb = _rt_to_matrix(R_g, t_g)
        T_Ct = T_X_inv @ np.linalg.inv(T_Gb) @ T_W
        samples_robot.append(_rt_to_list(R_g, t_g))
        samples_tracking.append(_rt_to_list(T_Ct[:3, :3], T_Ct[:3, 3]))

    return samples_robot, samples_tracking, R_X, t_X


def _pose_error(cal, R_X, t_X):
    R_cal, t_cal = CalibrationBackend.list_to_opencv(cal)
    t_err = float(np.linalg.norm(t_cal - t_X))
    r_err_deg = float(np.degrees(Rot.from_matrix(R_X.T @ R_cal).magnitude()))
    return t_err, r_err_deg


class ComputeCalibrationTest(unittest.TestCase):
    def test_recovers_ground_truth_noiseless(self) -> None:
        rng = np.random.default_rng(0)
        samples_robot, samples_tracking, R_X, t_X = _generate_eye_in_hand_dataset(12, rng)

        cal = CalibrationBackend.compute_calibration(samples_robot, samples_tracking)
        t_err, r_err_deg = _pose_error(cal, R_X, t_X)

        self.assertLess(t_err, 1e-6, f"translation error too high: {t_err}")
        self.assertLess(r_err_deg, 1e-4, f"rotation error too high: {r_err_deg}")

    def test_detailed_reports_zero_residual_and_no_rejections(self) -> None:
        rng = np.random.default_rng(1)
        samples_robot, samples_tracking, R_X, t_X = _generate_eye_in_hand_dataset(10, rng)

        detail = CalibrationBackend.compute_calibration_detailed(samples_robot, samples_tracking)

        self.assertEqual(detail['rejected_indices'], [])
        self.assertIn(detail['algorithm_used'], CalibrationBackend.AVAILABLE_ALGORITHMS)
        self.assertLess(detail['residuals']['mean_translation_m'], 1e-6)
        self.assertLess(detail['residuals']['mean_rotation_deg'], 1e-4)

    def test_nonlinear_refinement_improves_noisy_estimate(self) -> None:
        """Refinement must actually reduce error, measured over many trials.

        A single trial proves nothing: closed-form and refined estimates can
        land either side of the truth by chance. Assert on the aggregate, which
        is the claim the refinement step actually makes.
        """
        closed_form_errors = []
        refined_errors = []
        wins = 0
        trials = 25

        for seed in range(trials):
            rng = np.random.default_rng(100 + seed)
            samples_robot, samples_tracking, R_X, t_X = _generate_eye_in_hand_dataset(15, rng)

            # Small, zero-mean noise on the tracking samples (ChArUco pose
            # jitter), applied to every sample so none is an outlier.
            noisy_tracking = []
            for s in samples_tracking:
                R_c, t_c = CalibrationBackend.list_to_opencv(s)
                t_noisy = t_c + rng.normal(scale=0.0015, size=3)
                small_rot = Rot.from_rotvec(rng.normal(scale=np.radians(0.4), size=3))
                noisy_tracking.append(_rt_to_list(small_rot.as_matrix() @ R_c, t_noisy))

            detail = CalibrationBackend.compute_calibration_detailed(samples_robot, noisy_tracking)
            cf = _pose_error(detail['closed_form_transform'], R_X, t_X)
            rf = _pose_error(detail['transform'], R_X, t_X)
            closed_form_errors.append(cf[0])
            refined_errors.append(rf[0])
            if rf[0] < cf[0]:
                wins += 1

        mean_cf = float(np.mean(closed_form_errors))
        mean_rf = float(np.mean(refined_errors))
        self.assertLess(
            mean_rf, mean_cf,
            f"refinement did not reduce mean translation error ({mean_rf:.5f} vs {mean_cf:.5f} m)")
        self.assertGreater(
            wins, trials // 2,
            f"refinement improved only {wins}/{trials} trials — not a reliable improvement")

    def test_refinement_recovers_from_a_perturbed_initial_guess(self) -> None:
        """Directly exercise the optimizer: start off-truth, expect convergence."""
        rng = np.random.default_rng(11)
        samples_robot, samples_tracking, R_X, t_X = _generate_eye_in_hand_dataset(12, rng)

        perturbed_R = Rot.from_rotvec(np.radians([3.0, -2.5, 2.0])).as_matrix() @ R_X
        perturbed_t = t_X + np.array([0.01, -0.008, 0.012])
        start_err = float(np.linalg.norm(perturbed_t - t_X))

        R_ref, t_ref, info = CalibrationBackend._refine_nonlinear(
            perturbed_R, perturbed_t, samples_robot, samples_tracking, list(range(12)))

        end_err = float(np.linalg.norm(t_ref - t_X))
        end_rot_deg = float(np.degrees(Rot.from_matrix(R_X.T @ R_ref).magnitude()))
        self.assertTrue(info['converged'])
        self.assertLess(end_err, start_err / 10.0)
        self.assertLess(end_rot_deg, 0.05)

    def test_unknown_algorithm_raises_value_error(self) -> None:
        rng = np.random.default_rng(12)
        samples_robot, samples_tracking, _, _ = _generate_eye_in_hand_dataset(6, rng)
        with self.assertRaises(ValueError):
            CalibrationBackend.compute_calibration_detailed(
                samples_robot, samples_tracking, algorithm='NotAnAlgorithm')

    def test_outlier_sample_is_rejected_and_estimate_stays_accurate(self) -> None:
        rng = np.random.default_rng(3)
        samples_robot, samples_tracking, R_X, t_X = _generate_eye_in_hand_dataset(12, rng)

        # Corrupt one sample's tracking pose with a large offset.
        R_bad, t_bad = CalibrationBackend.list_to_opencv(samples_tracking[5])
        t_bad_corrupted = t_bad + np.array([0.05, -0.04, 0.06])
        bad_rot = Rot.from_rotvec(np.radians([15.0, -10.0, 8.0])).as_matrix() @ R_bad
        samples_tracking[5] = _rt_to_list(bad_rot, t_bad_corrupted)

        detail = CalibrationBackend.compute_calibration_detailed(samples_robot, samples_tracking)

        self.assertIn(5, detail['rejected_indices'])
        t_err, r_err_deg = _pose_error(detail['transform'], R_X, t_X)
        self.assertLess(t_err, 1e-3)
        self.assertLess(r_err_deg, 0.2)

    def test_raises_below_min_samples(self) -> None:
        rng = np.random.default_rng(4)
        samples_robot, samples_tracking, _, _ = _generate_eye_in_hand_dataset(2, rng)
        with self.assertRaises(ValueError):
            CalibrationBackend.compute_calibration_detailed(samples_robot, samples_tracking)


class PairwiseResidualsTest(unittest.TestCase):
    def test_zero_for_consistent_data(self) -> None:
        rng = np.random.default_rng(5)
        samples_robot, samples_tracking, R_X, t_X = _generate_eye_in_hand_dataset(8, rng)
        cal = _rt_to_list(R_X, t_X)
        res = CalibrationBackend.pairwise_residuals(samples_robot, samples_tracking, cal)
        self.assertEqual(res['pair_count'], 8 * 7 // 2)
        self.assertLess(res['mean_translation_m'], 1e-9)
        self.assertLess(res['mean_rotation_deg'], 1e-6)

    def test_none_for_fewer_than_two_samples(self) -> None:
        rng = np.random.default_rng(6)
        samples_robot, samples_tracking, R_X, t_X = _generate_eye_in_hand_dataset(1, rng)
        cal = _rt_to_list(R_X, t_X)
        self.assertIsNone(CalibrationBackend.pairwise_residuals(samples_robot, samples_tracking, cal))


class AverageTransformsTest(unittest.TestCase):
    def test_recovers_true_pose_from_noisy_burst(self) -> None:
        rng = np.random.default_rng(7)
        R_true, t_true = _random_rt(rng, translation_scale=0.5)
        burst = []
        for _ in range(15):
            t_noisy = t_true + rng.normal(scale=0.001, size=3)
            small_rot = Rot.from_rotvec(rng.normal(scale=np.radians(0.3), size=3))
            R_noisy = small_rot.as_matrix() @ R_true
            burst.append(_rt_to_list(R_noisy, t_noisy))

        avg, spread = CalibrationBackend.average_transforms(burst)
        R_avg, t_avg = CalibrationBackend.list_to_opencv(avg)

        self.assertLess(float(np.linalg.norm(t_avg - t_true)), 0.001)
        self.assertLess(math_degrees_angle(R_true, R_avg), 0.2)
        self.assertGreater(spread['max_translation_dev_m'], 0.0)
        # Clean burst: the outlier gate must not eat good frames.
        self.assertEqual(spread['rejected_frames'], 0)
        self.assertEqual(spread['count'], 15)

    def test_rotation_outlier_frame_is_rejected(self) -> None:
        """A single badly-misdetected frame must not drag the averaged rotation.

        The chordal quaternion mean is a least-squares estimator, so without
        explicit rejection one bad frame biases rotation even though the
        median-based translation is unaffected.
        """
        rng = np.random.default_rng(21)
        R_true, t_true = _random_rt(rng, translation_scale=0.4)
        burst = []
        for _ in range(9):
            t_noisy = t_true + rng.normal(scale=0.0005, size=3)
            small_rot = Rot.from_rotvec(rng.normal(scale=np.radians(0.2), size=3))
            burst.append(_rt_to_list(small_rot.as_matrix() @ R_true, t_noisy))
        # One frame off by 25 degrees and 5 cm.
        bad_rot = Rot.from_rotvec(np.radians([25.0, 0.0, 0.0])).as_matrix() @ R_true
        burst.append(_rt_to_list(bad_rot, t_true + np.array([0.05, 0.0, 0.0])))

        avg, spread = CalibrationBackend.average_transforms(burst)
        R_avg, t_avg = CalibrationBackend.list_to_opencv(avg)

        self.assertGreaterEqual(spread['rejected_frames'], 1)
        self.assertLess(float(np.linalg.norm(t_avg - t_true)), 0.001)
        self.assertLess(
            math_degrees_angle(R_true, R_avg), 0.5,
            "rotation average was dragged by the outlier frame")

    def test_gate_never_strips_burst_below_minimum(self) -> None:
        """With mutually inconsistent frames the gate must not over-prune."""
        rng = np.random.default_rng(22)
        R_true, _ = _random_rt(rng)
        burst = [
            _rt_to_list(Rot.from_rotvec(np.radians([i * 12.0, 0, 0])).as_matrix() @ R_true,
                        np.array([i * 0.03, 0.0, 0.0]))
            for i in range(5)
        ]
        avg, spread = CalibrationBackend.average_transforms(burst)
        self.assertGreaterEqual(spread['count'], CalibrationBackend.BURST_MIN_KEEP)
        self.assertEqual(len(avg), 7)

    def test_single_sample_burst_is_identity(self) -> None:
        rng = np.random.default_rng(8)
        R_true, t_true = _random_rt(rng)
        sample = _rt_to_list(R_true, t_true)
        avg, spread = CalibrationBackend.average_transforms([sample])
        for a, b in zip(avg, sample):
            self.assertAlmostEqual(a, b, places=9)
        self.assertEqual(spread['count'], 1)
        self.assertEqual(spread['max_translation_dev_m'], 0.0)


def math_degrees_angle(R_a, R_b):
    return float(np.degrees(Rot.from_matrix(R_a.T @ R_b).magnitude()))


def _noisy_dataset(n, rng, R_X, t_X, noise_t=0.0015, noise_r=0.4):
    """Dataset for a GIVEN ground-truth X, with measurement noise on tracking."""
    R_W, t_W = _random_rt(rng, translation_scale=1.0)
    T_X = _rt_to_matrix(R_X, t_X)
    T_W = _rt_to_matrix(R_W, t_W)
    T_X_inv = np.linalg.inv(T_X)

    samples_robot, samples_tracking = [], []
    for _ in range(n):
        R_g, t_g = _random_rt(rng, translation_scale=0.5)
        T_Ct = T_X_inv @ np.linalg.inv(_rt_to_matrix(R_g, t_g)) @ T_W
        R_c = Rot.from_rotvec(rng.normal(scale=np.radians(noise_r), size=3)).as_matrix() @ T_Ct[:3, :3]
        t_c = T_Ct[:3, 3] + rng.normal(scale=noise_t, size=3)
        samples_robot.append(_rt_to_list(R_g, t_g))
        samples_tracking.append(_rt_to_list(R_c, t_c))
    return samples_robot, samples_tracking


class BootstrapUncertaintyTest(unittest.TestCase):
    def _fit_and_bootstrap(self, samples_robot, samples_tracking, n_bootstrap=16, seed=0):
        detail = CalibrationBackend.compute_calibration_detailed(samples_robot, samples_tracking)
        uncertainty = CalibrationBackend.bootstrap_uncertainty(
            samples_robot, samples_tracking, detail['transform'],
            detail['algorithm_used'], n_bootstrap=n_bootstrap, seed=seed)
        return detail, uncertainty

    def test_reports_expected_structure(self) -> None:
        rng = np.random.default_rng(31)
        R_X, t_X = _random_rt(rng, translation_scale=0.15)
        sr, st = _noisy_dataset(10, rng, R_X, t_X)
        _, u = self._fit_and_bootstrap(sr, st)

        self.assertIsNotNone(u)
        self.assertEqual(u['method'], 'bootstrap')
        self.assertEqual(len(u['translation_sigma_m']), 3)
        self.assertEqual(len(u['rotation_sigma_deg']), 3)
        self.assertEqual(len(u['worst_direction_axis']), 3)
        self.assertGreaterEqual(u['worst_direction_sigma_m'], u['best_direction_sigma_m'])
        self.assertAlmostEqual(float(np.linalg.norm(u['worst_direction_axis'])), 1.0, places=6)
        self.assertTrue(u['guidance'])

    def test_is_deterministic_for_a_given_seed(self) -> None:
        rng = np.random.default_rng(32)
        R_X, t_X = _random_rt(rng, translation_scale=0.15)
        sr, st = _noisy_dataset(10, rng, R_X, t_X)
        _, a = self._fit_and_bootstrap(sr, st, seed=7)
        _, b = self._fit_and_bootstrap(sr, st, seed=7)
        self.assertEqual(a['translation_sigma_m'], b['translation_sigma_m'])

    def test_lower_measurement_noise_gives_lower_uncertainty(self) -> None:
        R_X, t_X = _random_rt(np.random.default_rng(33), translation_scale=0.15)
        sr_noisy, st_noisy = _noisy_dataset(
            12, np.random.default_rng(40), R_X, t_X, noise_t=0.004, noise_r=1.2)
        sr_clean, st_clean = _noisy_dataset(
            12, np.random.default_rng(40), R_X, t_X, noise_t=0.0004, noise_r=0.1)

        _, u_noisy = self._fit_and_bootstrap(sr_noisy, st_noisy)
        _, u_clean = self._fit_and_bootstrap(sr_clean, st_clean)

        self.assertLess(
            u_clean['translation_sigma_rms_m'], u_noisy['translation_sigma_rms_m'],
            "cleaner measurements must report a tighter calibration")

    def test_returns_none_when_too_few_samples(self) -> None:
        rng = np.random.default_rng(34)
        R_X, t_X = _random_rt(rng, translation_scale=0.15)
        sr, st = _noisy_dataset(8, rng, R_X, t_X)
        detail = CalibrationBackend.compute_calibration_detailed(sr, st)
        self.assertIsNone(CalibrationBackend.bootstrap_uncertainty(
            sr[:4], st[:4], detail['transform'], detail['algorithm_used'], n_bootstrap=8))

    def test_one_sigma_interval_actually_covers_the_truth(self) -> None:
        """The claim under test: +/-1 sigma should contain the truth ~68% of the time.

        This is the only check that says the uncertainty number means anything.
        Without it we would be publishing a confident-looking value that has
        never been shown to match reality. The band is wide because the trial
        count here is kept small enough to run in CI; a systematically
        over-confident estimator (the analytic sigma^2 * (J^T J)^-1 formula
        measured ~51% coverage) still fails it.
        """
        trials = 14
        n_bootstrap = 12
        R_X, t_X = _random_rt(np.random.default_rng(50), translation_scale=0.15)

        inside = 0
        total = 0
        for trial in range(trials):
            rng = np.random.default_rng(6000 + trial)
            sr, st = _noisy_dataset(12, rng, R_X, t_X)
            detail, u = self._fit_and_bootstrap(sr, st, n_bootstrap=n_bootstrap, seed=trial)
            if u is None:
                continue
            estimate = np.array(detail['transform'][:3])
            sigma = np.array(u['translation_sigma_m'])
            inside += int(np.sum(np.abs(estimate - t_X) <= sigma))
            total += 3

        self.assertGreater(total, 0)
        coverage = inside / total
        self.assertGreater(
            coverage, 0.45,
            f"1-sigma coverage {coverage:.2f} is far below 0.68 — the uncertainty is over-confident")
        self.assertLess(
            coverage, 0.95,
            f"1-sigma coverage {coverage:.2f} is far above 0.68 — the uncertainty is over-inflated")


if __name__ == '__main__':
    unittest.main()
