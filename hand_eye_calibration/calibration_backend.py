import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


class CalibrationBackend:
    MIN_SAMPLES = 4

    AVAILABLE_ALGORITHMS = {
        'Tsai-Lenz': cv2.CALIB_HAND_EYE_TSAI,
        'Park': cv2.CALIB_HAND_EYE_PARK,
        'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
        'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
        'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    # Normalization scales used to combine translation (m) and rotation (deg)
    # residuals into one dimensionless score, for algorithm selection, outlier
    # scoring and nonlinear-refinement weighting. Chosen to roughly match the
    # expected noise floor of ChArUco-based pose estimation at typical working
    # distances (a few mm / ~1 degree).
    TRANS_SCALE_M = 0.003
    ROT_SCALE_DEG = 1.0
    ROT_SCALE_RAD = math.radians(ROT_SCALE_DEG)

    OUTLIER_MAD_K = 3.5
    OUTLIER_SCORE_FLOOR = 1.0
    MAX_REJECT_FRACTION = 0.3

    # Burst-frame outlier gating (one capture point -> several camera frames).
    # The floors matter: with very consistent frames the MAD collapses to ~0,
    # and a pure median+K*MAD gate would then reject perfectly good frames.
    # Nothing below these deviations is worth rejecting.
    BURST_MAD_K = 3.5
    BURST_TRANS_FLOOR_M = 0.002
    BURST_ROT_FLOOR_DEG = 0.5
    BURST_MIN_KEEP = 3

    # ------------------------------------------------------------------
    # Basic conversions
    # ------------------------------------------------------------------
    @staticmethod
    def list_to_opencv(transform: list()):
        """
        transform = [tx, ty, tz, qx, qy, qz, qw]
        """
        t = transform
        tr = np.array([t[0], t[1], t[2]], dtype=np.float64)
        rot = Rot.from_quat([t[3], t[4], t[5], t[6]]).as_matrix()
        return rot, tr

    @staticmethod
    def _rot_tr_to_list(rot, tr):
        rot = np.asarray(rot, dtype=np.float64)
        tr = np.asarray(tr, dtype=np.float64).reshape(-1)
        qx, qy, qz, qw = (float(v) for v in Rot.from_matrix(rot).as_quat())
        return [float(tr[0]), float(tr[1]), float(tr[2]), qx, qy, qz, qw]

    @staticmethod
    def _to_rot_tr_arrays(samples):
        rots, trs = [], []
        for s in samples:
            r, t = CalibrationBackend.list_to_opencv(s)
            rots.append(r)
            trs.append(t)
        return rots, trs

    @staticmethod
    def _all_pairs(n):
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    @staticmethod
    def _mad_keep_mask(deviations, floor):
        """Median-absolute-deviation gate with a floor (see BURST_* constants)."""
        med = float(np.median(deviations))
        mad = float(np.median(np.abs(deviations - med)))
        threshold = max(med + CalibrationBackend.BURST_MAD_K * mad, floor)
        return deviations <= threshold

    @staticmethod
    def _burst_centre(translations, quats):
        """Component-wise median translation + chordal mean rotation."""
        centre_t = np.median(translations, axis=0)
        centre_rot = Rot.from_quat(quats).mean()
        return centre_t, centre_rot

    @staticmethod
    def _burst_deviations(translations, quats, centre_t, centre_rot):
        dev_t = np.linalg.norm(translations - centre_t, axis=1)
        dev_r_deg = np.degrees(
            (centre_rot.inv() * Rot.from_quat(quats)).magnitude()
        )
        return dev_t, np.atleast_1d(dev_r_deg)

    @staticmethod
    def average_transforms(transforms_7list, reject_outliers=True):
        """
        Robustly average a burst of [tx, ty, tz, qx, qy, qz, qw] samples that are
        assumed to all observe the *same* physical pose (e.g. several frames
        captured in quick succession for one calibration sample point).

        Both channels are made robust: a provisional centre (median translation,
        chordal mean rotation) is used to score each frame, frames deviating far
        beyond the MAD in EITHER translation or rotation are dropped, and the
        centre is recomputed on the survivors. Rotation needs this explicitly —
        the chordal mean alone is a least-squares estimator, so a single badly
        misdetected frame drags it noticeably even when the median-based
        translation is unaffected.

        Returns (avg_7list, spread); spread reports how much the *surviving*
        frames disagreed, plus how many were rejected — a useful signal for
        "did the robot or the board move during capture".
        """
        arr = np.atleast_2d(np.asarray(transforms_7list, dtype=np.float64))
        translations = arr[:, :3]
        quats = arr[:, 3:]
        n = len(arr)

        keep = np.ones(n, dtype=bool)
        if reject_outliers and n >= CalibrationBackend.BURST_MIN_KEEP + 1:
            centre_t, centre_rot = CalibrationBackend._burst_centre(translations, quats)
            dev_t, dev_r = CalibrationBackend._burst_deviations(
                translations, quats, centre_t, centre_rot)
            candidate = (
                CalibrationBackend._mad_keep_mask(dev_t, CalibrationBackend.BURST_TRANS_FLOOR_M)
                & CalibrationBackend._mad_keep_mask(dev_r, CalibrationBackend.BURST_ROT_FLOOR_DEG)
            )
            # Never let the gate strip the burst down to noise.
            if candidate.sum() >= CalibrationBackend.BURST_MIN_KEEP:
                keep = candidate

        avg_t, avg_rot = CalibrationBackend._burst_centre(translations[keep], quats[keep])
        avg_quat = avg_rot.as_quat()

        dev_t, dev_r_deg = CalibrationBackend._burst_deviations(
            translations[keep], quats[keep], avg_t, avg_rot)

        spread = {
            'count': int(keep.sum()),
            'input_count': int(n),
            'rejected_frames': int(n - keep.sum()),
            'mean_translation_dev_m': float(np.mean(dev_t)),
            'max_translation_dev_m': float(np.max(dev_t)),
            'mean_rotation_dev_deg': float(np.mean(dev_r_deg)),
            'max_rotation_dev_deg': float(np.max(dev_r_deg)),
        }
        avg = [
            float(avg_t[0]), float(avg_t[1]), float(avg_t[2]),
            float(avg_quat[0]), float(avg_quat[1]), float(avg_quat[2]), float(avg_quat[3]),
        ]
        return avg, spread

    @staticmethod
    def get_opencv_samples(samples_robot, samples_tracking):
        """
        Returns the sample list as a rotation matrix and a translation vector.
        :rtype: (np.array, np.array)
        """
        hand_base_rot = []
        hand_base_tr = []
        marker_camera_rot = []
        marker_camera_tr = []

        for robot_tf, tracking_tf in zip(samples_robot, samples_tracking):
            (mcr, mct) = CalibrationBackend.list_to_opencv(tracking_tf)
            marker_camera_rot.append(mcr)
            marker_camera_tr.append(mct)

            (hbr, hbt) = CalibrationBackend.list_to_opencv(robot_tf)
            hand_base_rot.append(hbr)
            hand_base_tr.append(hbt)

        return (hand_base_rot, hand_base_tr), (marker_camera_rot, marker_camera_tr)

    # ------------------------------------------------------------------
    # AX = XB residuals (used for algorithm selection, outlier rejection,
    # nonlinear refinement, and reporting — always over ALL sample pairs,
    # not just consecutive ones).
    # ------------------------------------------------------------------
    @staticmethod
    def _pair_residual_raw(Rg_i, tg_i, Rg_j, tg_j, Rc_i, tc_i, Rc_j, tc_j, R_X, t_X):
        # A_ij = Gb_j^-1 * Gb_i   (relative robot motion, "gripper2base"-style samples)
        # B_ij = Ct_j * Ct_i^-1   (relative tracking motion, "target2cam"-style samples)
        # Solving A X = X B, so a perfect X drives both residuals to zero.
        R_A = Rg_j.T @ Rg_i
        t_A = Rg_j.T @ (tg_i - tg_j)
        R_B = Rc_j @ Rc_i.T
        t_B = tc_j - R_B @ tc_i

        R_err = R_A @ R_X @ R_B.T @ R_X.T
        t_err_vec = R_A @ t_X + t_A - R_X @ t_B - t_X
        return t_err_vec, R_err

    @staticmethod
    def _pair_residual_rt(Rg_i, tg_i, Rg_j, tg_j, Rc_i, tc_i, Rc_j, tc_j, R_X, t_X):
        t_err_vec, R_err = CalibrationBackend._pair_residual_raw(
            Rg_i, tg_i, Rg_j, tg_j, Rc_i, tc_i, Rc_j, tc_j, R_X, t_X)
        rot_err_deg = math.degrees(Rot.from_matrix(R_err).magnitude())
        return float(np.linalg.norm(t_err_vec)), rot_err_deg

    @staticmethod
    def pairwise_residuals(samples_robot, samples_tracking, cal, indices=None):
        """
        AX=XB residual statistics over ALL sample pairs (not just consecutive
        ones), for a candidate calibration `cal` ([tx,ty,tz,qx,qy,qz,qw]).
        """
        idx = list(range(len(samples_robot))) if indices is None else list(indices)
        if len(idx) < 2:
            return None

        Rg, tg = CalibrationBackend._to_rot_tr_arrays([samples_robot[i] for i in idx])
        Rc, tc = CalibrationBackend._to_rot_tr_arrays([samples_tracking[i] for i in idx])
        R_X, t_X = CalibrationBackend.list_to_opencv(cal)

        pairs = CalibrationBackend._all_pairs(len(idx))
        t_errs, r_errs = [], []
        for (a, b) in pairs:
            t_e, r_e = CalibrationBackend._pair_residual_rt(
                Rg[a], tg[a], Rg[b], tg[b], Rc[a], tc[a], Rc[b], tc[b], R_X, t_X)
            t_errs.append(t_e)
            r_errs.append(r_e)

        return {
            'pair_count': len(pairs),
            'mean_translation_m': float(np.mean(t_errs)),
            'max_translation_m': float(np.max(t_errs)),
            'mean_rotation_deg': float(np.mean(r_errs)),
            'max_rotation_deg': float(np.max(r_errs)),
        }

    @staticmethod
    def _combined_score(t_err_m, r_err_deg):
        return t_err_m / CalibrationBackend.TRANS_SCALE_M + r_err_deg / CalibrationBackend.ROT_SCALE_DEG

    # ------------------------------------------------------------------
    # Closed-form fit + algorithm cross-validation
    # ------------------------------------------------------------------
    @staticmethod
    def _fit(samples_robot, samples_tracking, indices, algorithm):
        if algorithm not in CalibrationBackend.AVAILABLE_ALGORITHMS:
            raise ValueError(
                f"Unknown hand-eye algorithm {algorithm!r}; "
                f"available: {sorted(CalibrationBackend.AVAILABLE_ALGORITHMS)}")
        sr = [samples_robot[i] for i in indices]
        st = [samples_tracking[i] for i in indices]
        (hwr, hwt), (mcr, mct) = CalibrationBackend.get_opencv_samples(sr, st)
        method = CalibrationBackend.AVAILABLE_ALGORITHMS[algorithm]
        # OpenCV raises cv2.error on degenerate pose sets. Callers (and the ROS
        # service callback above them) should only ever have to handle
        # RuntimeError/ValueError, so translate it here rather than leaking a
        # cv2-specific exception type up the stack.
        try:
            rot, tr = cv2.calibrateHandEye(hwr, hwt, mcr, mct, method=method)
        except cv2.error as exc:
            raise RuntimeError(f"OpenCV hand-eye solver '{algorithm}' failed: {exc}") from exc
        return np.asarray(rot, dtype=np.float64), np.asarray(tr, dtype=np.float64).reshape(-1)

    @staticmethod
    def _select_algorithm(samples_robot, samples_tracking, indices):
        """
        Closed-form hand-eye solutions (Tsai, Park, Horaud, Andreff, Daniilidis)
        are all "1989-1998 era" linear solutions that can disagree sharply on
        noisy/degenerate pose sets. Instead of hard-coding one, fit all of them
        and keep whichever has the lowest AX=XB residual on the actual data —
        a cheap cross-validation that also flags badly-conditioned sample sets
        (if every method disagrees a lot, something is wrong with the samples).
        """
        best = None
        for algo in CalibrationBackend.AVAILABLE_ALGORITHMS:
            try:
                rot, tr = CalibrationBackend._fit(samples_robot, samples_tracking, indices, algo)
            except RuntimeError:
                continue  # this solver could not handle these poses; try the next
            if not np.all(np.isfinite(rot)) or not np.all(np.isfinite(tr)):
                continue
            cal = CalibrationBackend._rot_tr_to_list(rot, tr)
            res = CalibrationBackend.pairwise_residuals(samples_robot, samples_tracking, cal, indices=indices)
            if res is None:
                continue
            score = CalibrationBackend._combined_score(res['mean_translation_m'], res['mean_rotation_deg'])
            if best is None or score < best[0]:
                best = (score, algo, rot, tr)
        if best is None:
            raise RuntimeError("No hand-eye algorithm converged on the given samples")
        return best[1], best[2], best[3]

    # ------------------------------------------------------------------
    # Outlier rejection (greedy, MAD-based over all-pairs residual scores)
    # ------------------------------------------------------------------
    @staticmethod
    def _reject_outliers(samples_robot, samples_tracking, algorithm):
        n = len(samples_robot)
        kept = list(range(n))
        removed = []

        max_removals = max(0, min(n - CalibrationBackend.MIN_SAMPLES,
                                   int(n * CalibrationBackend.MAX_REJECT_FRACTION)))
        rot, tr = CalibrationBackend._fit(samples_robot, samples_tracking, kept, algorithm)
        if max_removals <= 0 or n < CalibrationBackend.MIN_SAMPLES + 2:
            return kept, removed, rot, tr

        while len(removed) < max_removals and len(kept) > CalibrationBackend.MIN_SAMPLES:
            Rg, tg = CalibrationBackend._to_rot_tr_arrays([samples_robot[i] for i in kept])
            Rc, tc = CalibrationBackend._to_rot_tr_arrays([samples_tracking[i] for i in kept])
            pairs = CalibrationBackend._all_pairs(len(kept))
            if not pairs:
                break

            per_sample = {}
            for (a, b) in pairs:
                t_e, r_e = CalibrationBackend._pair_residual_rt(
                    Rg[a], tg[a], Rg[b], tg[b], Rc[a], tc[a], Rc[b], tc[b], rot, tr)
                score = CalibrationBackend._combined_score(t_e, r_e)
                per_sample.setdefault(a, []).append(score)
                per_sample.setdefault(b, []).append(score)

            agg = {k: float(np.median(v)) for k, v in per_sample.items()}
            vals = np.array(list(agg.values()))
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) + 1e-9

            worst_local = max(agg, key=agg.get)
            worst_score = agg[worst_local]
            if worst_score < CalibrationBackend.OUTLIER_SCORE_FLOOR or worst_score <= med + CalibrationBackend.OUTLIER_MAD_K * mad:
                break

            removed.append(kept.pop(worst_local))
            rot, tr = CalibrationBackend._fit(samples_robot, samples_tracking, kept, algorithm)

        return kept, removed, rot, tr

    # ------------------------------------------------------------------
    # Nonlinear refinement
    # ------------------------------------------------------------------
    @staticmethod
    def _refine_nonlinear(rot0, tr0, samples_robot, samples_tracking, indices):
        """
        Closed-form hand-eye solutions minimize an algebraic proxy (rotation
        solved first, then translation via a separate linear system) rather
        than the true AX=XB residual jointly. Use the closed-form result as
        the initial guess and refine with Levenberg-Marquardt-style nonlinear
        least squares (via scipy's trust-region solver with a robust loss)
        minimizing that residual jointly and over ALL sample pairs at once.
        """
        Rg, tg = CalibrationBackend._to_rot_tr_arrays([samples_robot[i] for i in indices])
        Rc, tc = CalibrationBackend._to_rot_tr_arrays([samples_tracking[i] for i in indices])
        pairs = CalibrationBackend._all_pairs(len(indices))

        rotvec0 = Rot.from_matrix(rot0).as_rotvec()
        x0 = np.concatenate([rotvec0, tr0])

        def residual(x):
            rv, t = x[:3], x[3:]
            R_X = Rot.from_rotvec(rv).as_matrix()
            out = np.empty(len(pairs) * 6, dtype=np.float64)
            for k, (a, b) in enumerate(pairs):
                t_err_vec, R_err = CalibrationBackend._pair_residual_raw(
                    Rg[a], tg[a], Rg[b], tg[b], Rc[a], tc[a], Rc[b], tc[b], R_X, t)
                rot_err_vec = Rot.from_matrix(R_err).as_rotvec()
                out[k * 6:k * 6 + 3] = t_err_vec / CalibrationBackend.TRANS_SCALE_M
                out[k * 6 + 3:k * 6 + 6] = rot_err_vec / CalibrationBackend.ROT_SCALE_RAD
            return out

        result = least_squares(residual, x0, method='trf', loss='soft_l1', f_scale=1.0, max_nfev=2000)

        rv_ref, t_ref = result.x[:3], result.x[3:]
        R_ref = Rot.from_rotvec(rv_ref).as_matrix()

        delta_t = float(np.linalg.norm(t_ref - tr0))
        delta_r = float(math.degrees(Rot.from_matrix(rot0.T @ R_ref).magnitude()))
        info = {
            'converged': bool(result.success),
            'delta_translation_m': delta_t,
            'delta_rotation_deg': delta_r,
        }
        return R_ref, t_ref, info

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    @staticmethod
    def compute_calibration_detailed(samples_robot, samples_tracking, algorithm=None,
                                      reject_outliers=True, refine=True):
        """
        Full precision-oriented hand-eye pipeline:
          1. cross-validate closed-form algorithms (or use the one requested),
          2. greedily reject outlier samples using the all-pairs AX=XB residual,
          3. refine the survivor set with nonlinear least squares.

        Returns a dict with the final transform plus diagnostics (rejected
        samples, algorithm used, how much refinement moved the estimate).
        """
        n = len(samples_robot)
        if n < CalibrationBackend.MIN_SAMPLES:
            raise ValueError(f"Need at least {CalibrationBackend.MIN_SAMPLES} samples, got {n}")
        all_idx = list(range(n))

        if algorithm is not None:
            algo_used = algorithm
            rot, tr = CalibrationBackend._fit(samples_robot, samples_tracking, all_idx, algorithm)
        else:
            algo_used, rot, tr = CalibrationBackend._select_algorithm(samples_robot, samples_tracking, all_idx)

        if reject_outliers:
            kept, rejected, rot, tr = CalibrationBackend._reject_outliers(samples_robot, samples_tracking, algo_used)
        else:
            kept, rejected = all_idx, []

        closed_form_transform = CalibrationBackend._rot_tr_to_list(rot, tr)

        if refine and len(kept) >= CalibrationBackend.MIN_SAMPLES:
            rot_ref, tr_ref, refine_info = CalibrationBackend._refine_nonlinear(
                rot, tr, samples_robot, samples_tracking, kept)
        else:
            rot_ref, tr_ref = rot, tr
            refine_info = {'converged': None, 'delta_translation_m': 0.0, 'delta_rotation_deg': 0.0}

        transform = CalibrationBackend._rot_tr_to_list(rot_ref, tr_ref)
        residuals = CalibrationBackend.pairwise_residuals(samples_robot, samples_tracking, transform, indices=kept)

        return {
            'transform': transform,
            'closed_form_transform': closed_form_transform,
            'algorithm_used': algo_used,
            'kept_indices': kept,
            'rejected_indices': rejected,
            'refinement': refine_info,
            'residuals': residuals,
        }

    @staticmethod
    def _single_fit(samples_robot, samples_tracking, algorithm):
        """One closed-form fit + nonlinear refinement, no cross-validation or
        outlier rejection. This is the cheap inner loop of the bootstrap."""
        idx = list(range(len(samples_robot)))
        rot, tr = CalibrationBackend._fit(samples_robot, samples_tracking, idx, algorithm)
        rot_ref, tr_ref, _ = CalibrationBackend._refine_nonlinear(
            rot, tr, samples_robot, samples_tracking, idx)
        return CalibrationBackend._rot_tr_to_list(rot_ref, tr_ref)

    @staticmethod
    def bootstrap_uncertainty(samples_robot, samples_tracking, nominal_transform,
                              algorithm, n_bootstrap=40, seed=0):
        """
        Empirical uncertainty of the hand-eye estimate, by resampling.

        Why resampling rather than the usual analytic sigma^2 * (J^T J)^-1: the
        AX=XB residuals are built from all sample PAIRS, so they are not
        independent (n samples produce n(n-1)/2 pairs). The analytic formula
        treats them as if they were and comes out over-confident — measured
        against a known ground truth it reported ~1.0 mm where the true spread
        was ~1.34 mm, giving 51% coverage of a nominal 1-sigma interval instead
        of 68%. Correcting the degrees of freedom by sample count instead
        overshoots the other way (91% coverage). The scaling needed sits
        between the two and depends on sample count and pose geometry, so
        there is no constant to hard-code.

        Resampling the SAMPLES (the independent unit) sidesteps all of that and
        measured 65-75% coverage untuned, which is the honest answer.

        Returns None if too few resamples converged to say anything useful.
        """
        n = len(samples_robot)
        if n < CalibrationBackend.MIN_SAMPLES + 1:
            return None

        rng = np.random.default_rng(seed)
        fits = []
        attempts = 0
        max_attempts = n_bootstrap * 4

        while len(fits) < n_bootstrap and attempts < max_attempts:
            attempts += 1
            pick = rng.integers(0, n, n)
            # A resample of mostly-duplicate poses carries almost no relative
            # motion and cannot constrain the calibration; skip it rather than
            # feeding the solver a degenerate set.
            if len(set(pick.tolist())) < CalibrationBackend.MIN_SAMPLES:
                continue
            sr = [samples_robot[i] for i in pick]
            st = [samples_tracking[i] for i in pick]
            try:
                fits.append(CalibrationBackend._single_fit(sr, st, algorithm))
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue

        if len(fits) < max(8, n_bootstrap // 4):
            return None

        fits_arr = np.asarray(fits, dtype=np.float64)
        translations = fits_arr[:, :3]

        nominal_rot = Rot.from_quat(nominal_transform[3:])
        dev_rotvec = (nominal_rot.inv() * Rot.from_quat(fits_arr[:, 3:])).as_rotvec()
        dev_rotvec = np.atleast_2d(dev_rotvec)

        translation_sigma = translations.std(axis=0, ddof=1)
        rotation_sigma_deg = np.degrees(dev_rotvec.std(axis=0, ddof=1))

        # Direction analysis: which way is the estimate least constrained?
        # For hand-eye this is usually a real, actionable answer (translation
        # along the camera's optical axis is classically weakly observable).
        cov = np.cov(translations, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 0.0, None)
        worst_sigma = float(np.sqrt(eigvals[-1]))
        best_sigma = float(np.sqrt(eigvals[0]))
        worst_axis = eigvecs[:, -1]
        if worst_axis[np.argmax(np.abs(worst_axis))] < 0:
            worst_axis = -worst_axis  # sign is arbitrary; pick a stable one

        return {
            'method': 'bootstrap',
            'n_bootstrap': len(fits),
            'n_requested': int(n_bootstrap),
            'sample_count': int(n),
            'translation_sigma_m': [float(v) for v in translation_sigma],
            'rotation_sigma_deg': [float(v) for v in rotation_sigma_deg],
            'translation_sigma_rms_m': float(np.sqrt(np.mean(translation_sigma ** 2))),
            'rotation_sigma_rms_deg': float(np.sqrt(np.mean(rotation_sigma_deg ** 2))),
            'worst_direction_sigma_m': worst_sigma,
            'worst_direction_axis': [float(v) for v in worst_axis],
            'best_direction_sigma_m': best_sigma,
            'anisotropy': float(worst_sigma / best_sigma) if best_sigma > 1e-12 else float('inf'),
            'guidance': CalibrationBackend._uncertainty_guidance(worst_sigma, best_sigma, worst_axis),
        }

    @staticmethod
    def _uncertainty_guidance(worst_sigma, best_sigma, worst_axis):
        """Turn the covariance shape into an instruction the operator can act on."""
        axis_names = ('X', 'Y', 'Z')
        dominant = axis_names[int(np.argmax(np.abs(worst_axis)))]
        anisotropy = worst_sigma / best_sigma if best_sigma > 1e-12 else float('inf')

        if worst_sigma < 0.0005:
            return (f"Calibration is tight (worst-direction sigma {worst_sigma * 1000:.2f} mm). "
                    "Collecting more samples will not help much.")
        if anisotropy < 2.0:
            return (f"Uncertainty is roughly isotropic at {worst_sigma * 1000:.2f} mm. "
                    "More samples will shrink it; no particular direction is starved.")
        return (
            f"Uncertainty is {anisotropy:.1f}x worse along one direction "
            f"(mostly {dominant}, sigma {worst_sigma * 1000:.2f} mm vs {best_sigma * 1000:.2f} mm). "
            f"Add poses that move and rotate the arm along {dominant} — repeating similar "
            "poses will not fix this direction."
        )

    @staticmethod
    def compute_calibration(samples_robot, samples_tracking, algorithm=None):
        """
        Computes the calibration and returns just the transform, for callers
        that don't need the full diagnostics. See compute_calibration_detailed.
        :rtype: [tx, ty, tz, qx, qy, qz, qw]
        """
        return CalibrationBackend.compute_calibration_detailed(
            samples_robot, samples_tracking, algorithm=algorithm)['transform']
