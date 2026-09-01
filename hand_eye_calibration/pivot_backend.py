"""
Pivot (tool-tip) calibration backend.

The operator keeps the tool tip touching a single fixed point in the world
while capturing the flange pose from several different wrist orientations.
For each sample i, with flange rotation R_i and translation q_i (both in the
robot base frame):

    p_tip = R_i @ t + q_i

where t is the unknown flange -> tool_tip vector (in the flange frame) and
p_tip is the unknown, but fixed, pivot point (in the base frame). Rearranged:

    [R_i  -I] [t; p_tip] = -q_i

Stacking all samples gives a linear least-squares system solved for
[t; p_tip] jointly. This determines the tool TCP *translation* only --
touching one fixed point cannot recover the TCP orientation.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


class PivotCalibrationBackend:
    # Four poses are enough for an algebraic preview, but are not enough for a
    # calibration that should be trusted on a welding robot.  The status/save
    # layer applies the stricter 20-pose quality gate.
    MIN_SAMPLES = 4
    RANSAC_THRESHOLD_M = 0.002
    RANSAC_ITERATIONS = 500
    ROBUST_F_SCALE_M = 0.00075
    BOOTSTRAP_SAMPLES = 400

    @staticmethod
    def _build_system(flange_samples):
        rows = []
        rhs = []
        for sample in flange_samples:
            block, r = PivotCalibrationBackend._design_row(sample)
            rows.append(block)
            rhs.append(r)
        return np.vstack(rows), np.concatenate(rhs)

    @staticmethod
    def _linear_fit(flange_samples):
        a, b = PivotCalibrationBackend._build_system(flange_samples)
        solution, _, rank, singular_values = np.linalg.lstsq(a, b, rcond=None)
        condition_number = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > 1e-12
            else float("inf")
        )
        return solution, int(rank), condition_number

    @staticmethod
    def _residual_vectors(flange_samples, solution):
        a, b = PivotCalibrationBackend._build_system(flange_samples)
        return (a @ solution - b).reshape(len(flange_samples), 3)

    @staticmethod
    def _ransac_inliers(flange_samples, *, threshold_m, iterations, random_seed):
        """Return deterministic RANSAC inliers for the algebraic pivot model."""
        n = len(flange_samples)
        subset_size = PivotCalibrationBackend.MIN_SAMPLES
        rng = np.random.default_rng(random_seed)
        best = None

        # Always score the all-sample AOS fit as a deterministic fallback.
        candidates = [np.arange(n, dtype=int)]
        candidates.extend(
            rng.choice(n, size=subset_size, replace=False) for _ in range(iterations)
        )
        for subset in candidates:
            try:
                solution, rank, _ = PivotCalibrationBackend._linear_fit(
                    [flange_samples[int(i)] for i in subset]
                )
            except np.linalg.LinAlgError:
                continue
            if rank < 6:
                continue
            norms = np.linalg.norm(
                PivotCalibrationBackend._residual_vectors(flange_samples, solution),
                axis=1,
            )
            inliers = np.flatnonzero(norms <= threshold_m)
            if len(inliers) < subset_size:
                continue
            score = (
                len(inliers),
                -float(np.median(norms[inliers])),
                -float(np.sqrt(np.mean(norms[inliers] ** 2))),
            )
            if best is None or score > best[0]:
                best = (score, inliers)

        if best is None:
            return np.arange(n, dtype=int)
        return best[1]

    @staticmethod
    def _robust_fit(flange_samples, initial_solution, *, f_scale_m):
        a, b = PivotCalibrationBackend._build_system(flange_samples)
        result = least_squares(
            lambda x: a @ x - b,
            initial_solution,
            loss="huber",
            f_scale=f_scale_m,
            method="trf",
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return initial_solution
        return result.x

    @staticmethod
    def _uncertainty(flange_samples, *, bootstrap_samples, random_seed):
        """Bootstrap confidence interval and leave-one-out TCP influence."""
        n = len(flange_samples)
        rng = np.random.default_rng(random_seed)
        boot_tcp = []
        for _ in range(bootstrap_samples):
            pick = rng.integers(0, n, size=n)
            if len(set(pick.tolist())) < PivotCalibrationBackend.MIN_SAMPLES:
                continue
            try:
                solution, rank, _ = PivotCalibrationBackend._linear_fit(
                    [flange_samples[int(i)] for i in pick]
                )
            except np.linalg.LinAlgError:
                continue
            if rank == 6 and np.all(np.isfinite(solution)):
                boot_tcp.append(solution[:3])

        if boot_tcp:
            arr = np.asarray(boot_tcp)
            ci_low = np.percentile(arr, 2.5, axis=0)
            ci_high = np.percentile(arr, 97.5, axis=0)
            std = np.std(arr, axis=0, ddof=1) if len(arr) > 1 else np.zeros(3)
            ci_half = 0.5 * (ci_high - ci_low)
        else:
            ci_low = ci_high = std = ci_half = np.full(3, np.nan)

        full, _, _ = PivotCalibrationBackend._linear_fit(flange_samples)
        loo_shifts = []
        for omitted in range(n):
            reduced = flange_samples[:omitted] + flange_samples[omitted + 1 :]
            try:
                loo, rank, _ = PivotCalibrationBackend._linear_fit(reduced)
                shift = float(np.linalg.norm(loo[:3] - full[:3])) if rank == 6 else float("inf")
            except np.linalg.LinAlgError:
                shift = float("inf")
            loo_shifts.append(shift)

        return {
            "bootstrap_count": len(boot_tcp),
            "tcp_std_m": [float(v) for v in std],
            "tcp_ci95_low_m": [float(v) for v in ci_low],
            "tcp_ci95_high_m": [float(v) for v in ci_high],
            "tcp_ci95_half_width_m": [float(v) for v in ci_half],
            "max_tcp_ci95_half_width_m": float(np.max(ci_half)),
            "loo_tcp_shifts_m": loo_shifts,
            "max_loo_tcp_shift_m": float(np.max(loo_shifts)),
        }

    @staticmethod
    def _design_row(flange_tf):
        """flange_tf = [tx, ty, tz, qx, qy, qz, qw] -> (3x6 block, 3-vector rhs)."""
        tr = np.array(flange_tf[:3], dtype=float)
        rot = Rot.from_quat(flange_tf[3:]).as_matrix()
        block = np.hstack([rot, -np.eye(3)])
        rhs = -tr
        return block, rhs

    @staticmethod
    def compute_pivot(
        flange_samples,
        *,
        robust=True,
        ransac_threshold_m=RANSAC_THRESHOLD_M,
        ransac_iterations=RANSAC_ITERATIONS,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        random_seed=17,
    ):
        """
        flange_samples: list of [tx, ty, tz, qx, qy, qz, qw] flange poses in the
        robot base frame, captured while the tool tip touches one fixed point.

        Returns a dict with:
          tcp_translation: [x, y, z] flange -> tool_tip, in the flange frame
          fixed_point: [x, y, z] pivot point location, in the base frame
          condition_number: conditioning of the least-squares design matrix
                             (large => orientations were too similar)
          per_sample_residuals_m: per-sample translation residual norm
          rms_residual_m, max_residual_m: aggregate residual stats
        """
        n = len(flange_samples)
        if n < PivotCalibrationBackend.MIN_SAMPLES:
            raise ValueError(
                f"Need at least {PivotCalibrationBackend.MIN_SAMPLES} samples, got {n}"
            )

        if robust:
            inlier_indices = PivotCalibrationBackend._ransac_inliers(
                flange_samples,
                threshold_m=ransac_threshold_m,
                iterations=ransac_iterations,
                random_seed=random_seed,
            )
        else:
            inlier_indices = np.arange(n, dtype=int)
        inlier_samples = [flange_samples[int(i)] for i in inlier_indices]
        initial, rank, condition_number = PivotCalibrationBackend._linear_fit(inlier_samples)
        if rank < 6:
            raise ValueError(
                "Pivot system is rank-deficient; capture more varied wrist orientations."
            )
        solution = (
            PivotCalibrationBackend._robust_fit(
                inlier_samples, initial, f_scale_m=PivotCalibrationBackend.ROBUST_F_SCALE_M
            )
            if robust
            else initial
        )
        t = solution[:3]
        p = solution[3:]
        per_sample = PivotCalibrationBackend._residual_vectors(flange_samples, solution)
        per_sample_norms = np.linalg.norm(per_sample, axis=1)
        inlier_mask = np.zeros(n, dtype=bool)
        inlier_mask[inlier_indices] = True
        outlier_indices = np.flatnonzero(~inlier_mask)
        inlier_norms = per_sample_norms[inlier_mask]

        uncertainty = PivotCalibrationBackend._uncertainty(
            inlier_samples,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + 1,
        )
        influential_inlier_indices = [
            int(inlier_indices[i])
            for i, shift in enumerate(uncertainty["loo_tcp_shifts_m"])
            if shift > 0.001
        ]

        result = {
            "tcp_translation": [float(v) for v in t],
            "fixed_point": [float(v) for v in p],
            "condition_number": condition_number,
            "rank": int(rank),
            "method": "ransac_huber_aos" if robust else "aos",
            "ransac_threshold_m": float(ransac_threshold_m),
            "inlier_indices": [int(v) for v in inlier_indices],
            "outlier_indices": [int(v) for v in outlier_indices],
            "inlier_mask": [bool(v) for v in inlier_mask],
            "inlier_count": int(np.sum(inlier_mask)),
            "outlier_count": int(np.sum(~inlier_mask)),
            "per_sample_residuals_m": [float(v) for v in per_sample_norms],
            "rms_residual_m": float(np.sqrt(np.mean(inlier_norms**2))),
            "max_residual_m": float(np.max(per_sample_norms)),
            "inlier_max_residual_m": float(np.max(inlier_norms)),
            "influential_sample_indices": influential_inlier_indices,
        }
        result.update(uncertainty)
        return result

    @staticmethod
    def validate_pivot(flange_samples, pivot_result):
        """Evaluate held-out spike touches without refitting the TCP."""
        if not flange_samples:
            return None
        solution = np.concatenate(
            [pivot_result["tcp_translation"], pivot_result["fixed_point"]]
        )
        residuals = np.linalg.norm(
            PivotCalibrationBackend._residual_vectors(flange_samples, solution), axis=1
        )
        return {
            "sample_count": len(flange_samples),
            "per_sample_residuals_m": [float(v) for v in residuals],
            "rms_residual_m": float(np.sqrt(np.mean(residuals**2))),
            "max_residual_m": float(np.max(residuals)),
            "p95_residual_m": float(np.percentile(residuals, 95.0)),
        }

    @staticmethod
    def aggregate_pose_burst(flange_samples):
        """Robustly average a short burst and report whether the robot moved."""
        if not flange_samples:
            raise ValueError("Cannot aggregate an empty capture burst.")
        translations = np.asarray([sample[:3] for sample in flange_samples], dtype=float)
        rotations = Rot.from_quat([sample[3:] for sample in flange_samples])
        center_t = np.median(translations, axis=0)
        center_r = rotations.mean()
        translation_deviation = np.linalg.norm(translations - center_t, axis=1)
        rotation_deviation_deg = np.degrees((center_r.inv() * rotations).magnitude())
        quat = center_r.as_quat()
        return {
            "pose": [float(v) for v in np.concatenate([center_t, quat])],
            "sample_count": len(flange_samples),
            "translation_p95_m": float(np.percentile(translation_deviation, 95.0)),
            "translation_max_m": float(np.max(translation_deviation)),
            "rotation_p95_deg": float(np.percentile(rotation_deviation_deg, 95.0)),
            "rotation_max_deg": float(np.max(rotation_deviation_deg)),
        }

    # Minimum number of alignment poses for the axis; one is mathematically
    # enough, more are averaged to beat down the operator's alignment noise.
    MIN_ALIGN_SAMPLES = 1

    @staticmethod
    def frame_from_axis(axis_dir):
        """
        Build a full flange -> tool_tcp orientation from just the tool axis
        direction (a unit vector in the flange frame).

        Convention: the tool axis is the +Z of the TCP frame, pointing outward
        through the tip (the business end). Roll about that axis is not
        observable, so it is pinned by convention: TCP +X points to the flange
        -X projected into the plane perpendicular to Z (so TCP +X/+Y sit on the
        opposite side of the flange's own red/green axes, which is how the
        operator expects the tool frame to read next to the flange TF), and
        Y = Z x X. If the tool axis is nearly parallel to flange X, the hint
        falls back to flange Y.

        Returns axis_dir / quaternion [qx,qy,qz,qw] / rotation_matrix.
        """
        z = np.asarray(axis_dir, dtype=float)
        norm = float(np.linalg.norm(z))
        if norm < 1e-9:
            raise ValueError("axis_dir is a zero vector; cannot build a frame.")
        z = z / norm

        # Pin roll: base the in-plane X on flange X projected ⊥ Z. When the tool
        # axis is nearly parallel to flange X that projection vanishes, so fall
        # back to flange Y as the Gram-Schmidt hint.
        hint = np.array([1.0, 0.0, 0.0])
        used_y_fallback = abs(float(np.dot(hint, z))) > 0.9
        if used_y_fallback:
            hint = np.array([0.0, 1.0, 0.0])
        x = hint - np.dot(hint, z) * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        if used_y_fallback:
            x, y = y, -x
        # Flip X and Y 180° about the tool axis so TCP +X/+Y point to the
        # opposite side of the flange's red/green axes (operator convention).
        # Negating two columns keeps a right-handed frame and the same tool +Z.
        x, y = -x, -y
        rot = np.column_stack([x, y, z])
        quat = Rot.from_matrix(rot).as_quat()  # scalar-last [x, y, z, w]

        return {
            "axis_dir": [float(v) for v in z],
            "quaternion": [float(v) for v in quat],
            "rotation_matrix": [[float(v) for v in row] for row in rot],
        }

    @staticmethod
    def compute_axis_from_alignment(
        alignment_samples,
        spike_axis_base=(0.0, 0.0, 1.0),
        tip_translation=None,
    ):
        """
        Recover the tool axis by physically aligning the tool's straight tip
        segment with a reference direction that is KNOWN in the base frame
        (typically the calibration spike, which stands vertical -> [0,0,1] when
        the base is level).

        A single pivot round recovers only the tip *position*; it cannot observe
        how the tool "points". Here the operator instead makes the tool's
        straight segment collinear with the spike and captures the flange pose.
        Because the tool axis then coincides with the known spike direction
        ``n`` in the base frame, its expression in the flange frame is simply

            a_flange = R_iᵀ @ n

        for flange rotation R_i (base -> flange). Several alignment poses are
        averaged; their spread is the axis uncertainty. Roll about the axis
        stays unobservable and is pinned by frame_from_axis.

        alignment_samples: list of [tx,ty,tz,qx,qy,qz,qw] flange poses (base
        frame), each captured with the tool aligned to the spike.
        spike_axis_base: the known reference direction in the base frame.
        tip_translation: flange -> tip vector (from the tip pivot round). Used
        only to fix the axis SIGN so +Z points outward through the tip; without
        it the sign is taken from the first pose and may be flipped.

        Returns frame_from_axis(...) plus:
          alignment_spread_deg: max angle of any pose's axis from the mean
          sample_count, spike_axis_base
        """
        n = len(alignment_samples)
        if n < PivotCalibrationBackend.MIN_ALIGN_SAMPLES:
            raise ValueError(
                f"Need at least {PivotCalibrationBackend.MIN_ALIGN_SAMPLES} "
                f"alignment pose(s), got {n}."
            )

        spike = np.asarray(spike_axis_base, dtype=float)
        spike_norm = float(np.linalg.norm(spike))
        if spike_norm < 1e-9:
            raise ValueError("spike_axis_base is a zero vector.")
        spike = spike / spike_norm

        dirs = []
        for sample in alignment_samples:
            rot = Rot.from_quat(sample[3:]).as_matrix()
            a = rot.T @ spike
            na = float(np.linalg.norm(a))
            if na > 1e-9:
                dirs.append(a / na)
        if not dirs:
            raise ValueError("Alignment poses produced no valid axis directions.")

        # Sign reference: the tool axis (+Z) should point outward through the
        # tip. When we know the tip offset, use it; otherwise anchor to the
        # first pose so the average does not cancel.
        if tip_translation is not None:
            t = np.asarray(tip_translation, dtype=float)
            nt = float(np.linalg.norm(t))
            ref = t / nt if nt > 1e-9 else dirs[0]
        else:
            ref = dirs[0]
        signed = [d if float(np.dot(d, ref)) >= 0.0 else -d for d in dirs]

        mean = np.mean(signed, axis=0)
        mean_norm = float(np.linalg.norm(mean))
        if mean_norm < 1e-9:
            raise ValueError(
                "Alignment directions cancel out — poses are inconsistent."
            )
        axis = mean / mean_norm

        spread_deg = max(
            float(np.degrees(np.arccos(np.clip(float(np.dot(d, axis)), -1.0, 1.0))))
            for d in signed
        )

        frame = PivotCalibrationBackend.frame_from_axis(axis)
        frame["alignment_spread_deg"] = spread_deg
        frame["sample_count"] = n
        frame["spike_axis_base"] = [float(v) for v in spike]
        return frame
