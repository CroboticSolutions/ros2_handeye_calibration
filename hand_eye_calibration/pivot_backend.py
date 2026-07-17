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
from scipy.spatial.transform import Rotation as Rot


class PivotCalibrationBackend:
    MIN_SAMPLES = 4

    @staticmethod
    def _design_row(flange_tf):
        """flange_tf = [tx, ty, tz, qx, qy, qz, qw] -> (3x6 block, 3-vector rhs)."""
        tr = np.array(flange_tf[:3], dtype=float)
        rot = Rot.from_quat(flange_tf[3:]).as_matrix()
        block = np.hstack([rot, -np.eye(3)])
        rhs = -tr
        return block, rhs

    @staticmethod
    def compute_pivot(flange_samples):
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

        rows = []
        rhs = []
        for sample in flange_samples:
            block, r = PivotCalibrationBackend._design_row(sample)
            rows.append(block)
            rhs.append(r)
        a = np.vstack(rows)
        b = np.concatenate(rhs)

        solution, _, rank, singular_values = np.linalg.lstsq(a, b, rcond=None)
        t = solution[:3]
        p = solution[3:]

        condition_number = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > 1e-12
            else float("inf")
        )

        predicted = a @ solution
        per_sample = (predicted - b).reshape(n, 3)
        per_sample_norms = np.linalg.norm(per_sample, axis=1)

        return {
            "tcp_translation": [float(v) for v in t],
            "fixed_point": [float(v) for v in p],
            "condition_number": condition_number,
            "rank": int(rank),
            "per_sample_residuals_m": [float(v) for v in per_sample_norms],
            "rms_residual_m": float(np.sqrt(np.mean(per_sample_norms**2))),
            "max_residual_m": float(np.max(per_sample_norms)),
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
