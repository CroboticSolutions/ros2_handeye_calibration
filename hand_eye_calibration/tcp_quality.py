"""Tool-TCP quality: orientation diversity, tool frame, acceptance gate.

Pivot calibration solves [R_i  -I][t; p] = -q_i. If all wrist rotations share
one axis, the tip offset along that axis is unobservable: that is why the
first welding-gun TCP (4 touches, rotations mostly about one axis) moved 29 mm
when one touch was left out. Diversity is therefore measured in the rotations'
own principal directions (basis independent), not per flange X/Y/Z.

Industrial references: ABB 4/5/6-point, KUKA XYZ 4-point + ABC, FANUC 6-point
(full frame for bent torches), Yaskawa/UR reorientation check; accepted mean
error below ~0.5-1 mm, after which the tool is rotated about the TCP and the tip
must stay on the spike.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_LIMITS = {
    'min_samples': 8,
    # Second principal rotation span: rotations about at least two axes.
    'min_second_axis_span_deg': 20.0,
    'max_rms_m': 0.001,
    'max_loo_shift_m': 0.0015,
    # Reorientation / held-out touches, scored without refitting.
    'min_validation_samples': 3,
    'max_validation_m': 0.001,
    # Axis (6-point style): several alignments, consistent, close to CAD.
    'min_align_samples': 3,
    'max_align_spread_deg': 1.0,
    'max_cad_axis_deviation_deg': 5.0,
}


def rotation_spans_deg(flange_samples):
    """Spans (max-min, degrees) of the wrist rotations along their principal
    directions, largest first. [R_i relative to the mean orientation]"""
    if len(flange_samples) < 2:
        return [0.0, 0.0, 0.0]
    rotations = Rotation.from_quat([s[3:7] for s in flange_samples])
    vectors = (rotations.mean().inv() * rotations).as_rotvec()
    centred = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=True)
    projected = centred @ vt.T
    spans = np.degrees(np.ptp(projected, axis=0))
    spans = np.sort(np.r_[spans, np.zeros(3)][:3])[::-1]
    return [float(v) for v in spans]


def bend_plane_frame(axis_dir, flange_axis=(0.0, 0.0, 1.0)):
    """flange -> tool rotation with an explicit roll.

    +Z is the wire axis (out of the tip). +X lies in the torch-neck bend plane
    (the plane spanned by the wire axis and the flange axis), pointing back
    along the flange axis; +Y = Z x X. Only a straight tool (axis parallel to
    the flange axis) has no bend plane; then None is returned and the caller
    keeps its previous convention."""
    z = np.asarray(axis_dir, dtype=float)
    z /= np.linalg.norm(z)
    f = np.asarray(flange_axis, dtype=float)
    x = f - (f @ z) * z
    if np.linalg.norm(x) < math.sin(math.radians(2.0)):
        return None
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    matrix = np.column_stack([x, y, z])
    return {'rotation_matrix': matrix.tolist(), 'quaternion': [float(v) for v in Rotation.from_matrix(matrix).as_quat()],
            'axis_dir': [float(v) for v in z], 'roll_convention': 'x_in_neck_bend_plane'}


def cad_axis_deviation_deg(axis_dir, cad_angle_deg, flange_axis=(0.0, 0.0, 1.0)):
    """|angle(axis, flange Z) - CAD neck angle|. Catches a wrong or flipped axis."""
    if cad_angle_deg is None or cad_angle_deg < 0:
        return None
    z = np.asarray(axis_dir, dtype=float)
    angle = math.degrees(math.acos(np.clip(z @ np.asarray(flange_axis) / np.linalg.norm(z), -1, 1)))
    return abs(angle - cad_angle_deg)


def _check(name, value, limit, ok, detail):
    return {'name': name, 'value': value, 'limit': limit, 'passed': bool(ok), 'detail': detail}


def evaluate(pivot, spans, validation, axis=None, axis_mode=False, cad_deviation=None, limits=None):
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    checks = []
    n = 0 if pivot is None else len(pivot.get('per_sample_residuals_m') or [])
    checks.append(_check('sample_count', n, limits['min_samples'], n >= limits['min_samples'], 'tip touches'))
    second = spans[1] if spans else 0.0
    checks.append(_check('second_axis_span_deg', second, limits['min_second_axis_span_deg'],
                         second >= limits['min_second_axis_span_deg'],
                         'rotation about a second, non-parallel axis'))
    if pivot is not None:
        rms = pivot.get('rms_residual_m')
        checks.append(_check('rms_m', rms, limits['max_rms_m'], rms is not None and rms <= limits['max_rms_m'], 'pivot fit'))
        loo = pivot.get('max_loo_tcp_shift_m')
        checks.append(_check('loo_shift_m', loo, limits['max_loo_shift_m'],
                             loo is not None and math.isfinite(loo) and loo <= limits['max_loo_shift_m'],
                             'largest TCP change when one touch is left out'))
    count = 0 if not validation else validation.get('sample_count', 0)
    checks.append(_check('validation_samples', count, limits['min_validation_samples'],
                         count >= limits['min_validation_samples'], 'reorientation touches (not in the fit)'))
    if validation:
        worst = validation.get('max_residual_m')
        checks.append(_check('validation_max_m', worst, limits['max_validation_m'],
                             worst is not None and worst <= limits['max_validation_m'], 'tip stays on the spike'))
    if axis_mode:
        k = 0 if axis is None else int(axis.get('sample_count') or 0)
        checks.append(_check('align_samples', k, limits['min_align_samples'], k >= limits['min_align_samples'],
                             'alignments to the spike direction'))
        if axis is not None:
            spread = axis.get('alignment_spread_deg')
            checks.append(_check('align_spread_deg', spread, limits['max_align_spread_deg'],
                                 spread is not None and spread <= limits['max_align_spread_deg'],
                                 'agreement between alignments'))
        if cad_deviation is not None:
            checks.append(_check('cad_axis_deviation_deg', cad_deviation, limits['max_cad_axis_deviation_deg'],
                                 cad_deviation <= limits['max_cad_axis_deviation_deg'], 'neck angle vs CAD'))
    failed = [c for c in checks if not c['passed']]
    summary = 'All TCP acceptance checks passed.' if not failed else 'Failed: ' + ', '.join(
        f"{c['name']}={_fmt(c['value'])} (limit {_fmt(c['limit'])})" for c in failed)
    return {'passed': not failed, 'checks': checks, 'summary': summary, 'limits': limits}


def _fmt(v):
    if v is None:
        return 'n/a'
    return f'{v:.4g}' if isinstance(v, float) else str(v)


def reorientation_targets(reference_quat, angle_deg=25.0, spin_deg=45.0, spike_axis=(0.0, 0.0, 1.0)):
    """Tool orientations for the reorientation check: the reference wrist
    orientation tilted +-angle about two base axes perpendicular to the spike and
    spun +-spin about the spike axis. Rotations are about the tip point."""
    n = np.asarray(spike_axis, dtype=float)
    n /= np.linalg.norm(n)
    a = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    ref = Rotation.from_quat(reference_quat)
    out = []
    for axis, angle in ((a, angle_deg), (a, -angle_deg), (b, angle_deg), (b, -angle_deg),
                        (n, spin_deg), (n, -spin_deg)):
        out.append((Rotation.from_rotvec(axis * math.radians(angle)) * ref).as_quat())
    return [np.asarray(q, dtype=float) for q in out]


def flange_pose_for_tip(tip_base, quat_flange, tcp_translation):
    """Flange pose [x y z qx qy qz qw] that puts the tip at tip_base."""
    rotation = Rotation.from_quat(quat_flange)
    position = np.asarray(tip_base, dtype=float) - rotation.apply(np.asarray(tcp_translation, dtype=float))
    return np.r_[position, rotation.as_quat()]


def tip_preserving_waypoints(start_quat, end_quat, tip_base, tcp_translation, steps=8):
    """Flange poses that rotate the tool about its (estimated) tip."""
    key = Rotation.from_quat([start_quat, end_quat])
    from scipy.spatial.transform import Slerp
    slerp = Slerp([0.0, 1.0], key)
    return [flange_pose_for_tip(tip_base, slerp([s])[0].as_quat(), tcp_translation)
            for s in np.linspace(0.0, 1.0, steps + 1)[1:]]
