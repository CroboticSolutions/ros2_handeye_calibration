"""Acceptance gate for a hand-eye result.

Internal consistency (bootstrap sigma, residuals) says how repeatable the
estimate is; the held-out checks say how well it predicts views it never saw.
A calibration is written to the active file only when both pass. Defaults are
sized for MIG/TIG seams (~1-2 mm): a camera mount error of a few mm is already
larger than the seam tolerance.
"""
import math

DEFAULT_LIMITS = {
    'min_samples': 12,
    'max_position_sigma_m': 0.002,
    'max_loo_position_rms_m': 0.003,
    'max_loo_reprojection_rms_px': 3.0,
    'max_board_spread_rms_m': 0.003,
    'max_validation_position_m': 0.004,
}


def _check(name, value, limit, detail=''):
    ok = value is not None and math.isfinite(value) and value <= limit
    return {'name': name, 'value': value, 'limit': limit, 'passed': bool(ok), 'detail': detail}


def evaluate(report, sample_count, limits=None):
    """report: output of reprojection_calibration.calibrate (or None when only
    the closed-form estimate exists). Returns {'passed', 'checks', 'summary'}."""
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    checks = [{'name': 'sample_count', 'value': sample_count, 'limit': limits['min_samples'],
               'passed': sample_count >= limits['min_samples'], 'detail': 'minimum poses'}]
    if report is None:
        checks.append({'name': 'reprojection_solver', 'value': None, 'limit': None, 'passed': False,
                       'detail': 'no raw corner observations; validation impossible'})
    else:
        uncertainty = report.get('uncertainty') or {}
        checks.append(_check('position_sigma_m', uncertainty.get('worst_direction_sigma_m'),
                             limits['max_position_sigma_m'], 'bootstrap, worst direction'))
        spread = report.get('board_spread') or {}
        checks.append(_check('board_spread_rms_m', spread.get('position_rms_m'),
                             limits['max_board_spread_rms_m'], 'static board seen from every pose'))
        loo = report.get('leave_one_out')
        if loo is None:
            checks.append({'name': 'leave_one_out', 'value': None, 'limit': None, 'passed': False,
                           'detail': 'not enough views'})
        else:
            checks.append(_check('loo_position_rms_m', loo.get('position_rms_m'),
                                 limits['max_loo_position_rms_m'], 'held-out board position'))
            checks.append(_check('loo_reprojection_rms_px', loo.get('reprojection_rms_px'),
                                 limits['max_loo_reprojection_rms_px'], 'held-out corner pixels'))
        comparison = report.get('refinement_comparison')
        if comparison is not None:
            checks.append({'name': 'refinement_non_regression', 'value': bool(comparison.get('passed')),
                           'limit': True, 'passed': bool(comparison.get('passed')),
                           'detail': comparison.get('reason', 'Missing held-out comparison')})
        validation = report.get('validation_views')
        if validation:
            checks.append(_check('validation_position_max_m', validation.get('position_max_m'),
                                 limits['max_validation_position_m'], 'dedicated validation poses'))
    failed = [c for c in checks if not c['passed']]
    summary = 'All acceptance checks passed.' if not failed else 'Failed: ' + ', '.join(
        f"{c['name']}={_fmt(c['value'])} (limit {_fmt(c['limit'])})" for c in failed)
    return {'passed': not failed, 'checks': checks, 'summary': summary, 'limits': limits}


def _fmt(value):
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.4g}'
    return str(value)
