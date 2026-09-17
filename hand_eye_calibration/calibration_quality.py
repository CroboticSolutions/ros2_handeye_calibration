"""Automatic save gate; uncertainty is an estimate, not absolute tool accuracy."""
import math


def position_uncertainty_ok(uncertainty, limit):
    try:
        sigma = float(uncertainty['worst_direction_sigma_m'])
        count = int(uncertainty['n_bootstrap'])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(sigma) and 0 <= sigma <= limit and count >= 8
