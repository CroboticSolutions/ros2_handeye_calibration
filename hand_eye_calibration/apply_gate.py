"""Acceptance gate shared by every Apply/bake CLI (store versions and xacro mounts).

A calibration YAML is baked only when its own acceptance verdict passed. A YAML
without a verdict (hand-edited, offline solve output) is refused like a failed one;
``force`` (CLI --force, GUI "Force save & apply") bakes it anyway. The one explicit
exception is ``calibration_mode: manual``: values the operator typed in (UR pendant
TCP), where nothing was measured that a verdict could judge.

Only the calibration's own quality is checked here, never how far the new value is
from the active one: a first calibration of an unknown tool may change anything.
"""


class AcceptanceRefused(SystemExit):
    """Raised (exit status 1 with the message) when a YAML may not be applied."""


def acceptance_problem(data, force=False):
    """None when ``data`` may be applied, else why not."""
    if force:
        return None
    acceptance = (data or {}).get('acceptance')
    if acceptance is None:
        if (data or {}).get('calibration_mode') == 'manual':
            return None
        return 'no acceptance verdict in the YAML'
    if not acceptance.get('passed'):
        return f'acceptance failed ({acceptance.get("summary")})'
    return None


def require_acceptance(data, source, force=False):
    """Raise AcceptanceRefused unless ``data`` passed acceptance (or ``force``)."""
    problem = acceptance_problem(data, force)
    if problem is not None:
        raise AcceptanceRefused(f'{source}: {problem}; refusing to apply. '
                                'Use --force (GUI: Force save & apply) only if you know why.')
