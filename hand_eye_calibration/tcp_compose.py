"""Position and orientation of a TCP Apply chosen independently.

A TCP store version is a position (flange -> tip) and an orientation (wire axis
plus roll). They need not come from the same place:

position     calibration  measured tip from the saved YAML (pivot / touch-off)
             active       keep the active version's position
             manual       typed in (mm, in the store's parent link frame)
orientation  calibration  measured / neck-angle / touch-off axis from the saved YAML
             active       keep the active version's orientation
             default      the tool model's default orientation (with its axis convention)

Which parts a YAML actually measured is its ``tcp_components`` list; older files
are inferred from ``calibration_mode`` (``position``: tip only, ``orientation``:
axis only, everything else: both). Unset sources default to "what the YAML has,
the active value for the rest", so a position-only pivot no longer replaces the
orientation with the flange axis.
"""

POSITION_SOURCES = ('calibration', 'active', 'manual')
ORIENTATION_SOURCES = ('calibration', 'active', 'default')
COMPONENTS = ('position', 'orientation')


def yaml_components(data):
    """Set of TCP parts the calibration YAML measured."""
    components = (data or {}).get('tcp_components')
    if components:
        unknown = set(components) - set(COMPONENTS)
        if unknown:
            raise SystemExit(f'Unknown tcp_components {sorted(unknown)}; expected {list(COMPONENTS)}.')
        return set(components)
    mode = (data or {}).get('calibration_mode')
    if mode == 'position':
        return {'position'}
    if mode == 'orientation':
        return {'orientation'}
    return set(COMPONENTS)


def resolve_sources(position=None, orientation=None, position_mm=None, components=None):
    """(position, orientation) sources after defaults and checks.

    components: yaml_components() of the calibration YAML, or None when there is
    no usable YAML (then "calibration" is never a default)."""
    if position is not None and position not in POSITION_SOURCES:
        raise SystemExit(f'Unknown position source {position!r}; expected one of {POSITION_SOURCES}.')
    if orientation is not None and orientation not in ORIENTATION_SOURCES:
        raise SystemExit(f'Unknown orientation source {orientation!r}; expected one of {ORIENTATION_SOURCES}.')
    have = components or set()
    if position is None:
        position = 'manual' if position_mm is not None else (
            'calibration' if 'position' in have else 'active')
    if orientation is None:
        orientation = 'calibration' if 'orientation' in have else 'active'
    if position == 'manual' and position_mm is None:
        raise SystemExit('Position source "manual" needs --position-mm X Y Z.')
    if position != 'manual' and position_mm is not None:
        raise SystemExit(f'--position-mm is only used with position source "manual", not {position!r}.')
    if position_mm is not None and len(position_mm) != 3:
        raise SystemExit('--position-mm takes three values (X Y Z in mm).')
    if position == 'active' and orientation == 'active':
        raise SystemExit('Position and orientation both "active": nothing to apply.')
    if components is not None:
        for part, source in (('position', position), ('orientation', orientation)):
            if source == 'calibration' and part not in components:
                raise SystemExit(f'The calibration YAML has no measured {part} '
                                 f'(tcp_components {sorted(components)}); pick another {part} source.')
    return position, orientation


def needs_yaml(position, orientation):
    return 'calibration' in (position, orientation)


def add_arguments(parser):
    """--position / --position-mm / --orientation for a `tcp` subcommand."""
    parser.add_argument('--position', choices=POSITION_SOURCES,
                        help='where the TCP position comes from (default: the YAML if it measured '
                             'one, else the active TCP; "manual" with --position-mm)')
    parser.add_argument('--position-mm', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='manual TCP position in mm, in the TCP parent link (flange) frame')
    parser.add_argument('--orientation', choices=ORIENTATION_SOURCES,
                        help='where the TCP orientation comes from (default: the YAML if it '
                             'measured one, else the active TCP)')


def composition_record(position, orientation, position_mm, base_id, default_path=None):
    """What the version was assembled from (stored with it)."""
    record = {'position': position, 'orientation': orientation}
    if position_mm is not None:
        record['position_mm'] = [round(float(v), 4) for v in position_mm]
    if 'active' in (position, orientation):
        record['active_from_id'] = base_id
    if orientation == 'default':
        record['default_yaml'] = default_path
    return record


def peek_components(calibration_yaml):
    """yaml_components() of a YAML without the acceptance gate; None if there is no file."""
    import os

    import yaml

    if not calibration_yaml:
        return None
    path = os.path.expanduser(calibration_yaml)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return yaml_components(yaml.safe_load(f) or {})


def require_yaml(calibration_yaml, position, orientation):
    """Clear error when a source needs the saved calibration but there is none."""
    import os

    if not calibration_yaml or not os.path.exists(os.path.expanduser(calibration_yaml)):
        parts = [p for p, s in (('position', position), ('orientation', orientation)) if s == 'calibration']
        raise SystemExit(f'No saved TCP calibration at {calibration_yaml!r}, but the {" and ".join(parts)} '
                         'should come from it. Save a calibration, or pick another source.')
