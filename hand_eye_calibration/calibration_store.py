"""Per-robot calibration store outside the source tree.

Layout (one directory per mount, e.g. the welding-gun Femto Bolt):

    $ARMS_CALIBRATION_DIR (default ~/.ros/calibration)
      /<robot_id>/<mount>/calib_<UTC>.yaml   immutable versions
      /<robot_id>/<mount>/current.yaml       symlink to the active version

The xacro reads current.yaml through xacro.load_yaml (the launch files and
robot_description_sources pass its path), so Apply = write a new version and
repoint the symlink; rollback = repoint it at an older version. Nothing is
ever rewritten in place and the repo only ships a default file.
"""
import datetime
import glob
import hashlib
import os
import re
import socket

import yaml

CURRENT = 'current.yaml'
FEMTO_HANDEYE = 'femto_bolt_handeye'
WELDING_GUN_TCP = 'welding_gun_tcp'


def root():
    return os.path.expanduser(os.environ.get('ARMS_CALIBRATION_DIR', '~/.ros/calibration'))


def robot_id():
    return os.environ.get('ARMS_ROBOT_ID', 'piper')


def mount_dir(mount):
    return os.path.join(root(), robot_id(), mount)


def current_path(mount):
    return os.path.join(mount_dir(mount), CURRENT)


def resolve(mount, default_path):
    """File the xacro should load: the active version, else the repo default."""
    path = current_path(mount)
    return path if os.path.exists(path) else default_path


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def active(mount, default_path=None):
    path = current_path(mount)
    if os.path.exists(path):
        data = load(path)
        data['_file'] = os.path.realpath(path)
        return data
    if default_path is not None and os.path.exists(default_path):
        data = load(default_path)
        data['_file'] = default_path
        return data
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(os.path.expanduser(path), 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def _version_file(mount, calibration_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', calibration_id or ''):
        raise ValueError(f'Invalid calibration id {calibration_id!r}')
    return os.path.join(mount_dir(mount), f'calib_{calibration_id}.yaml')


def save_version(mount, record, now=None):
    """Write an immutable version; returns (calibration_id, path, full record). Does not activate."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    calibration_id = now.strftime('%Y%m%dT%H%M%S%fZ')
    previous = active(mount)
    record = {
        'calibration_id': calibration_id,
        'created_utc': now.isoformat(),
        'robot_id': robot_id(),
        'mount': mount,
        'host': socket.gethostname(),
        'previous_id': previous.get('calibration_id') if previous else None,
        **record,
    }
    os.makedirs(mount_dir(mount), exist_ok=True)
    path = _version_file(mount, calibration_id)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        yaml.safe_dump(record, f, sort_keys=False)
    os.replace(tmp, path)
    return calibration_id, path, record


def ensure_seeded(mount, default_path):
    """First use: store the default as an activated version, so a rollback to
    what the robot ran before the first Apply is always possible."""
    if os.path.exists(current_path(mount)) or not os.path.exists(default_path):
        return None
    data = load(default_path)
    data.pop('calibration_id', None)
    data['label'] = f'imported from {default_path}'
    data['updated'] = []
    calibration_id, _path, _record = save_version(mount, data)
    activate(mount, calibration_id)
    return calibration_id


def activate(mount, calibration_id):
    """Atomically point current.yaml at an existing version."""
    path = _version_file(mount, calibration_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'No calibration {calibration_id!r} for {mount} in {mount_dir(mount)}')
    link = current_path(mount)
    tmp = link + '.tmp'
    if os.path.lexists(tmp):
        os.unlink(tmp)
    os.symlink(os.path.basename(path), tmp)
    os.replace(tmp, link)
    return path


def versions(mount):
    """Newest first, with the fields a UI needs to pick a rollback target."""
    current = os.path.realpath(current_path(mount)) if os.path.exists(current_path(mount)) else None
    out = []
    for path in sorted(glob.glob(os.path.join(mount_dir(mount), 'calib_*.yaml')), reverse=True):
        try:
            data = load(path)
        except (OSError, yaml.YAMLError):
            continue
        out.append({
            'calibrationId': data.get('calibration_id'),
            'createdUtc': data.get('created_utc'),
            'label': data.get('label'),
            'updated': data.get('updated'),
            'previousId': data.get('previous_id'),
            'metrics': data.get('metrics'),
            'active': os.path.realpath(path) == current,
        })
    return out
