import os
import subprocess

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from hand_eye_calibration import crx_apply_cli as cli
from hand_eye_calibration.welding_gun_apply_cli import chain_from_urdf

pytestmark = pytest.mark.skipif(
    subprocess.run(['ros2', 'pkg', 'prefix', 'crx_description'], capture_output=True).returncode != 0,
    reason='needs the crx_description overlay (xacro)')

# A plausible flange -> optical pose: 12 cm forward, 9 cm up, looking along the tool.
OPTICAL = np.eye(4)
OPTICAL[:3, :3] = Rotation.from_euler('xyz', [-1.4, 0.1, -1.6]).as_matrix()
OPTICAL[:3, 3] = [0.12, -0.01, 0.09]


def _transform(T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return {'tx': float(T[0, 3]), 'ty': float(T[1, 3]), 'tz': float(T[2, 3]),
            'qx': float(q[0]), 'qy': float(q[1]), 'qz': float(q[2]), 'qw': float(q[3])}


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


def handeye(tmp_path, optical='camera_color_optical_frame', effector='flange'):
    return write(tmp_path, 'he.yaml', {
        'calibration_type': 'eye-in-hand', 'robot_effector_frame': effector,
        'tracking_base_frame': optical, 'transform': _transform(OPTICAL),
        'acceptance': {'passed': True, 'summary': 'ok', 'checks': []}})


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('ARMS_CALIBRATION_DIR', str(tmp_path / 'store'))
    monkeypatch.delenv('ARMS_ROBOT_ID', raising=False)
    return tmp_path


def _xacro(path, *args):
    return subprocess.run(['xacro', path, *args], check=True, capture_output=True, text=True).stdout


def _share(pkg):
    prefix = subprocess.run(['ros2', 'pkg', 'prefix', pkg], check=True, capture_output=True, text=True)
    return os.path.join(prefix.stdout.strip(), 'share', pkg)


def _sim_urdf(camera, cal='', tcp=''):
    return _xacro(f'{_share("seam_ros2_pkg")}/urdf/crx10ia_gz.urdf.xacro', 'model:=crx10ia_l',
                  f'wrist_camera:={camera}', f'camera_calibration:={cal}', f'tcp_calibration:={tcp}')


def _real_urdf(camera, cal='', tcp=''):
    return _xacro(f'{_share("crx_description")}/urdf/crx10ia_l/crx10ia_l.xacro', f'wrist_camera:={camera}',
                  f'camera_calibration:={cal}', f'tcp_calibration:={tcp}')


@pytest.mark.parametrize('camera,gz', [('femto_bolt', 'femto_bolt'), ('realsense_d435', 'gazebo_realsense')])
def test_nominal_camera_version_reproduces_the_calibration_in_gazebo(store, camera, gz):
    # Gazebo reports its optical frame under the old FANUC alias.
    path = handeye(store, optical='oak_rgb_camera_optical_frame')
    _, record = cli.apply_camera('crx10ia_l', camera, path, target='nominal')
    assert record['updated'] == ['nominal'] and record['parent_link'] == 'flange'
    current = cli.calibration_store.current_path(cli.CAMERAS[camera]['mount'], 'crx10ia_l')
    urdf = _sim_urdf(gz, cal=current)
    T = chain_from_urdf(urdf, 'oak_rgb_camera_optical_frame', 'flange')
    np.testing.assert_allclose(T, OPTICAL, atol=1e-9)


def test_real_d435_version_reproduces_the_calibration_on_the_robot_urdf(store):
    _, record = cli.apply_camera('crx10ia_l', 'realsense_d435', handeye(store), target='real')
    assert record['child_link'] == 'camera_bottom_screw_frame'
    current = cli.calibration_store.current_path(cli.calibration_store.CRX_D435_HANDEYE, 'crx10ia_l')
    T = chain_from_urdf(_real_urdf('realsense_d435', cal=current), 'camera_color_optical_frame', 'flange')
    np.testing.assert_allclose(T, OPTICAL, atol=1e-9)


def test_real_femto_uses_the_driver_inner_chain(store):
    inner = np.eye(4)
    inner[:3, 3] = [0.01, 0.02, 0.03]
    inner[:3, :3] = Rotation.from_euler('xyz', [-np.pi / 2, 0, -np.pi / 2]).as_matrix()
    _, record = cli.apply_camera('crx10ia_l', 'femto_bolt', handeye(store), target='real', inner_real=inner)
    mount = cli._origin_matrix(record['real'])
    np.testing.assert_allclose(mount @ inner, OPTICAL, atol=1e-9)
    # Store default carried over for the simulation origin.
    assert record['nominal'] == yaml.safe_load(open(cli.camera_default('femto_bolt')))['nominal']


def test_camera_calibration_must_be_relative_to_the_flange(store):
    with pytest.raises(SystemExit, match='flange'):
        cli.apply_camera('crx10ia_l', 'femto_bolt', handeye(store, effector='J6_link'), target='nominal')


def test_robots_have_separate_stores(store):
    cli.apply_camera('crx10ia', 'realsense_d435', handeye(store), target='nominal')
    mount = cli.calibration_store.CRX_D435_HANDEYE
    assert cli.calibration_store.active(mount, robot='crx10ia') is not None
    assert cli.calibration_store.active(mount, robot='crx10ia_l') is None
    assert '/crx10ia/' in cli.calibration_store.current_path(mount, 'crx10ia')


def _tool(z_axis, xyz):
    z = np.asarray(z_axis, float) / np.linalg.norm(z_axis)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = xyz
    return T


@pytest.mark.parametrize('target,link,urdf', [('real', 'tcp', _real_urdf), ('sim', 'welding_tcp', _sim_urdf)])
def test_tcp_puts_the_target_wire_axis_on_the_calibrated_tool_z(store, target, link, urdf):
    tool = _tool([0.9, 0.05, -0.4], [0.41, 0.003, -0.02])
    path = write(store, 'tcp.yaml', {'transform': _transform(tool), 'parent_frame': 'flange'})
    _, record = cli.apply_tcp('crx10ia_l', path, target=target)
    current = cli.calibration_store.current_path(cli.calibration_store.CRX_WELDING_TCP, 'crx10ia_l')
    T = chain_from_urdf(urdf('femto_bolt', tcp=current), link, 'flange')
    np.testing.assert_allclose(T[:3, 3], tool[:3, 3], atol=1e-9)
    wire = np.asarray(record[target]['wire_axis_tcp'], float)
    np.testing.assert_allclose(T[:3, :3] @ (wire / np.linalg.norm(wire)), tool[:3, 2], atol=1e-9)
    other = 'sim' if target == 'real' else 'real'
    assert record[other] == yaml.safe_load(open(cli.tcp_default()))[other]


def test_tcp_any_change_is_accepted_for_a_first_calibration(store):
    tool = _tool([-1, 0, 0], [0.0, 0.3, 0.2])
    _, record = cli.apply_tcp('crx10ia_l', write(store, 'tcp.yaml', {'transform': _transform(tool)}), 'real')
    assert record['delta_from_previous']['real']['wire_axis_deg'] > 170


def test_tcp_offset_moves_along_the_wire_and_rolls_back(store):
    mount = cli.calibration_store.CRX_WELDING_TCP
    first, before = cli.apply_tcp('crx10ia_l', write(store, 'tcp.yaml', {
        'transform': _transform(_tool([1, 0, 0], [0.4, 0, 0]))}), 'sim')
    _, record = cli.offset_tcp('crx10ia_l', 5.0, 'sim')
    delta = cli._origin_matrix(record['sim'])[:3, 3] - cli._origin_matrix(before['sim'])[:3, 3]
    np.testing.assert_allclose(delta, [0.005, 0, 0], atol=1e-9)
    cli.main(['--robot', 'crx10ia_l', 'activate', '--mount', 'tcp', first])
    assert cli.calibration_store.active(mount, robot='crx10ia_l')['calibration_id'] == first


def test_failed_acceptance_is_refused_without_force(store):
    path = write(store, 'tcp.yaml', {'transform': _transform(np.eye(4)),
                                     'acceptance': {'passed': False, 'summary': 'rms'}})
    with pytest.raises(SystemExit, match='acceptance failed'):
        cli.apply_tcp('crx10ia_l', path)
    cli.apply_tcp('crx10ia_l', path, force=True)
