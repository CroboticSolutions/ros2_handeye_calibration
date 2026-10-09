import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from hand_eye_calibration import welding_gun_apply_cli as cli

# TCP-v1 (2026-09-23) and the arm_tcp_joint origin that was baked by hand from it.
TCP_V1 = {'tx': 0.05971650727739351, 'ty': -0.002810430193906347, 'tz': 0.19015803585545335,
          'qx': 0.3245431126388426, 'qy': 0.008110963748323693, 'qz': 0.9458320329677071,
          'qw': 0.002783112994304413}
TCP_V1_RPY = [-0.027929237913, -0.909524738174, 0.022044250167]


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


TCP_BASE = {'calibration_id': 'repo-default-tcp', 'parent_link': 'piper_welding_gun', 'child_link': 'arm_tcp',
            'real': {'xyz': '0 0 0', 'rpy': '-0.03 -0.9 0.02'},
            'sim': {'xyz': '0.05360585785 0 0.2085160675', 'rpy': '0 -0.9599310886 0'}}


@pytest.fixture
def tcp_store(tmp_path, monkeypatch):
    monkeypatch.setenv('ARMS_CALIBRATION_DIR', str(tmp_path / 'store'))
    monkeypatch.setenv('ARMS_ROBOT_ID', 'test_robot')
    default = tmp_path / 'tcp_default.yaml'
    default.write_text(yaml.safe_dump(TCP_BASE))
    monkeypatch.setattr(cli, 'TCP_DEFAULT_YAML', str(default))
    return tmp_path


def test_tcp_real_origin_reproduces_hand_baked_origin(tcp_store):
    path = write(tcp_store, 'tcp.yaml', {'transform': TCP_V1, 'parent_frame': 'link6'})
    calibration_id, record = cli.apply_tcp(path)
    xyz = np.fromstring(record['real']['xyz'], sep=' ')
    rpy = np.fromstring(record['real']['rpy'], sep=' ')
    np.testing.assert_allclose(rpy, TCP_V1_RPY, atol=1e-9)
    np.testing.assert_allclose(xyz, [TCP_V1['tx'], TCP_V1['ty'], TCP_V1['tz']])
    assert record['sim'] == TCP_BASE['sim']  # simulation branch untouched
    # arm_tcp +X is the calibrated tool +Z.
    _, T = cli.load_calibration(path)
    tcp_x = Rotation.from_euler('xyz', rpy).as_matrix() @ [1, 0, 0]
    np.testing.assert_allclose(tcp_x, T[:3, :3] @ [0, 0, 1], atol=1e-9)
    mount = cli.calibration_store.WELDING_GUN_TCP
    assert cli.calibration_store.active(mount)['calibration_id'] == calibration_id
    assert record['previous_id'] == cli.calibration_store.versions(mount)[-1]['calibrationId']


def test_tcp_any_change_is_accepted_for_a_first_calibration(tcp_store):
    """Unknown tool, first calibration: no limit on how far the TCP or wire axis moves."""
    flipped = {'tx': 0.2, 'ty': 0.1, 'tz': -0.05, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0}
    _, record = cli.apply_tcp(write(tcp_store, 'tcp.yaml', {'transform': flipped}))
    assert record['delta_from_previous']['real']['translation_mm'] > 200
    assert record['delta_from_previous']['real']['wire_axis_deg'] > 30


def test_tcp_offset_along_wire_is_a_new_version_and_rolls_back(tcp_store):
    mount = cli.calibration_store.WELDING_GUN_TCP
    first, _ = cli.apply_tcp(write(tcp_store, 'tcp.yaml', {'transform': TCP_V1}))
    second, record = cli.offset_tcp(-8.0)
    before, after = cli._origin_matrix(cli.calibration_store.load(
        cli.calibration_store._version_file(mount, first))['real']), cli._origin_matrix(record['real'])
    np.testing.assert_allclose(after[:3, 3] - before[:3, 3], before[:3, 0] * -0.008, atol=1e-12)
    assert record['real']['rpy'] == cli.calibration_store.load(cli.calibration_store._version_file(mount, first))['real']['rpy']
    assert record['manual_offset'] == {'along_wire_mm': -8.0, 'from_id': first}
    cli.main(['activate', '--mount', 'tcp', first])
    assert cli.calibration_store.active(mount)['calibration_id'] == first


BASE = {'calibration_id': 'repo-default', 'parent_link': 'link5',
        'real': {'xyz': '4 5 6', 'rpy': '0 0 0'}, 'nominal': {'xyz': '1 2 3', 'rpy': '0 0 0'}}
CAM = {'calibration_type': 'eye-in-hand', 'robot_effector_frame': 'link5',
       'tracking_base_frame': 'camera_color_optical_frame',
       'acceptance': {'passed': True, 'summary': 'ok', 'checks': [{'name': 'sample_count', 'value': 28}]}}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('ARMS_CALIBRATION_DIR', str(tmp_path / 'store'))
    monkeypatch.setenv('ARMS_ROBOT_ID', 'test_robot')
    default = tmp_path / 'default.yaml'
    default.write_text(yaml.safe_dump(BASE))
    monkeypatch.setattr(cli, 'FEMTO_DEFAULT_YAML', str(default))
    monkeypatch.setattr(cli, 'femto_parent_link', lambda: 'link5')
    return tmp_path


def _cam_yaml(tmp_path, T, name='cal.yaml'):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    transform = dict(zip(('tx', 'ty', 'tz'), T[:3, 3].tolist())) | dict(zip(('qx', 'qy', 'qz', 'qw'), q.tolist()))
    return write(tmp_path, name, CAM | {'transform': transform})


def test_camera_real_origin_is_calibration_times_inverse_inner_chain(store):
    T = cli._matrix([-.11, -.03, .01], [.3, -.2, 1.5])
    inner = cli._matrix([.01, .02, .03], [-1.57, 0, -1.57])
    calibration_id, record = cli.apply_camera(_cam_yaml(store, T), 'real', inner)
    np.testing.assert_allclose(cli._origin_matrix(record['real']) @ inner, T, atol=1e-9)
    assert record['nominal'] == BASE['nominal']  # nominal carried over
    seeded = cli.calibration_store.versions(cli.calibration_store.FEMTO_HANDEYE)[-1]
    assert record['updated'] == ['real'] and record['previous_id'] == seeded['calibrationId']
    assert seeded['label'].startswith('imported from')
    assert record['metrics']['checks'] == {'sample_count': 28}
    active = cli.calibration_store.active(cli.calibration_store.FEMTO_HANDEYE)
    assert active['calibration_id'] == calibration_id


def test_versions_are_immutable_and_rollback_repoints_current(store):
    mount = cli.calibration_store.FEMTO_HANDEYE
    inner = np.eye(4)
    first, _ = cli.apply_camera(_cam_yaml(store, cli._matrix([.1, 0, 0], [0, 0, 0])), 'real', inner)
    second, rec = cli.apply_camera(_cam_yaml(store, cli._matrix([.1, .005, 0], [0, 0, 0]), 'b.yaml'), 'real', inner)
    assert rec['previous_id'] == first
    assert rec['delta_from_previous']['real']['translation_mm'] == pytest.approx(5.0)
    assert [v['calibrationId'] for v in cli.calibration_store.versions(mount)][:2] == [second, first]
    cli.calibration_store.activate(mount, first)
    assert cli.calibration_store.active(mount)['calibration_id'] == first
    assert [v['active'] for v in cli.calibration_store.versions(mount)] == [False, True, False]
    with pytest.raises(ValueError):
        cli.calibration_store.activate(mount, '../../etc')


def test_camera_from_other_parent_is_refused(store):
    path = write(store, 'x.yaml', CAM | {'robot_effector_frame': 'link6', 'transform': TCP_V1})
    with pytest.raises(SystemExit):
        cli.apply_camera(path, 'real', np.eye(4))


def test_failed_acceptance_is_refused(tmp_path):
    path = write(tmp_path, 'cal.yaml', {'transform': TCP_V1, 'acceptance': {'passed': False, 'summary': 'x'}})
    with pytest.raises(SystemExit):
        cli.load_calibration(path)
    cli.load_calibration(path, force=True)
