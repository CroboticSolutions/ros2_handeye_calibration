"""Real DataCollector (intrinsic_pose): acceptance gate, forced save, dataset, offline CLI."""
import os

import pytest
import rclpy
import yaml
from std_srvs.srv import Trigger

from hand_eye_calibration import calibration_dataset, offline_solve_cli
from test_reprojection_calibration import K, make_samples


@pytest.fixture
def node(tmp_path):
    from hand_eye_calibration.node import DataCollector
    calibration = tmp_path / 'hand_eye_calibration.yaml'
    calibration.write_text('previous: true\n')
    rclpy.init(args=['--ros-args',
                     '-p', f'calibration_file:={calibration}',
                     '-p', f'dataset_dir:={tmp_path / "runs"}',
                     '-p', 'calibration_type:=eye-in-hand',
                     '-p', 'tracking_base_frame:=camera', '-p', 'tracking_marker_frame:=board',
                     '-p', 'robot_base_frame:=base', '-p', 'robot_effector_frame:=tool',
                     '-p', 'bootstrap_samples:=10', '-p', "pointcloud_topic:=''"])
    n = DataCollector()
    n._camera_model = {'k': K, 'd': [0.] * 5, 'width': 1280, 'height': 720, 'distortion_model': 'plumb_bob'}
    yield n, calibration, tmp_path
    n.automatic.close()
    n.destroy_node()
    rclpy.shutdown()


def load(node, samples):
    for s in samples:
        node.robot_samples.append(s['robot'])
        node.tracking_samples.append(s['tracking'])
        node.sample_metrics.append({})
        node.sample_frames.append(s['frames'])
        node.sample_joints.append(None)
        node.sample_roles.append('training')
        node.sample_images.append(None)


def test_clean_manual_run_is_saved_with_dataset_and_says_it_has_no_held_out_views(node):
    n, calibration, tmp = node
    load(n, make_samples(n=22, robot_noise_m=.0001))
    response = n.save_calibration_service_callback(Trigger.Request(), Trigger.Response())
    assert response.success, response.message
    data = yaml.safe_load(calibration.read_text())
    assert data['solver'] == 'intrinsic_pose'
    assert data['acceptance']['passed']
    # Manual capture: every view is training; the verdict and status must not claim validation.
    assert data['acceptance']['held_out_validation'] is False
    assert 'no held-out validation' in data['acceptance']['summary']
    assert 'no held-out validation' in n._status_payload['summary']
    assert n._status_payload['held_out_validation'] is False
    assert 'reprojection' not in data and 'axxb_transform' not in data
    assert os.path.exists(str(calibration) + '.previous')
    run = data['dataset_path']
    assert os.path.exists(os.path.join(run, 'dataset.json'))
    dataset = calibration_dataset.load(run)
    assert len(dataset['samples']) == 22
    # The offline reprojection solve (handeye_offline_solve) still reads the dataset.
    assert offline_solve_cli.main([run, '--bootstrap', '0', '--yaml', str(tmp / 'offline.yaml')]) in (0, 2)


def test_legacy_solver_is_refused(tmp_path):
    from hand_eye_calibration.node import DataCollector
    rclpy.init(args=['--ros-args', '-p', 'solver:=reprojection',
                     '-p', f'calibration_file:={tmp_path / "x.yaml"}', '-p', "pointcloud_topic:=''"])
    try:
        with pytest.raises(ValueError, match='removed'):
            DataCollector()
    finally:
        rclpy.shutdown()


def test_inconsistent_robot_poses_do_not_replace_active_file(node):
    n, calibration, _ = node
    load(n, make_samples(n=22, robot_noise_m=.004))
    response = n.save_calibration_service_callback(Trigger.Request(), Trigger.Response())
    assert not response.success
    assert 'Acceptance failed' in response.message
    assert calibration.read_text() == 'previous: true\n'
    rejected = yaml.safe_load(open(str(calibration) + '.rejected.yaml'))
    assert not rejected['acceptance']['passed']
    assert os.path.exists(os.path.join(rejected['dataset_path'], 'dataset.json'))


def test_force_save_makes_a_failed_candidate_active_and_records_it(node):
    n, calibration, _ = node
    load(n, make_samples(n=22, robot_noise_m=.004))
    response = n.force_save_calibration_service_callback(Trigger.Request(), Trigger.Response())
    assert response.success, response.message
    assert 'FORCED despite failed acceptance' in response.message
    data = yaml.safe_load(calibration.read_text())
    assert data['forced'] is True
    assert not data['acceptance']['passed']          # the verdict stays truthful
    assert os.path.exists(str(calibration) + '.previous')
    assert yaml.safe_load(open(os.path.join(data['dataset_path'], 'calibration.yaml')))['forced'] is True
    # A passing candidate saved through the override is not marked forced.
    n.robot_samples.clear(); n.tracking_samples.clear(); n.sample_metrics.clear()
    n.sample_frames.clear(); n.sample_joints.clear(); n.sample_roles.clear(); n.sample_images.clear()
    load(n, make_samples(n=22, robot_noise_m=.0001))
    assert n.force_save_calibration_service_callback(Trigger.Request(), Trigger.Response()).success
    assert yaml.safe_load(calibration.read_text())['forced'] is False


def test_too_few_samples_are_refused(node):
    n, calibration, _ = node
    load(n, make_samples(n=8))
    response = n.save_calibration_service_callback(Trigger.Request(), Trigger.Response())
    assert not response.success
    assert 'sample_count' in response.message
    assert calibration.read_text() == 'previous: true\n'


def test_validation_samples_are_excluded_from_the_fit(node):
    n, _, _ = node
    load(n, make_samples(n=25, robot_noise_m=.0001))
    n.sample_roles[22:] = ['validation'] * 3
    cal = n.get_calibration(full=True)
    assert cal is not None
    report = n._last_reprojection
    assert report['training_views'] == 22
    assert report['validation_views']['views'] == 3


def test_status_blocks_save_below_minimum(node):
    n, _, _ = node
    load(n, make_samples(n=6))
    n._publish_status(None, None)
    assert not n._status_payload['ready_to_save']
    assert n._status_payload['min_save_samples'] == 12
