"""Real DataCollector: reprojection solve, acceptance gate, dataset, offline CLI."""
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
                     '-p', 'solver:=reprojection', '-p', 'bootstrap_samples:=10', '-p', "pointcloud_topic:=''"])
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


def test_clean_run_is_saved_with_reprojection_evidence_and_dataset(node):
    n, calibration, tmp = node
    load(n, make_samples())
    response = n.save_calibration_service_callback(Trigger.Request(), Trigger.Response())
    assert response.success, response.message
    data = yaml.safe_load(calibration.read_text())
    assert data['solver'] == 'reprojection'
    assert data['acceptance']['passed']
    assert data['reprojection']['leave_one_out']['position_rms_m'] < .003
    assert os.path.exists(str(calibration) + '.previous')
    run = data['dataset_path']
    assert os.path.exists(os.path.join(run, 'dataset.json'))
    dataset = calibration_dataset.load(run)
    assert len(dataset['samples']) == 15
    out = offline_solve_cli.solve(dataset, bootstrap=0)
    assert out['difference_reprojection_vs_online']['translation_m'] < 1e-4
    assert offline_solve_cli.main([run, '--bootstrap', '10', '--yaml', str(tmp / 'offline.yaml')]) == 0


def test_inconsistent_robot_poses_do_not_replace_active_file(node):
    n, calibration, _ = node
    load(n, make_samples(robot_noise_m=.004))
    response = n.save_calibration_service_callback(Trigger.Request(), Trigger.Response())
    assert not response.success
    assert 'Acceptance failed' in response.message
    assert calibration.read_text() == 'previous: true\n'
    rejected = yaml.safe_load(open(str(calibration) + '.rejected.yaml'))
    assert not rejected['acceptance']['passed']
    assert os.path.exists(os.path.join(rejected['dataset_path'], 'dataset.json'))


def test_too_few_samples_are_refused(node):
    n, calibration, _ = node
    load(n, make_samples(n=8))
    response = n.save_calibration_service_callback(Trigger.Request(), Trigger.Response())
    assert not response.success
    assert 'sample_count' in response.message
    assert calibration.read_text() == 'previous: true\n'


def test_validation_samples_are_excluded_from_the_fit(node):
    n, _, _ = node
    load(n, make_samples(n=18))
    n.sample_roles[15:] = ['validation'] * 3
    cal = n.get_calibration(full=True)
    assert cal is not None
    report = n._last_reprojection
    assert report['training_views'] == 15
    assert report['validation_views']['views'] == 3


def test_status_blocks_save_below_minimum(node):
    n, _, _ = node
    load(n, make_samples(n=6))
    n._publish_status(None, None)
    assert not n._status_payload['ready_to_save']
    assert n._status_payload['min_save_samples'] == 12
