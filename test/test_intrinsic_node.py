"""Actual ROS collector, isolated domain: no hardware nodes, no robot commands."""
import copy
import json
import numpy as np
import pytest
import rclpy
import yaml
from std_srvs.srv import Trigger
from hand_eye_calibration.node import DataCollector
from test_node_save_gate import load
from test_reprojection_calibration import make_samples


@pytest.fixture
def node(tmp_path):
    active=tmp_path/'active.yaml';active.write_text('previous: true\n')
    rclpy.init(args=['--ros-args','-p','solver:=intrinsic_pose','-p','bootstrap_samples:=10',
                     '-p','calibration_type:=eye-in-hand','-p',f'calibration_file:={active}',
                     '-p',f'dataset_dir:={tmp_path / "runs"}'])
    n=DataCollector()
    yield n,active
    n.automatic.close();n.destroy_node();rclpy.shutdown()


def test_save_uses_new_solver_and_preserves_all_views(node):
    n,path=node;load(n,make_samples(n=24,robot_noise_m=.0001))
    response=n.save_calibration_service_callback(Trigger.Request(),Trigger.Response())
    assert response.success,response.message
    data=yaml.safe_load(path.read_text())
    assert data['solver']=='intrinsic_pose'
    assert data['algorithm_used']=='SHAH+NONLINEAR'
    assert data['rejected_sample_indices']==[]
    assert data['pose_calibration']['training_views']==24
    assert data['reprojection'] is None
    assert path.with_name('active.yaml.previous').read_text()=='previous: true\n'


def test_held_out_samples_cannot_change_training_fit(node):
    n,_=node;load(n,make_samples(n=25,robot_noise_m=.0001))
    n.sample_roles[20:]=['validation']*5
    first=n.get_calibration(full=True)
    n.tracking_samples[-1][0]+=.05
    second=n.get_calibration(full=True)
    np.testing.assert_array_equal(first,second)
    assert n._last_reprojection['validation_views']['position_max_m']>.04
    response=n.save_calibration_service_callback(Trigger.Request(),Trigger.Response())
    assert not response.success


def test_inconsistent_dataset_never_replaces_active_calibration(node):
    n,path=node;samples=make_samples(n=24);samples[-1]['tracking'][0]+=.1;load(n,samples)
    response=n.save_calibration_service_callback(Trigger.Request(),Trigger.Response())
    assert not response.success
    assert path.read_text()=='previous: true\n'
    data=yaml.safe_load(path.with_name('active.yaml.rejected.yaml').read_text())
    assert data['rejected_sample_indices']==[]
    assert data['pose_calibration']['training_views']==24
