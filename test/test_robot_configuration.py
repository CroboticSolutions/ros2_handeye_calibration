import json
from pathlib import Path
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from hand_eye_calibration.robot_configuration import chain_joints,matching_group,matching_controller
from test_automatic_calibration import synthetic_sequence, sample_matrix


def profile(seven=False):
    fixture=Path(__file__).parent/'fixtures'
    root=ET.fromstring((fixture/'piper_calibration_chain.urdf').read_text())
    q=json.loads((fixture/'piper_calibration_joints.json').read_text())
    if seven:
        root.append(ET.fromstring('<joint name="joint7" type="revolute"><parent link="link6"/><child link="link7"/><origin xyz="0 0 .03"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" velocity="1"/></joint>'))
        q.append(0.)
    names=[]
    for i in range(1,len(q)+1):
        name=f'axis_{i}';root.find(f"joint[@name='joint{i}']").set('name',name);names.append(name)
    return dict(urdf=ET.tostring(root,encoding='unicode'),joints=q,names=names,tip=f'link{len(q)}')


@pytest.mark.parametrize('seven',[False,True])
def test_discovers_non_piper_joint_names_group_and_controller(seven):
    p=profile(seven);root=ET.fromstring(p['urdf'])
    names=chain_joints(root,'base_link',p['tip'])
    semantic=f'<robot><group name="manipulator"><chain base_link="base_link" tip_link="{p["tip"]}"/></group></robot>'
    assert names==p['names']
    assert matching_group(root,semantic,names)=='manipulator'
    controller=NS(name='robot/trajectory',state='active',type='joint_trajectory_controller/JointTrajectoryController',claimed_interfaces=[n+'/position' for n in names])
    assert matching_controller([controller],names)=='/robot/trajectory/follow_joint_trajectory'
    with pytest.raises(ValueError,match='one active'):
        matching_controller([controller,controller],names)


def test_unknown_group_and_wrong_robot_chain_are_rejected():
    p=profile();root=ET.fromstring(p['urdf'])
    with pytest.raises(ValueError,match='No robot chain'):chain_joints(root,'base_link','wrong_tip')
    with pytest.raises(ValueError,match='does not match'):matching_group(root,'<robot/>',p['names'],'arm')


def test_seven_axis_chain_is_refused_before_any_motion(monkeypatch):
    """ssik solves 6R chains only. The removed bootstrap/numeric samplers used to drive a
    seven-axis chain; now the run stops with an explicit error and sends no motion."""
    p=profile(True);r,n,mount=synthetic_sequence(monkeypatch,robot_profile=p)
    r.run()
    assert r.status['state']=='failed',r.status
    assert 'ssik' in r.status['message'] and 'no motion sent' in r.status['message']
    assert len(r.names)==7
    r.move.assert_not_called()
    n.save_calibration_service_callback.assert_not_called()
