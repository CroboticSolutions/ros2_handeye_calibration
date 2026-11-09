from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from moveit_msgs.srv import GetMotionPlan
from trajectory_msgs.msg import JointTrajectoryPoint
from hand_eye_calibration.motion_planning import decode_path, plan_joint_path
from test_automatic_calibration import synthetic_sequence


def response(names, points):
    result=GetMotionPlan.Response();p=result.motion_plan_response
    p.error_code.val=1;p.trajectory.joint_trajectory.joint_names=names
    for q in points:p.trajectory.joint_trajectory.points.append(JointTrajectoryPoint(positions=list(map(float,q))))
    return result


def test_reorders_moveit_joints_and_preserves_detour():
    result=response(['b','a'],[[0,0],[.1,0],[.1,.1],[0,.1]])
    path=decode_path(result,['a','b'],np.zeros(2),np.array([.1,0]),np.full(2,-1),np.ones(2))
    assert any(np.allclose(q,[0,.1]) for q in path)
    assert any(np.allclose(q,[.1,.1]) for q in path)


@pytest.mark.parametrize('points',[[[0,0],[float('nan'),0]],[[.1,0],[.2,0]],[[0,0],[2,0]]])
def test_invalid_trajectory_is_rejected(points):
    with pytest.raises(RuntimeError):
        decode_path(response(['a','b'],points),['a','b'],np.zeros(2),np.array([.2,0]),-np.ones(2),np.ones(2))


def test_planner_failure_returns_no_path():
    r=response(['a'],[[0],[.1]]);r.motion_plan_response.error_code.val=-1
    assert decode_path(r,['a'],np.zeros(1),np.array([.1]),-np.ones(1),np.ones(1)) is None


def test_missing_planner_stops_before_any_motion(monkeypatch):
    r,n,_=synthetic_sequence(monkeypatch)
    r.planning.wait_for_service=lambda **_:False
    r.run()
    assert r.status['state']=='failed'
    assert 'plan_kinematic_path' in r.status['message']
    r.move.assert_not_called()
    n.save_calibration_service_callback.assert_not_called()


def test_request_keeps_attached_state_and_never_executes_trajectory():
    from sensor_msgs.msg import JointState
    start=np.zeros(2);goal=np.array([.1,.2]);seen=[]
    def request(req):seen.append(req);return response(['b','a'],[[0,0],[.2,.1]])
    runner=NS(check=lambda:None,planning=NS(wait_for_service=lambda **_:True,call_async=request),
        names=['a','b'],move_group='different_robot_arm',joints=JointState(name=['a','b','gripper'],position=[0.,0.,.03]),
        lower=-np.ones(2),upper=np.ones(2),wait=lambda r,_:r,joint_positions=lambda:start.copy(),action=Mock())
    path=plan_joint_path(runner,start,goal)
    assert path is not None
    assert seen[0].motion_plan_request.start_state.is_diff
    assert seen[0].motion_plan_request.start_state.joint_state.position[-1]==.03
    assert seen[0].motion_plan_request.group_name=='different_robot_arm'
    # Not the group default: crx10ia_l's RRTstar spends the full planning time.
    assert seen[0].motion_plan_request.planner_id=='RRTConnect'
    runner.action.send_goal_async.assert_not_called()


def planning_runner(start, seen, points):
    from sensor_msgs.msg import JointState
    def request(req):seen.append(req);return response(['a','b'],points)
    return NS(check=lambda:None,planning=NS(wait_for_service=lambda **_:True,call_async=request),
        names=['a','b'],move_group='arm',joints=JointState(name=['a','b'],position=list(start)),
        lower=-np.ones(2),upper=np.ones(2),wait=lambda r,_:r,joint_positions=lambda:start.copy())


def test_start_just_past_a_limit_is_planned_from_the_limit():
    # Run 20261014_090523: robot settled at j5 = 1.22028 > 1.22 and MoveIt refused the return.
    start=np.array([0.,1.0028]);goal=np.array([.1,.9]);seen=[]
    path=plan_joint_path(planning_runner(start,seen,[[0,1],[.1,.9]]),start,goal)
    assert path is not None
    assert seen[0].motion_plan_request.start_state.joint_state.position[1]==1.
    # A goal that was such a measured pose is likewise planned at the limit.
    seen.clear();back=np.array([0.,1.0028])
    assert plan_joint_path(planning_runner(goal,seen,[[.1,.9],[0,1]]),goal,back) is not None
    assert seen[0].motion_plan_request.goal_constraints[0].joint_constraints[1].position==1.


def test_start_well_past_a_limit_is_sent_unchanged():
    start=np.array([0.,1.02]);seen=[]
    runner=planning_runner(start,seen,[[0,1.02],[.1,.9]])
    request=runner.planning.call_async
    def refused(req):
        result=request(req);result.motion_plan_response.error_code.val=-1;return result
    runner.planning.call_async=refused
    assert plan_joint_path(runner,start,np.array([.1,.9])) is None
    assert seen[0].motion_plan_request.start_state.joint_state.position[1]==1.02
