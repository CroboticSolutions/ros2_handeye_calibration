from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from moveit_msgs.srv import GetMotionPlan
from trajectory_msgs.msg import JointTrajectoryPoint
from hand_eye_calibration.motion_planning import decode_path, plan_joint_path
from hand_eye_calibration import informative_views
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


def test_actual_route_is_followed_instead_of_straight_line(monkeypatch):
    monkeypatch.setattr(informative_views,'ViewInformation',lambda *a:NS(gain=lambda _:1.))
    monkeypatch.setattr(informative_views,'informative_view',lambda *a:True)
    robot=lambda q:np.eye(4)
    path=[np.array(q) for q in [[0,0],[0,.1],[.1,.1],[.1,0]]]
    runner=NS(check=lambda:None,plan_joint_path=Mock(return_value=path),framing=NS(contains=lambda *a,**k:True))
    planner=informative_views.InformativeViews.__new__(informative_views.InformativeViews)
    planner.s=NS(r=runner,estimate=np.eye(4),views=[],fk=NS(camera=robot),MAX_STEP=.05)
    planner.camera=robot;planner.goal=None;planner.route=[];planner.targets=[path[-1]]
    q=path[0];visited=[]
    for _ in range(6):
        q=planner.next_step(q,np.eye(4));visited.append(q)
    np.testing.assert_allclose(visited[0],[0,.05])
    np.testing.assert_allclose(visited[3],[.1,.1])
    np.testing.assert_allclose(visited[-1],[.1,0])
    runner.plan_joint_path.assert_called_once()


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
    runner.action.send_goal_async.assert_not_called()


def test_failed_top_candidates_do_not_starve_later_reachable_target(monkeypatch):
    monkeypatch.setattr(informative_views,'ViewInformation',lambda *a:NS(gain=lambda _:1.))
    monkeypatch.setattr(informative_views,'informative_view',lambda *a:True)
    robot=lambda q:np.eye(4)
    goals=[np.array([x]) for x in np.linspace(.1,.7,7)]
    def plan(start,goal):return [start,goal] if goal[0]>.65 else None
    runner=NS(check=lambda:None,plan_joint_path=Mock(side_effect=plan),framing=NS(contains=lambda *a,**k:True))
    planner=informative_views.InformativeViews.__new__(informative_views.InformativeViews)
    planner.s=NS(r=runner,estimate=np.eye(4),views=[],fk=NS(camera=robot),MAX_STEP=.05)
    planner.camera=robot;planner.goal=None;planner.route=[];planner.targets=goals
    assert planner.next_step(np.zeros(1),np.eye(4)) is None
    assert planner.pending_candidates
    assert planner.next_step(np.zeros(1),np.eye(4)) is not None
    assert runner.plan_joint_path.call_count==7
