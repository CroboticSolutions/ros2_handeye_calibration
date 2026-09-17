"""Request a MoveIt joint path; retain the timed trajectory for continuous execution."""
import copy
import numpy as np
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes


def decode_path(response, names, start, goal, lower, upper):
    if response is None or response.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
        return None
    trajectory = response.motion_plan_response.trajectory
    joints = trajectory.joint_trajectory
    if trajectory.multi_dof_joint_trajectory.points or set(joints.joint_names) != set(names) or len(joints.joint_names) != len(names):
        raise RuntimeError('MoveIt returned an incompatible calibration trajectory.')
    order = [joints.joint_names.index(n) for n in names]
    path=[]
    for point in joints.points:
        if len(point.positions) != len(names):
            raise RuntimeError('MoveIt returned incomplete joint positions.')
        q=np.asarray(point.positions)[order]
        if not np.isfinite(q).all() or np.any(q<lower) or np.any(q>upper):
            raise RuntimeError('MoveIt returned invalid joint positions.')
        path.append(q)
    if not path or np.max(np.abs(path[0]-start))>.015 or np.max(np.abs(path[-1]-goal))>.015:
        raise RuntimeError('MoveIt path does not match the requested start and goal.')
    # Bound travel, including a planner that takes an unexpectedly long detour.
    path=[start.copy()]+path+[goal.copy()]
    if sum(np.max(np.abs(b-a)) for a,b in zip(path,path[1:])) > 2.4:
        return None
    # Remove duplicate/collinear interpolation points without cutting corners.
    # Each corner is retained, so this is the same geometric path.
    compact=[]
    for q in path:
        if compact and np.max(np.abs(q-compact[-1]))<1e-9:
            continue
        while len(compact)>=2:
            a,b=compact[-2:];direction=q-a
            length=float(direction@direction)
            fraction=float((b-a)@direction/length) if length>1e-16 else -1.
            if not 0<=fraction<=1 or np.max(np.abs(a+fraction*direction-b))>1e-8:
                break
            compact.pop()
        compact.append(q)
    return compact


def plan_joint_path(runner, start, goal):
    runner.check()
    if not runner.planning.wait_for_service(timeout_sec=1):
        raise RuntimeError('MoveIt /plan_kinematic_path is unavailable; no calibration motion was sent.')
    req=GetMotionPlan.Request()
    motion=req.motion_plan_request
    motion.group_name=runner.move_group
    motion.num_planning_attempts=2
    motion.allowed_planning_time=2.
    motion.max_velocity_scaling_factor=.3
    motion.max_acceleration_scaling_factor=.3
    motion.start_state.is_diff=True
    motion.start_state.joint_state=copy.deepcopy(runner.joints)
    positions=dict(zip(runner.names,start))
    motion.start_state.joint_state.position=[float(positions.get(n,p))
        for n,p in zip(runner.joints.name,runner.joints.position)]
    constraints=Constraints()
    for name,value in zip(runner.names,goal):
        constraints.joint_constraints.append(JointConstraint(joint_name=name,position=float(value),
            tolerance_above=.003,tolerance_below=.003,weight=1.))
    motion.goal_constraints=[constraints]
    result=runner.wait(runner.planning.call_async(req),5)
    runner.check()
    if np.max(np.abs(runner.joint_positions()-start))>.01:
        raise RuntimeError('Robot moved during planning; no calibration motion was sent.')
    path = decode_path(result,runner.names,start,goal,runner.lower,runner.upper)
    runner.planned_trajectory = None
    if path is not None:
        trajectory = copy.deepcopy(result.motion_plan_response.trajectory.joint_trajectory)
        order = [trajectory.joint_names.index(n) for n in runner.names]
        trajectory.joint_names = list(runner.names)
        for point in trajectory.points:
            point.positions = [point.positions[i] for i in order]
            for field in ('velocities', 'accelerations', 'effort'):
                values = getattr(point, field)
                if values:
                    if len(values) != len(order) or not np.isfinite(values).all():
                        raise RuntimeError('MoveIt returned invalid trajectory derivatives.')
                    setattr(point, field, [values[i] for i in order])
        runner.planned_trajectory = trajectory
    return path
