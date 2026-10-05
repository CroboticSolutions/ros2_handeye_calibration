# Copyright 2026 Intrinsic Innovation LLC
# Modifications: ROS/MoveIt adaptation, 2026. Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""Separate pose sampling, collection, solve and validation phases.

Adapted from Intrinsic sample_calibration_poses and collect_calibration_data.
All physical moves use the existing guarded MoveIt/CAN executor. No unplanned
pre-calibration moves, no previous camera TF, and no online outlier rejection.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from .visibility import BoardFraming, CameraKinematics
from .bootstrap_calibration import _clear_samples, sample_pose
from . import intrinsic_solver as solver


def pre_calibration_poses(initial, distance=.02, angle_degrees=12.):
    axes=np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1],
                   [.5,.5,.5],[-.5,-.5,-.5]],float)
    for axis in axes:
        p=initial.copy();p[:3,3]+=distance*axis
        p[:3,:3]=R.from_rotvec(np.deg2rad(angle_degrees)*axis/np.linalg.norm(axis)).as_matrix()@initial[:3,:3]
        yield p


def random_camera_pose(reference, board_center, rng, halfsize, tilt=15., roll=35.):
    """Intrinsic look-at construction followed by local XYZ random tilts."""
    p=reference.copy();p[:3,3]+=rng.uniform(-1,1,3)*halfsize
    z=board_center-p[:3,3];z/=np.linalg.norm(z)
    if z@reference[:3,2]<0:z=-z
    x=np.cross(reference[:3,1],z)
    if np.linalg.norm(x)<1e-8:raise ValueError('Degenerate look-at direction')
    x/=np.linalg.norm(x);y=np.cross(z,x)
    angles=np.deg2rad(rng.uniform(-1,1,3)*[tilt,tilt,roll])
    # Intrinsic multiplies Rx * Ry * Rz (intrinsic uppercase XYZ in SciPy).
    p[:3,:3]=np.column_stack([x,y,z])@R.from_euler('XYZ',angles).as_matrix()
    return p


class IntrinsicSession:
    def __init__(self, runner, root):
        self.r=runner;self.n=runner.node
        info=runner.camera_info
        if info is None or info.header.frame_id!=self.n.tracking_base_frame:
            raise ValueError('Detector and CameraInfo must use the same optical frame')
        # The frame check above guarantees detection is already expressed in
        # CameraInfo's optical frame. Shared observation/monitor code requires
        # this mapping even though no camera mount estimate exists yet.
        self.r.optical_tracking=np.eye(4)
        self.r.framing=BoardFraming(runner.board_spec,info)
        self.fk=CameraKinematics(root,self.n.robot_base_frame,self.n.robot_effector_frame,runner.names,np.eye(4))
        # Camera before the last joint (Piper on link5): fewer than six joints move it, so
        # arbitrary 6-D targets are unreachable and the nearest reachable pose is used.
        self.underactuated=sum(kind!='fixed' for _,_,kind,_ in getattr(self.fk,'chain',[(0,0,'revolute',0)]*6))<6
        self.initial=self.r.joint_positions().copy()
        self.views=[];self.validation=[];self.estimate=None;self.board=None
        self.attempts=0;self.rejections={}
        self.main_count=runner.min_training_samples
        self.total=8+self.main_count+runner.validation_views
        self.rng=np.random.default_rng()

    def pause(self):
        end=time.monotonic()+1.
        while time.monotonic()<end:
            self.r.check();self.r.stop_event.wait(min(.1,max(0,end-time.monotonic())))
        self.r.check()

    def ik(self, pose, nearest=False):
        current=self.r.joint_positions()
        def residual(q):
            self.r.check();actual=self.fk.camera(q)
            return np.r_[(actual[:3,3]-pose[:3,3])/.01,
                         R.from_matrix(pose[:3,:3].T@actual[:3,:3]).as_rotvec()/.05]
        fit=least_squares(residual,np.clip(current,self.r.lower+1e-8,self.r.upper-1e-8),
                          bounds=(self.r.lower,self.r.upper),max_nfev=100)
        e=residual(fit.x)
        if np.linalg.norm(e[:3])<=.2 and np.linalg.norm(e[3:])<=.2:return fit.x
        # Nearest reachable pose; callers re-check framing and duplicates on the actual pose.
        # Bounds: 15 mm, ~4 deg (scaled residuals).
        if self.underactuated and np.linalg.norm(e[:3])<=1.5 and np.linalg.norm(e[3:])<=1.4:return fit.x
        # Main/validation sampling: random 6-D targets (roll up to 35 deg) are mostly unreachable
        # with five camera joints; the caller accepts the reachable pose only if framed and new.
        if self.underactuated and nearest:return fit.x
        return None

    def move(self, target):
        self.r.check()
        self.r.planned_trajectory=None
        path=self.r.plan_joint_path(self.r.joint_positions(),target)
        if path is None:return False
        trajectory=self.r.planned_trajectory
        if trajectory is None:raise RuntimeError('A timed MoveIt trajectory is required; no motion sent')
        self.r.planned_trajectory=None
        self.r.move_planned(trajectory)
        self.r.settle(np.asarray(trajectory.points[-1].positions,float))
        return True

    def collect(self, target, phase):
        self.attempts+=1
        self.r.update('moving','Following a checked trajectory to the next sampled pose.',
                      phase=phase,attempts=self.attempts,search_mode='targeted')
        if not self.move(target):
            self.last_failure='planning'
            return False
        self.pause()
        self.r.capture_role='validation' if phase=='validation' else 'training'
        try:
            captured=self.r.capture()
        finally:
            self.r.capture_role='training'
        self.last_failure=None if captured else 'capture'
        if captured:
            collection=self.validation if phase=='validation' else self.views
            collection.append(sample_pose(self.n.robot_samples[-1]))
            self.r.update('capturing','Captured a stationary pose pair.',accepted=len(self.views),
                          initial_samples=min(len(self.views),self.initial_count if hasattr(self,'initial_count') else 8),
                          targeted_samples=max(0,len(self.views)-getattr(self,'initial_count',8)),
                          validation_views=len(self.validation),pose=len(self.views)+len(self.validation))
        if getattr(self.r,'intrinsic_return_to_base',True):
            if not self.move(self.initial):
                raise RuntimeError('No checked return path to the reference pose; robot stopped')
        return captured

    def sample_initial(self):
        """Replace failed initial views within a bounded, deterministic budget.

        Smaller rotations reduce tool occlusion and IK failures. Capture still
        uses the full detector/stationarity gates; no weak observation is saved
        merely to fill the quota. No camera extrinsic is assumed here.
        """
        reference=self.fk.camera(self.initial)
        candidates=0;without_sample=0
        stages=((6., .02), (4., .015), (3., .01))
        for angle,distance in stages:
            for axis,pose in enumerate(pre_calibration_poses(reference,distance,angle),1):
                self.r.check();candidates+=1
                reason=None
                target=self.ik(pose)
                if target is None:
                    reason='initial_ik'
                elif any(R.from_matrix(p[:3,:3].T@self.fk.camera(target)[:3,:3]).magnitude()<np.deg2rad(2)
                         and np.linalg.norm(p[:3,3]-self.fk.camera(target)[:3,3])<.01 for p in self.views):
                    reason='initial_duplicate'
                elif not self.collect(target,'bootstrap'):
                    reason='initial_'+self.last_failure
                if reason is not None:
                    self.rejections[reason]=self.rejections.get(reason,0)+1
                    without_sample+=1
                else:
                    without_sample=0
                message=(f'Initial candidate {candidates}/24, axis {axis}, {angle:g} deg: '
                         f'{reason or "captured"}; {len(self.views)}/8 accepted.')
                self.r.update('planning',message,initial_samples=len(self.views),
                              planner_rejections=dict(self.rejections),
                              attempts_without_sample=without_sample,
                              max_attempts_without_sample=24,
                              initial_candidates=candidates,initial_candidate_limit=24)
                if hasattr(self.n,'get_logger'):
                    self.n.get_logger().info(message)
                if len(self.views)>=8 and solver.geometry(self.views)['ready']:
                    return
        # Six remains the minimum; neither the diversity nor final quality
        # requirements are weakened when the replacement budget is exhausted.
        if len(self.views)<6 or not solver.geometry(self.views)['ready']:
            raise RuntimeError(f'Initial acquisition exhausted 24 candidates: {len(self.views)} accepted; '
                               f'need at least six diverse views. Rejections: {self.rejections}; no calibration saved')

    def fit(self, full=False):
        self.r.check()
        cal=self.n.get_calibration(full=full)
        if cal is None:raise RuntimeError('Joint camera/board solve failed; no calibration saved')
        report=self.n._last_reprojection
        if not report['geometry']['ready']:
            raise RuntimeError('Collected poses do not constrain all rotation axes; no calibration saved')
        return report

    def sample_main(self, count, phase):
        # Freeze the planning reference for the entire collection phase.
        reference=self.fk.camera(self.initial)@self.estimate
        center=(self.board@np.r_[self.r.framing.center,1])[:3]
        collected=0;tries=0;failures=0
        while collected<count and tries<count*20:
            self.r.check();tries+=1
            desired=random_camera_pose(reference,center,self.rng,np.array([.06,.06,.04]))
            if np.linalg.norm(desired[:3,3]-reference[:3,3])>self.r.max_camera_excursion:continue
            if not self.r.framing.contains(np.linalg.inv(desired)@self.board,margin=.13):continue
            target=self.ik(desired@np.linalg.inv(self.estimate),nearest=True)
            if target is None:continue
            actual=self.fk.camera(target)
            if self.underactuated:
                camera=actual@self.estimate
                if (np.linalg.norm(camera[:3,3]-reference[:3,3])>self.r.max_camera_excursion
                        or not self.r.framing.contains(np.linalg.inv(camera)@self.board,margin=.13)):continue
            # Avoid near-duplicates, while keeping every successfully captured pose.
            previous=self.views+self.validation
            if any(R.from_matrix(p[:3,:3].T@actual[:3,:3]).magnitude()<np.deg2rad(3)
                   and np.linalg.norm(p[:3,3]-actual[:3,3])<.02 for p in previous):continue
            if self.collect(target,phase):
                collected+=1;failures=0
            else:
                failures+=int(self.last_failure=='capture')
                if failures>=3:
                    raise RuntimeError('Three sampled poses failed capture; robot stopped, no calibration saved')
                self.rejections['path_or_capture']=self.rejections.get('path_or_capture',0)+1
            self.r.update('planning',f'Collected {collected}/{count} {phase} views.',planner_rejections=self.rejections)
        if collected<count:
            raise RuntimeError(f'Only {collected}/{count} {phase} views were collected; no calibration saved')

    def run(self):
        if self.n.solver_name!='intrinsic_pose':
            raise ValueError('Intrinsic acquisition requires solver:=intrinsic_pose')
        if not self.r.framing.contains(self.r.observed_board(recovering=True)):
            raise ValueError('Center the board with 10% image margin before starting; no motion sent')
        _clear_samples(self.n)
        self.r.update('planning','Collecting diverse initial views with replacement candidates (up to 24).',strategy='intrinsic_staged',
                      phase='bootstrap',target_samples=8+self.main_count,max_training_samples=8+self.main_count,
                      total=self.total,required_validation_views=self.r.validation_views,validation_views=0,accepted=0)
        self.sample_initial()
        self.initial_count=len(self.views)
        self.r.update('assessing','Solving the initial pose batch.',
                      target_samples=self.initial_count+self.main_count,
                      max_training_samples=self.initial_count+self.main_count,
                      total=self.initial_count+self.main_count+self.r.validation_views)
        if self.initial_count<6:raise RuntimeError('Fewer than six pre-calibration poses captured; no calibration saved')
        report=self.fit()
        m=report['pose_metrics']
        if m['position_max_m']>.01 or m['rotation_max_deg']>6:
            raise RuntimeError('Pre-calibration is too inconsistent for main-view planning; no calibration saved')
        self.estimate=solver.matrix(report['transform']);self.board=solver.matrix(report['board_in_base'])
        self.r.update('planning','Initial joint estimate ready; collecting the main dataset without refitting or discarding views.',phase='refinement')
        self.sample_main(self.main_count,'refinement')
        report=self.fit(full=True)
        quality=solver.evaluate(report,sigma_limit=self.r.max_position_sigma,min_samples=self.main_count)
        self.r.update('assessing',quality['summary'],validation=quality,
                      position_sigma_m=(report.get('uncertainty') or {}).get('worst_direction_sigma_m'))
        if not quality['passed']:raise RuntimeError(quality['summary']+'; no calibration saved')
        self.estimate=solver.matrix(report['transform']);self.board=solver.matrix(report['board_in_base'])
        self.r.update('planning','Training frozen; collecting independent validation views.',phase='validation')
        self.sample_main(self.r.validation_views,'validation')
        ids=self.n.validation_indices()
        val=solver.metrics(np.array([solver.matrix(self.n.robot_samples[i]) for i in ids]),
                           np.array([solver.matrix(self.n.tracking_samples[i]) for i in ids]),self.estimate,self.board)
        report['validation_views']=val
        quality=solver.evaluate(report,sigma_limit=self.r.max_position_sigma,min_samples=self.main_count,require_validation=True)
        self.r.update('assessing',quality['summary'],validation=quality)
        if not quality['passed']:raise RuntimeError(quality['summary']+'; no calibration saved')
