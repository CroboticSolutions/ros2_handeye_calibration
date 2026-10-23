# Copyright 2026 Intrinsic Innovation LLC
# Modifications: ROS/MoveIt adaptation, 2026. Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""Separate pose sampling, collection, solve and validation phases.

Adapted from Intrinsic sample_calibration_poses and collect_calibration_data.
All physical moves use the existing guarded MoveIt/CAN executor. No unplanned
pre-calibration moves, no previous camera TF, and no online outlier rejection.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from .visibility import BoardFraming, CameraKinematics
from .closed_form_views import AnalyticIKViews, ClosedFormViews, NumericIKViews, POOL_PER_VIEW, shortest_route
from .joint_space_views import JointSpaceViews
from .bootstrap_calibration import _clear_samples, sample_pose
from .motion_planning import SAMPLER_JOINT_MARGIN_RAD
from . import intrinsic_solver as solver

EXTRA_VIEW_BATCH = 2   # views per refit while position sigma is above its limit


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
        # arbitrary 6-D targets are unreachable; views come from sampled joints instead.
        self.underactuated=sum(kind!='fixed' for _,_,kind,_ in getattr(self.fk,'chain',[(0,0,'revolute',0)]*6))<6
        # Main/validation views from a wide pool: closed-form IK for a 6-joint camera chain
        # (closed_form_views), every ssik branch when it has no closed form (CRX), numeric
        # IK on the URDF chain without ssik, sampled joints + FK for a camera before the
        # last joint (joint_space_views).
        # None keeps the local random sampler (no URDF chain).
        self.closed_form=None
        if self.underactuated and hasattr(self.fk,'chain'):
            self.closed_form=JointSpaceViews(self.fk,runner.names,runner.lower,runner.upper)
        elif not self.underactuated:
            if getattr(runner,'robot_description',None):
                for views in (ClosedFormViews,AnalyticIKViews):
                    self.closed_form=views.create(
                        runner.robot_description,self.n.robot_base_frame,self.n.robot_effector_frame,
                        runner.names,runner.lower,runner.upper,self.fk.camera)
                    if self.closed_form is not None:
                        break
            if self.closed_form is None and hasattr(self.fk,'chain'):
                self.closed_form=NumericIKViews(self.fk.camera,runner.lower,runner.upper)
        self.pool_left={}
        self.initial=self.r.joint_positions().copy()
        self.views=[];self.validation=[];self.estimate=None;self.board=None
        self.attempts=0;self.rejections={}
        self.main_count=runner.min_training_samples
        # Views added after the main set while position sigma is above its limit.
        self.max_main_count=max(self.main_count,int(getattr(runner,'max_training_samples',self.main_count)))
        self.total=8+self.main_count+runner.validation_views
        self.rng=np.random.default_rng()

    def ik(self, pose, nearest=False):
        current=self.r.joint_positions()
        def residual(q):
            self.r.check();actual=self.fk.camera(q)
            return np.r_[(actual[:3,3]-pose[:3,3])/.01,
                         R.from_matrix(pose[:3,:3].T@actual[:3,:3]).as_rotvec()/.05]
        lower=self.r.lower+SAMPLER_JOINT_MARGIN_RAD;upper=self.r.upper-SAMPLER_JOINT_MARGIN_RAD
        fit=least_squares(residual,np.clip(current,lower+1e-8,upper-1e-8),
                          bounds=(lower,upper),max_nfev=100)
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
        # move_planned settles at the trajectory end and stamps the stationary
        # time that capture() requires; a second settle only added >= 0.8 s.
        self.r.move_planned(trajectory)
        return True

    def collect(self, target, phase):
        self.attempts+=1
        self.r.update('moving','Following a checked trajectory to the next sampled pose.',
                      phase=phase,attempts=self.attempts,search_mode='targeted')
        if not self.move(target):
            self.last_failure='planning'
            return False
        # No fixed extra wait: capture() takes only frames >= 50 ms after the
        # measured stop, checks stationary_window_s and the burst spread
        # (mount-settling test 30.9.: no repeatable damping after 1-3 s).
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
        # Intrinsic's dress-pack option; off by default: joint-space goals on the
        # branch nearest the reference keep cable state per pose, and it doubled motion.
        if getattr(self.r,'intrinsic_return_to_base',False):
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

    def duplicate(self, actual):
        return any(R.from_matrix(p[:3,:3].T@actual[:3,:3]).magnitude()<np.deg2rad(3)
                   and np.linalg.norm(p[:3,3]-actual[:3,3])<.02 for p in self.views+self.validation)

    def collected_or_failed(self, target, phase, failures):
        """(captured, consecutive capture failures); three capture failures stop the run."""
        if self.collect(target,phase):
            return True,0
        failures+=int(self.last_failure=='capture')
        if failures>=3:
            raise RuntimeError('Three sampled poses failed capture; robot stopped, no calibration saved')
        self.rejections['path_or_capture']=self.rejections.get('path_or_capture',0)+1
        return False,failures

    def sample_closed_form(self, count, phase, reference, center, reuse=False):
        """Greedy views from the closed-form pool; returns how many were captured.

        ``reuse`` continues the phase's remaining pool (views added for sigma)."""
        label=getattr(self.closed_form,'label','closed-form IK')
        pool=self.pool_left.get(phase) if reuse else None
        if not pool:
            self.r.update('planning',f'Sampling {phase} views around the board with {label}.',view_sampler=label)
            pool,sampled=self.closed_form.pool(self.board,center,self.estimate,self.r.framing,reference,
                                               self.r.max_camera_excursion,self.initial,self.rng,
                                               POOL_PER_VIEW*count,check=self.r.check)
            message=f'{label[0].upper()+label[1:]}: {len(pool)} reachable, framed {phase} views of {sampled} sampled.'
            self.r.update('planning',message,closed_form_pool=len(pool),closed_form_sampled=sampled)
            if hasattr(self.n,'get_logger'):
                self.n.get_logger().info(message)
        self.pool_left[phase]=pool
        collected=0;failures=0
        while collected<count and pool:
            # Pick the missing views for information (greedy, as if all succeed),
            # then visit them along the shortest joint route; failed views are
            # replaced in the next round from the remaining pool.
            rotations=[p[:3,:3] for p in self.views+self.validation]
            batch=[]
            while len(batch)<count-collected and pool:
                self.r.check()
                batch.append(pool.pop(ClosedFormViews.best(pool,rotations+[e[1][:3,:3] for e in batch])))
            for target,actual in shortest_route(self.r.joint_positions(),batch):
                self.r.check()
                if collected>=count:break
                if self.duplicate(actual):continue
                captured,failures=self.collected_or_failed(target,phase,failures)
                collected+=int(captured)
                self.r.update('planning',f'Collected {collected}/{count} {phase} views ({label}).',
                              planner_rejections=self.rejections)
        return collected

    def local_sampler_warning(self, reason):
        """Shown in the GUI: local views barely tilt the camera (weak hand-eye translation)."""
        warning=(f'{reason}; using local random views (+-6 cm, tilt +-15 deg, roll +-35 deg). '
                 'Expect a weakly observable camera translation (position sigma).')
        self.r.update('planning',warning,view_sampler='local random',view_sampler_warning=warning)
        if hasattr(self.n,'get_logger'):
            self.n.get_logger().warning(warning)

    def sample_main(self, count, phase, reuse=False):
        # Freeze the planning reference for the entire collection phase.
        reference=self.fk.camera(self.initial)@self.estimate
        center=(self.board@np.r_[self.r.framing.center,1])[:3]
        collected=0;tries=0;failures=0
        if self.closed_form is not None:
            collected=self.sample_closed_form(count,phase,reference,center,reuse)
            if collected<count:
                self.local_sampler_warning(f'View pool exhausted at {collected}/{count} {phase} views')
        else:
            self.local_sampler_warning('No wide view pool for this robot chain')
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
            if self.duplicate(actual):continue
            captured,failures=self.collected_or_failed(target,phase,failures)
            collected+=int(captured)
            self.r.update('planning',f'Collected {collected}/{count} {phase} views.',planner_rejections=self.rejections)
        if collected<count:
            raise RuntimeError(f'Only {collected}/{count} {phase} views were collected; no calibration saved')

    def add_views_for_sigma(self, report):
        """Add refinement views (same frozen estimate) while position sigma exceeds its limit.

        Pool views are picked for the weakest translation direction; at most
        ``auto_max_training_samples`` main views. Returns the latest full fit."""
        def sigma(report):
            return (report.get('uncertainty') or {}).get('worst_direction_sigma_m')
        added=0
        while True:
            value=sigma(report)
            if value is None or value<=self.r.max_position_sigma:
                return report
            room=self.max_main_count-self.main_count-added
            if room<=0:
                return report
            batch=min(EXTRA_VIEW_BATCH,room)
            self.r.update('planning',f'Position sigma {value*1000:.2f} mm > {self.r.max_position_sigma*1000:.2f} mm; '
                          f'adding {batch} views for the weakest direction '
                          f'({self.main_count+added}/{self.max_main_count} main views).',
                          position_sigma_m=value,
                          target_samples=self.initial_count+self.main_count+added+batch,
                          max_training_samples=self.initial_count+self.max_main_count,
                          total=self.initial_count+self.main_count+added+batch+self.r.validation_views)
            self.sample_main(batch,'refinement',reuse=True)
            added+=batch
            report=self.fit(full=True)

    def return_to_reference(self):
        """One planned return after the last view (no per-view returns by default)."""
        if np.max(np.abs(self.r.joint_positions()-self.initial))<.003:
            return
        self.r.update('moving','All views collected; returning to the reference pose.')
        if not self.move(self.initial):
            # The data is complete; a missing path must not discard the calibration.
            warning='No checked return path to the reference pose; the robot stays at the last view.'
            self.r.update('planning',warning,return_warning=warning)
            if hasattr(self.n,'get_logger'):
                self.n.get_logger().warning(warning)

    def run(self):
        if self.n.solver_name!='intrinsic_pose':
            raise ValueError('Intrinsic acquisition requires solver:=intrinsic_pose')
        if not self.r.framing.contains(self.r.observed_board(recovering=True)):
            raise ValueError('Center the board with 10% image margin before starting; no motion sent')
        _clear_samples(self.n)
        self.r.update('planning','Collecting diverse initial views with replacement candidates (up to 24).',strategy='intrinsic_staged',
                      phase='bootstrap',target_samples=8+self.main_count,max_training_samples=8+self.max_main_count,
                      total=self.total,required_validation_views=self.r.validation_views,validation_views=0,accepted=0,
                      view_sampler=getattr(self.closed_form,'label','local random'),view_sampler_warning=None)
        self.sample_initial()
        self.initial_count=len(self.views)
        self.r.update('assessing','Solving the initial pose batch.',
                      target_samples=self.initial_count+self.main_count,
                      max_training_samples=self.initial_count+self.max_main_count,
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
        report=self.add_views_for_sigma(report)
        quality=solver.evaluate(report,sigma_limit=self.r.max_position_sigma,min_samples=self.main_count)
        self.r.update('assessing',quality['summary'],validation=quality,
                      position_sigma_m=(report.get('uncertainty') or {}).get('worst_direction_sigma_m'))
        if not quality['passed']:raise RuntimeError(quality['summary']+'; no calibration saved')
        self.estimate=solver.matrix(report['transform']);self.board=solver.matrix(report['board_in_base'])
        self.r.update('planning','Training frozen; collecting independent validation views.',phase='validation')
        self.sample_main(self.r.validation_views,'validation')
        self.return_to_reference()
        ids=self.n.validation_indices()
        val=solver.metrics(np.array([solver.matrix(self.n.robot_samples[i]) for i in ids]),
                           np.array([solver.matrix(self.n.tracking_samples[i]) for i in ids]),self.estimate,self.board)
        report['validation_views']=val
        quality=solver.evaluate(report,sigma_limit=self.r.max_position_sigma,min_samples=self.main_count,require_validation=True)
        self.r.update('assessing',quality['summary'],validation=quality)
        if not quality['passed']:raise RuntimeError(quality['summary']+'; no calibration saved')
