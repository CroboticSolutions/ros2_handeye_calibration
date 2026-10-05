import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from hand_eye_calibration import intrinsic_solver as solver
from hand_eye_calibration.intrinsic_acquisition import pre_calibration_poses, random_camera_pose
from test_reprojection_calibration import make_samples, X_TRUE, B_TRUE


def solve_samples(samples):
    return solver.solve([s['robot'] for s in samples],[s['tracking'] for s in samples])


def test_shah_joint_fit_recovers_both_transforms():
    report=solve_samples(make_samples(n=24))
    np.testing.assert_allclose(solver.matrix(report['transform']),X_TRUE,atol=1e-8)
    np.testing.assert_allclose(solver.matrix(report['board_in_base']),B_TRUE,atol=1e-8)
    assert report['geometry']['ready'] and report['rejected_views']==[]


def test_corrupt_view_is_retained_and_fails_quality():
    samples=make_samples(n=24)
    samples[-1]['tracking'][0]+=.1
    report=solve_samples(samples)
    assert report['training_views']==24 and report['kept_views']==list(range(24))
    assert report['pose_metrics']['position_max_m']>.02
    report['uncertainty']={'worst_direction_sigma_m':.001}
    assert not solver.evaluate(report)['passed']


def test_single_axis_data_is_not_observable():
    poses=[]
    for angle in np.linspace(-1,1,24):
        m=np.eye(4);m[:3,:3]=Rotation.from_rotvec([0,0,angle]).as_matrix();poses.append(m)
    assert not solver.geometry(poses)['ready']


def test_systematic_initial_poses_excite_multiple_axes_without_a_camera_guess():
    initial=np.eye(4);initial[:3,3]=[.3,.1,.4]
    poses=list(pre_calibration_poses(initial))
    assert len(poses)==8 and solver.geometry(poses)['ready']
    assert max(np.linalg.norm(p[:3,3]-initial[:3,3]) for p in poses)<=.02000001
    assert all(np.degrees(Rotation.from_matrix(p[:3,:3]).magnitude())==pytest.approx(12) for p in poses)


def test_random_pose_looks_at_board_without_tilt_and_is_bounded():
    ref=np.eye(4);center=np.array([.1,.1,.5]);rng=np.random.default_rng(1)
    for _ in range(20):
        p=random_camera_pose(ref,center,rng,np.full(3,.05),tilt=0,roll=0)
        np.testing.assert_allclose(p[:3,2],(center-p[:3,3])/np.linalg.norm(center-p[:3,3]),atol=1e-10)
        assert np.max(np.abs(p[:3,3]))<=.05


def test_bootstrap_and_validation_use_same_pose_model():
    samples=make_samples(n=24,robot_noise_m=.0001)
    report=solve_samples(samples[:20])
    report['uncertainty']=solver.uncertainty([s['robot'] for s in samples[:20]],
                                            [s['tracking'] for s in samples[:20]],report,n=10)
    report['validation_views']=solver.metrics(np.array([solver.matrix(s['robot']) for s in samples[20:]]),
                                              np.array([solver.matrix(s['tracking']) for s in samples[20:]]),
                                              solver.matrix(report['transform']),solver.matrix(report['board_in_base']))
    assert solver.evaluate(report,require_validation=True)['passed']
    report['validation_views']['position_max_m']=.01
    assert not solver.evaluate(report,require_validation=True)['passed']


def test_staged_acquisition_collects_main_batch_without_online_refits(monkeypatch):
    from types import SimpleNamespace
    from test_automatic_calibration import synthetic_sequence
    from hand_eye_calibration import intrinsic_acquisition as acquisition
    r,n,_=synthetic_sequence(monkeypatch)
    n.solver_name='intrinsic_pose'
    n.validation_indices=lambda:[i for i,role in enumerate(n.sample_roles) if role=='validation']
    from hand_eye_calibration import bootstrap_calibration
    monkeypatch.setattr(acquisition,'CameraKinematics',bootstrap_calibration.CameraKinematics)
    r.names=n.get_parameter('auto_joint_names').value
    r.lower=np.full(6,-3.);r.upper=np.full(6,3.)
    r.max_camera_excursion=.3;r.max_position_sigma=.002;r.intrinsic_return_to_base=True
    counts=[]
    def calibrate(full=False):
        ids=n.training_indices();counts.append(len(ids))
        n._last_reprojection=solver.solve([n.robot_samples[i] for i in ids],[n.tracking_samples[i] for i in ids])
        if full:n._last_reprojection['uncertainty']={'worst_direction_sigma_m':.0001}
        return n._last_reprojection['transform']
    n.get_calibration=calibrate
    s=acquisition.IntrinsicSession(r,None)
    # The fixture's capture expects the current acquisition session.
    r.session=s
    s.rng=np.random.default_rng(1)
    s.pause=lambda:None
    s.view_planner=None  # legacy virtual camera fixture metadata
    def simulated_move(target):
        start=r.joint_positions()
        for fraction in np.linspace(0,1,int(np.ceil(np.max(np.abs(target-start))/.04))+1)[1:]:
            r.move(start+fraction*(target-start))
        return True
    s.move=simulated_move
    s.run()
    assert counts==[8,28]  # no solve on each new targeted view, none on held-out views
    assert n.training_count()==28
    assert len(n.validation_indices())==5
    assert n._last_reprojection['rejected_views']==[]
    assert r.status['validation']['passed']


def test_unplanned_pose_is_never_sent(monkeypatch):
    from unittest.mock import Mock
    from types import SimpleNamespace
    from hand_eye_calibration.intrinsic_acquisition import IntrinsicSession
    s=IntrinsicSession.__new__(IntrinsicSession)
    s.r=SimpleNamespace(check=lambda:None,joint_positions=lambda:np.zeros(6),
                        planned_trajectory='stale',plan_joint_path=lambda *a:None,move_planned=Mock())
    assert not s.move(np.ones(6))
    assert s.r.planned_trajectory is None
    s.r.move_planned.assert_not_called()


def test_initial_visibility_uses_real_runner_observation_path(monkeypatch):
    """Do not mock observed_board: it consumes session optical-frame setup."""
    from types import SimpleNamespace as NS
    from builtin_interfaces.msg import Time
    from hand_eye_calibration.automatic_calibration import AutomaticCalibration
    from hand_eye_calibration import intrinsic_acquisition as acquisition
    from hand_eye_calibration import bootstrap_calibration
    from test_automatic_calibration import synthetic_sequence
    r,n,_=synthetic_sequence(monkeypatch)
    monkeypatch.setattr(acquisition,'CameraKinematics',bootstrap_calibration.CameraKinematics)
    r.names=n.get_parameter('auto_joint_names').value
    board=r.observed_board(recovering=True)
    n.tracking_marker_frame='board'
    n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=0))
    n.tf_buffer=NS(lookup_transform=lambda *a:NS(header=NS(stamp=Time(sec=1))))
    r.fresh_board=lambda:board.copy()
    r.freshness=lambda seconds:seconds
    assert not hasattr(r,'optical_tracking')
    acquisition.IntrinsicSession(r,None)
    np.testing.assert_array_equal(r.optical_tracking,np.eye(4))
    actual=AutomaticCalibration.observed_board(r,recovering=True)
    np.testing.assert_allclose(actual,board)
    r.move.assert_not_called()


def test_optical_frame_mismatch_rejected_before_motion(monkeypatch):
    from hand_eye_calibration.intrinsic_acquisition import IntrinsicSession
    from test_automatic_calibration import synthetic_sequence
    r,n,_=synthetic_sequence(monkeypatch)
    r.camera_info.header.frame_id='unrelated_camera'
    with pytest.raises(ValueError,match='same optical frame'):
        IntrinsicSession(r,None)
    assert not hasattr(r,'optical_tracking')
    r.move.assert_not_called()


def initial_session(ik_failures=(), capture_failures=()):
    from types import SimpleNamespace as NS
    from hand_eye_calibration.intrinsic_acquisition import IntrinsicSession
    s=IntrinsicSession.__new__(IntrinsicSession)
    s.initial=np.zeros(6);s.views=[];s.rejections={};s.n=NS()
    s.fk=NS(camera=lambda q:np.eye(4) if q.shape==(6,) else q)
    s.r=NS(check=lambda:None,update=lambda *a,**kw:updates.append(kw))
    updates=[];calls={'ik':0,'capture':0}
    def ik(p):
        calls['ik']+=1
        return None if calls['ik'] in ik_failures else p
    def collect(p,phase):
        calls['capture']+=1
        if calls['capture'] in capture_failures:
            s.last_failure='capture';return False
        s.views.append(p);return True
    s.ik=ik;s.collect=collect
    return s,calls,updates


def test_initial_replaces_unreachable_and_occluded_views():
    s,calls,updates=initial_session(ik_failures=(2,7),capture_failures=(4,))
    s.sample_initial()
    assert len(s.views)==8 and calls['ik']>8
    assert s.rejections['initial_ik']==2
    assert s.rejections['initial_capture']==1
    assert solver.geometry(s.views)['ready']
    assert updates[-1]['attempts_without_sample']==0


def test_initial_exhaustion_is_bounded_and_reports_failures():
    s,calls,updates=initial_session(ik_failures=range(1,25))
    with pytest.raises(RuntimeError,match='24 candidates: 0 accepted'):
        s.sample_initial()
    assert calls=={'ik':24,'capture':0}
    assert s.rejections=={'initial_ik':24}
    assert updates[-1]['attempts_without_sample']==24


def test_initial_stop_is_not_swallowed():
    s,calls,_=initial_session()
    def stop():raise RuntimeError('Operator stop')
    s.r.check=stop
    with pytest.raises(RuntimeError,match='Operator stop'):s.sample_initial()
    assert calls['ik']==0
