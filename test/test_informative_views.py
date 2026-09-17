"""View geometry and numerical equivalence without commanding hardware."""
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from hand_eye_calibration.informative_views import informative_view, information_gain
from hand_eye_calibration.calibration_backend import CalibrationBackend
from hand_eye_calibration.node import DataCollector
from test_calibration_backend import _generate_eye_in_hand_dataset


def pose(angles=(0,0,0), translation=(0,0,0)):
    result=np.eye(4); result[:3,:3]=Rotation.from_euler('xyz',angles,degrees=True).as_matrix()
    result[:3,3]=translation
    return result


def test_parallel_translations_do_not_count_as_informative_views():
    assert not informative_view(pose(translation=(.1,0,0)),[pose()])
    assert not informative_view(pose((2,0,0),(.1,0,0)),[pose()])
    assert informative_view(pose((6,0,0)),[pose()])
    assert informative_view(pose((3,0,0),(.04,0,0)),[pose()])
    assert not informative_view(pose((6,0,0)),[pose(),pose((6,0,0))])


def test_rotation_about_new_axis_constrains_previously_unobservable_direction():
    previous=[pose(),pose((12,0,0)),pose((-12,0,0))]
    assert information_gain(pose((0,12,0)),previous)>information_gain(pose((24,0,0)),previous)


def test_batched_solver_residual_is_identical_to_scalar_pair_calculation(monkeypatch):
    from hand_eye_calibration import calibration_backend as module
    robot,tracking,rotation,translation=_generate_eye_in_hand_dataset(15,np.random.default_rng(47))
    rg,tg=CalibrationBackend._to_rot_tr_arrays(robot)
    rc,tc=CalibrationBackend._to_rot_tr_arrays(tracking)
    def check(residual,x0,**kwargs):
        for delta in (np.zeros(6),np.array([.02,-.03,.01,.003,-.002,.001])):
            x=x0+delta; rx=Rotation.from_rotvec(x[:3]).as_matrix(); expected=[]
            for a,b in CalibrationBackend._all_pairs(len(robot)):
                t,r=CalibrationBackend._pair_residual_raw(rg[a],tg[a],rg[b],tg[b],rc[a],tc[a],rc[b],tc[b],rx,x[3:])
                expected.extend(t/CalibrationBackend.TRANS_SCALE_M)
                expected.extend(Rotation.from_matrix(r).as_rotvec()/CalibrationBackend.ROT_SCALE_RAD)
            np.testing.assert_allclose(residual(x),expected,atol=1e-11)
        return SimpleNamespace(x=x0,success=True)
    monkeypatch.setattr(module,'least_squares',check)
    CalibrationBackend._refine_nonlinear(rotation,translation,robot,tracking,list(range(15)))


def test_uncertainty_cache_invalidates_on_measurement_change(monkeypatch):
    uncertainty={'translation_sigma_m':[.001]*3,'rotation_sigma_deg':[.1]*3,
                 'n_bootstrap':40,'worst_direction_sigma_m':.001,'guidance':'test'}
    compute=Mock(return_value=uncertainty)
    monkeypatch.setattr(CalibrationBackend,'bootstrap_uncertainty',compute)
    n=SimpleNamespace(bootstrap_samples=40,_last_calibration_detail={'algorithm_used':'Tsai'},
                      robot_samples=[[0]*7],tracking_samples=[[0]*7],get_logger=lambda:Mock())
    cal=[0,0,0,0,0,0,1]
    DataCollector._compute_uncertainty(n,cal)
    DataCollector._compute_uncertainty(n,cal)
    assert compute.call_count==1
    n.tracking_samples[0][0]=.001
    DataCollector._compute_uncertainty(n,cal)
    assert compute.call_count==2


@pytest.mark.parametrize('visible,collision_free',[(False,True),(True,False)])
def test_planner_does_not_choose_a_path_with_bad_visibility_or_collision(visible,collision_free,monkeypatch):
    from hand_eye_calibration import informative_views
    monkeypatch.setattr(informative_views,"ViewInformation",lambda *a:SimpleNamespace(gain=lambda _:1.))
    from hand_eye_calibration.informative_views import InformativeViews
    planner=InformativeViews.__new__(InformativeViews)
    camera=lambda q:pose((0,float(q[0])*180/np.pi,0))
    framing=SimpleNamespace(contains=lambda *a,**k:visible)
    runner=SimpleNamespace(check=Mock(),framing=framing,collision_free_path=Mock(return_value=collision_free))
    planner.s=SimpleNamespace(estimate=np.eye(4),r=runner,views=[pose()],fk=SimpleNamespace(camera=camera),MAX_STEP=.05)
    runner.plan_joint_path=lambda start,end:[start,end] if runner.collision_free_path(start,end) else None
    planner.route=[]
    planner.camera=camera;planner.goal=None;planner.targets=[np.array([.15])]
    assert planner.next_step(np.array([0.]),pose(translation=(0,0,1))) is None
    runner.collision_free_path.assert_called_once()


def test_board_marker_is_in_base_frame_at_board_centre():
    from hand_eye_calibration.automatic_calibration import AutomaticCalibration
    from visualization_msgs.msg import Marker
    runner=SimpleNamespace(node=SimpleNamespace(robot_base_frame='base_link'),
        framing=SimpleNamespace(center=np.array([.1,.06,0.])),
        board_spec={'squares_x':10,'squares_y':6,'square_length_m':.02})
    AutomaticCalibration.show_board_estimate(runner,pose(translation=(.3,.2,.5)))
    marker=runner.board_marker
    assert marker.header.frame_id=='base_link'
    assert marker.type==Marker.CUBE
    assert marker.pose.position.x==pytest.approx(.4)
    assert marker.pose.position.y==pytest.approx(.26)
    assert marker.pose.position.z==pytest.approx(.5)
    assert marker.scale.x==pytest.approx(.2)
    assert marker.scale.y==pytest.approx(.12)
