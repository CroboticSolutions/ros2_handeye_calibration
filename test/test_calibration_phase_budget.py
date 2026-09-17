"""Fifteen training samples, with no separate validation captures."""
import numpy as np
import pytest
from test_bootstrap_calibration import session


@pytest.mark.parametrize('sigma,passes',[(.001,True),(.012,False)])
def test_all_fifteen_samples_are_used_and_quality_is_required(monkeypatch,sigma,passes):
    s,r,n=session(monkeypatch)
    sample=[0.,0.,0.,0.,0.,0.,1.]
    n.robot_samples=[sample.copy() for _ in range(15)]
    n.tracking_samples=[sample.copy() for _ in range(15)]
    n.sample_metrics=[{} for _ in range(15)]
    s.views=[np.eye(4) for _ in range(15)]
    s.estimate=np.eye(4)
    n.get_calibration=lambda:sample.copy()
    n._compute_uncertainty=lambda _:dict(worst_direction_sigma_m=sigma,n_bootstrap=40)
    if passes:
        assert s.quality_ready()
    else:
        with pytest.raises(RuntimeError,match='Position uncertainty 12.00 mm'):
            s.quality_ready()
    s.capture()
    assert len(n.robot_samples)==len(n.tracking_samples)==len(n.sample_metrics)==15


def test_bootstrap_translation_alone_is_not_a_new_sample(monkeypatch):
    from scipy.spatial.transform import Rotation
    s,_,_=session(monkeypatch)
    s.views=[np.eye(4)]
    p=np.eye(4);p[0,3]=.2
    assert not s.distinct(p)
    p[:3,:3]=Rotation.from_rotvec([0,.06,0]).as_matrix()
    assert s.distinct(p)


def test_bootstrap_ranks_rotation_above_large_translation(monkeypatch):
    s,r,n=session(monkeypatch)
    s.views=[np.eye(4)]
    from scipy.spatial.transform import Rotation
    def robot(q):
        p=np.eye(4);p[:3,3]=q[:3]*20
        p[:3,:3]=Rotation.from_rotvec(q[3:]).as_matrix()
        return p
    s.fk.camera=robot
    candidates=s.candidates(np.zeros(6),np.zeros(6),np.array([[.4,.4],[.6,.6]]),set())
    assert candidates
    assert np.linalg.norm(candidates[0][1][3:])>0


def test_total_pose_cap_prevents_another_capture(monkeypatch):
    s,r,n=session(monkeypatch)
    n.robot_samples=[[0.,0.,0.,0.,0.,0.,1.] for _ in range(15)]
    r.capture=__import__('unittest.mock',fromlist=['Mock']).Mock()
    s.capture()
    r.capture.assert_not_called()
