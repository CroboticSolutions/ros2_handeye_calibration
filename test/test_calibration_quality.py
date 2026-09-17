from types import SimpleNamespace
from unittest.mock import Mock
import threading
import numpy as np
import pytest
from std_srvs.srv import Trigger
from hand_eye_calibration.calibration_quality import position_uncertainty_ok
from hand_eye_calibration.node import DataCollector
from hand_eye_calibration.automatic_calibration import Stopped
from test_automatic_calibration import synthetic_sequence


@pytest.mark.parametrize('sigma,count,ok', [(.00855,40,False),(.002,40,True),(.001,7,False),(float('nan'),40,False),(-.001,40,False)])
def test_uncertainty_gate(sigma,count,ok):
    assert position_uncertainty_ok({'worst_direction_sigma_m':sigma,'n_bootstrap':count},.002) == ok
    assert not position_uncertainty_ok(None,.002)


@pytest.mark.parametrize('real_geometry',[False,True])
def test_uncertainty_failure_does_not_extend_fifteen_pose_budget(monkeypatch,real_geometry):
    import json
    from pathlib import Path
    fixture=Path(__file__).parent/'fixtures'
    profile={'urdf':(fixture/'piper_calibration_chain.urdf').read_text(),
             'joints':json.loads((fixture/'piper_calibration_joints.json').read_text())} if real_geometry else None
    r,n,_ = synthetic_sequence(monkeypatch,robot_profile=profile)
    counts=[]
    def uncertainty(cal):
        count=len(n.robot_samples); counts.append(count)
        return {'worst_direction_sigma_m': .00855 if count<20 else .0018,
                'n_bootstrap':40,'worst_direction_axis':[0.,0.,1.]}
    n._compute_uncertainty=uncertainty
    r.run()
    assert r.status['state']=='failed', r.status
    assert counts==[15]
    assert len(r.session.views)==15
    assert len(n.robot_samples)==15
    assert r.status['validation'] is None
    n.save_calibration_service_callback.assert_not_called()


@pytest.mark.parametrize('missing',[False,True])
def test_unmet_uncertainty_stops_at_sample_limit_without_save(monkeypatch,missing):
    r,n,_=synthetic_sequence(monkeypatch)
    r.max_training_samples=20
    n._compute_uncertainty=lambda cal: None if missing else {'worst_direction_sigma_m':.00855,'n_bootstrap':40}
    r.run()
    assert r.status['state']=='failed',r.status
    assert 'Sample limit reached' in r.status['message']
    assert len(n.robot_samples)==15
    n.save_calibration_service_callback.assert_not_called()


def test_stop_during_uncertainty_assessment_does_not_move_or_save(monkeypatch):
    r,n,_=synthetic_sequence(monkeypatch)
    calls=[]
    def uncertainty(cal):
        calls.append(r.move.call_count)
        r.check=Mock(side_effect=Stopped('Stopped'))
        return {'worst_direction_sigma_m':.001,'n_bootstrap':40}
    n._compute_uncertainty=uncertainty
    r.run()
    assert r.status['state']=='stopped'
    assert calls==[r.move.call_count]
    n.save_calibration_service_callback.assert_not_called()


@pytest.mark.parametrize('uncertainty',[None,{'worst_direction_sigma_m':.00855,'n_bootstrap':40}])
def test_save_gate_preserves_existing_file(tmp_path,uncertainty):
    path=tmp_path/'calibration.yaml'; path.write_text('previous calibration')
    n=SimpleNamespace(automatic=SimpleNamespace(active=True,thread=threading.current_thread(),max_position_sigma=.002),
        get_calibration=lambda:[0,0,0,0,0,0,1],get_parameter=lambda _:SimpleNamespace(value=str(path)),
        _calibration_residuals=lambda _: {},_diversity_summary=lambda:{},_compute_uncertainty=lambda _:uncertainty)
    response=DataCollector.save_calibration_service_callback(n,Trigger.Request(),Trigger.Response())
    assert not response.success
    assert 'uncertainty' in response.message
    assert path.read_text()=='previous calibration'
    assert list(tmp_path.iterdir())==[path]


def test_ranking_rewards_rotation_that_constrains_weak_translation(monkeypatch):
    from test_bootstrap_calibration import session
    s,r,n=session(monkeypatch)
    s.views=[np.eye(4)]
    s.estimate=np.eye(4)
    pixels=np.array([[.4,.4],[.6,.6]])
    baseline=s.candidates(np.zeros(6),np.zeros(6),pixels,set())
    s.uncertainty={'worst_direction_axis':[0.,0.,1.]}
    ranked=s.candidates(np.zeros(6),np.zeros(6),pixels,set())
    scores={tuple(q):score for score,q,_ in baseline}
    improvements={tuple(q):score-scores[tuple(q)] for score,q,_ in ranked}
    about_x=np.zeros(6); about_x[3]=.01
    about_z=np.zeros(6); about_z[5]=.01
    assert improvements[tuple(about_x)]>0
    assert improvements[tuple(about_z)]==pytest.approx(0)
