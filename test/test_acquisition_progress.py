"""Regression: bounded acquisition when motion produces no useful samples."""
import numpy as np
import pytest
from hand_eye_calibration.acquisition_progress import AcquisitionProgress
from hand_eye_calibration.bootstrap_calibration import BootstrapSession
from test_automatic_calibration import synthetic_sequence
from test_bootstrap_calibration import session


def test_local_stagnation_stops_before_ninth_attempt():
    progress=AcquisitionProgress(20)
    for _ in range(8):
        progress.observe(20)
        progress.before_step(False)
    with pytest.raises(RuntimeError,match='8 local attempts'):
        progress.before_step(False)
    assert progress.without_sample==8


def test_targeted_motion_cannot_continue_indefinitely_without_samples():
    progress=AcquisitionProgress(20)
    for _ in range(16):
        progress.before_step(True)
    with pytest.raises(RuntimeError,match='16 attempts'):
        progress.before_step(True)


def test_new_holdout_or_training_view_resets_stagnation():
    progress=AcquisitionProgress(20)
    for _ in range(8): progress.before_step(False)
    progress.observe(21)
    progress.before_step(False)
    assert progress.local_without_sample==progress.without_sample==1


def test_local_search_penalizes_revisits_but_allows_transit(monkeypatch):
    s,r,n=session(monkeypatch)
    current=np.zeros(6);current[5]=.01
    s.visited_joints=[np.zeros(6)]
    options=s.candidates(current,np.zeros(6),np.array([[.4,.4],[.6,.6]]),set())
    assert options
    assert any(np.max(np.abs(q))<.008 for _,q,_ in options)
    assert np.max(np.abs(options[0][1]))>=.008


def test_plateau_before_pose_budget_stops_and_does_not_save(monkeypatch):
    r,n,_=synthetic_sequence(monkeypatch)
    capture=BootstrapSession.capture
    at_plateau=[]
    def capped(self):
        if len(n.robot_samples)>=8:
            if not at_plateau: at_plateau.append(r.move.call_count)
            return
        capture(self)
        if len(n.robot_samples)>=8 and not at_plateau:
            at_plateau.append(r.move.call_count)
    monkeypatch.setattr(BootstrapSession,'capture',capped)
    n._compute_uncertainty=lambda _: {'worst_direction_sigma_m':.01227,'n_bootstrap':40}
    r.run()
    assert at_plateau, r.status
    assert r.status['state']=='failed'
    assert r.move.call_count-at_plateau[0]<=48
    assert any(word in r.status['message'] for word in ('stalled','exhausted','unvisited'))
    n.save_calibration_service_callback.assert_not_called()


def test_checked_transit_can_cross_known_views_but_is_still_bounded():
    progress=AcquisitionProgress(20)
    progress.before_step(True)
    for _ in range(47):progress.before_step(True,advancing=True)
    with pytest.raises(RuntimeError,match='48 attempts'):
        progress.before_step(True,advancing=True)


def test_nonadvancing_route_cannot_bypass_stagnation_limit():
    progress=AcquisitionProgress(20)
    for _ in range(16):progress.before_step(True,advancing=True)
    with pytest.raises(RuntimeError,match='16 attempts'):
        progress.before_step(True,advancing=False)
