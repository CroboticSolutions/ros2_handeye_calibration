"""Regression evidence for adaptive initialization and held-out acquisition."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from hand_eye_calibration.acquisition_quality import bootstrap_geometry, common_board


def pose(angles=(0,0,0), xyz=(0,0,0)):
    m=np.eye(4);m[:3,:3]=Rotation.from_euler('xyz',angles,degrees=True).as_matrix();m[:3,3]=xyz
    return m


def test_many_small_or_single_axis_rotations_do_not_initialize():
    assert not bootstrap_geometry([pose((i, i%3, 0)) for i in range(6)])['ready']
    assert not bootstrap_geometry([pose((i*3, 0, 0)) for i in range(20)])['ready']
    rich=[pose(v) for v in [(0,0,0),(25,0,0),(-20,0,0),(0,20,0),(0,-20,0),(0,0,20)]]
    assert bootstrap_geometry(rich)['ready']


def test_common_board_uses_all_views_and_physical_centre():
    robots=[pose() for _ in range(7)];tracking=[pose(xyz=(0,0,.5)) for _ in robots]
    tracking[-1]=pose(xyz=(.1,0,.5))
    board,fit=common_board(robots,tracking,pose(),np.array([.075,.1,0]))
    assert board[0,3]==pytest.approx(0.)
    assert fit['max_translation_m']==pytest.approx(.1)
    assert fit['reference_point']=='board_center'


def test_joint_fit_board_takes_precedence_over_latest_observation():
    report={'board_in_base':[.2,.1,.5,0,0,0,1]}
    board,_=common_board([pose()]*6,[pose(xyz=(0,0,.7))]*6,pose(),np.zeros(3),report)
    np.testing.assert_allclose(board[:3,3],[.2,.1,.5])


def test_weak_training_quality_requests_more_poses_then_stops_at_limit(monkeypatch):
    from test_bootstrap_calibration import session
    from unittest.mock import Mock
    s,r,n=session(monkeypatch)
    n.get_calibration=Mock(return_value=[0,0,0,0,0,0,1])
    n._last_reprojection={'uncertainty':{'worst_direction_sigma_m':.01},
                         'board_spread':{'position_rms_m':.001},
                         'leave_one_out':{'position_rms_m':.001,'reprojection_rms_px':1.}}
    n._compute_uncertainty=lambda cal:{'worst_direction_sigma_m':.01, 'n_bootstrap':40}
    s.estimate=np.eye(4);s.views=[pose()]*20
    n.training_count=lambda:len(s.views)
    assert not s.quality_ready()
    assert not s.training_complete and not s.validating
    s.views=[pose()]*30
    with pytest.raises(RuntimeError,match='sample|quality failed'):
        s.quality_ready()
    n.save_calibration_service_callback.assert_not_called()


def test_validation_cannot_update_frozen_fit_and_bad_holdout_fails(monkeypatch):
    from test_bootstrap_calibration import session
    from unittest.mock import Mock
    s,r,n=session(monkeypatch);s.training_complete=True
    s.estimate=pose();s.board=pose(xyz=(0,0,.5));s.validation=[pose()]*5
    s.validation_tracking=[pose(xyz=(0,0,.5))]*4+[pose(xyz=(.02,0,.5))]
    n.get_calibration=Mock(side_effect=AssertionError('Validation refit forbidden'))
    with pytest.raises(RuntimeError,match='Independent validation failed'):
        s.done()
    n.get_calibration.assert_not_called()
    s.validation_tracking[-1]=pose(xyz=(0,0,.5))
    assert s.done()
    n.get_calibration.assert_not_called()


def test_recorded_failed_run_gets_coverage_increasing_candidate(monkeypatch):
    """Replay FK ranking only; visibility/collision remain live checks, not simulated evidence."""
    import json
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from types import SimpleNamespace
    from test_bootstrap_calibration import session
    from hand_eye_calibration.bootstrap_calibration import sample_pose
    from hand_eye_calibration.visibility import CameraKinematics
    fixtures = Path(__file__).parent / 'fixtures'
    data = json.loads((fixtures / 'bootstrap_150005.json').read_text())['samples']
    s,r,n = session(monkeypatch)
    s.views = [sample_pose(sample['robot']) for sample in data]
    geometry = bootstrap_geometry(s.views)
    assert geometry['max_rotation_deg'] == pytest.approx(19.775725775, abs=1e-6)
    assert not geometry['ready']
    assert s.max_training == 30
    n.training_count = lambda: len(s.views)
    assert not s.quality_ready()  # old mixed configuration stopped here
    root = ET.fromstring((fixtures / 'piper_calibration_chain.urdf').read_text())
    s.fk = CameraKinematics(root, 'base_link', 'link6', r.names, np.eye(4))
    current = np.array([data[-1]['joints'][name] for name in r.names])
    initial = np.array([data[0]['joints'][name] for name in r.names])
    np.testing.assert_allclose(s.fk.camera(current), s.views[-1], atol=1e-6)
    pixels = np.array([[.4,.4],[.6,.6]])
    s.image_motion = SimpleNamespace(centered_delta=lambda *a:np.zeros(6),
                                    predict=lambda *a:pixels, confident=lambda *a:True)
    candidates = s.candidates(current, initial, pixels, set())
    assert candidates
    _, target, limit = candidates[0]
    assert np.max(np.abs(target-current)) <= limit + 1e-9
    extended = bootstrap_geometry(s.views + [s.fk.camera(target)])
    assert extended['ready'], extended
    r.move.assert_not_called()


def test_geometry_failure_message_does_not_claim_solver_failed(monkeypatch):
    from test_bootstrap_calibration import session
    s,r,n = session(monkeypatch)
    s.views = [pose((i*.4, 0, 0)) for i in range(30)]
    n.training_count = lambda:len(s.views)
    with pytest.raises(RuntimeError, match='Initial solve was not attempted'):
        s.quality_ready()
    n.save_calibration_service_callback.assert_not_called()
