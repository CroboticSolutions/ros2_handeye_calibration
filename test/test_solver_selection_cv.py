"""Selection and refinement must predict poses they did not fit."""
import copy
import numpy as np
from hand_eye_calibration.calibration_backend import CalibrationBackend as Backend
from hand_eye_calibration.reprojection_calibration import Camera, Dataset, calibrate, pose_matrix
from hand_eye_calibration.acceptance import evaluate
from test_reprojection_calibration import make_samples, K, X_TRUE


def test_algorithm_selection_really_omits_each_pose(monkeypatch):
    samples = make_samples(n=7, noise_px=0)
    robot, tracking = [s['robot'] for s in samples], [s['tracking'] for s in samples]
    original = Backend._fit
    calls = []
    def fit(sr, st, indices, method):
        calls.append((method, tuple(indices)))
        return original(sr, st, indices, method)
    monkeypatch.setattr(Backend, '_fit', fit)
    method, rot, tr, scores = Backend._select_algorithm(robot, tracking, list(range(7)), with_scores=True)
    for name in Backend.AVAILABLE_ALGORITHMS:
        folds = [idx for a, idx in calls if a == name and len(idx) == 6]
        assert len(folds) == 7
        assert {tuple(set(range(7)) - set(f)) for f in folds} == {(i,) for i in range(7)}
    assert scores[method]['folds'] == 7
    assert np.linalg.norm(tr - X_TRUE[:3, 3]) < 1e-6


def test_axxb_exclusion_survives_pixel_refinement():
    samples = make_samples(n=9)
    for f in samples[4]['frames']:
        f['image_points'] = (np.array(f['image_points']) + [180, -120]).tolist()
    report = calibrate(Dataset(samples, Camera(K, [])), X_TRUE,
                       excluded_indices=[4], bootstrap_samples=0, leave_one_out=False)
    assert report['axxb_rejected_views'] == [4]
    assert 4 in report['rejected_views'] and 4 not in report['kept_views']
    assert report['training_views'] == 8
    assert np.linalg.norm(pose_matrix(report['transform'])[:3, 3] - X_TRUE[:3, 3]) < .001


def test_refinement_fold_never_uses_held_out_pixels_or_global_exclusions():
    samples = make_samples(n=7, noise_px=.15)
    first = calibrate(Dataset(samples, Camera(K, [])), X_TRUE,
                      bootstrap_samples=0, compare_refinement_cv=True, excluded_indices=[0])
    changed = copy.deepcopy(samples)
    for f in changed[0]['frames']:
        f['image_points'] = (np.array(f['image_points']) + [20, 5]).tolist()
    second = calibrate(Dataset(changed, Camera(K, [])), X_TRUE,
                       bootstrap_samples=0, compare_refinement_cv=True, excluded_indices=[0])
    a = first['refinement_comparison']['folds'][0]
    b = second['refinement_comparison']['folds'][0]
    assert a['held_out'] == b['held_out'] == 0
    assert 0 not in a['fit_indices']
    assert np.allclose(a['axxb_transform'], b['axxb_transform'], atol=1e-12)
    assert np.allclose(a['refined_transform'], b['refined_transform'], atol=1e-12)
    assert b['reprojection']['reprojection_px'] > a['reprojection']['reprojection_px'] + 10
    assert second['leave_one_out']['views'] == 7  # includes globally excluded pose


def test_regression_blocks_otherwise_acceptable_result():
    report = {'uncertainty': {'worst_direction_sigma_m': .001},
              'board_spread': {'position_rms_m': .001},
              'leave_one_out': {'position_rms_m': .001, 'reprojection_rms_px': .5},
              'refinement_comparison': {'passed': False, 'reason': 'Refinement worsens held-out predictions'}}
    verdict = evaluate(report, 15)
    assert not verdict['passed']
    assert [c['name'] for c in verdict['checks'] if not c['passed']] == ['refinement_non_regression']


def test_failed_fold_cannot_make_algorithm_look_better(monkeypatch):
    samples = make_samples(n=7, noise_px=0)
    original = Backend._fit
    def fit(sr, st, indices, method):
        if method == 'Tsai-Lenz' and 2 not in indices:
            raise RuntimeError('degenerate training fold')
        return original(sr, st, indices, method)
    monkeypatch.setattr(Backend, '_fit', fit)
    method, _, _, scores = Backend._select_algorithm(
        [s['robot'] for s in samples], [s['tracking'] for s in samples], list(range(7)), with_scores=True)
    assert method != 'Tsai-Lenz'
    assert scores['Tsai-Lenz']['score'] is None
    assert 'degenerate' in scores['Tsai-Lenz']['error']
