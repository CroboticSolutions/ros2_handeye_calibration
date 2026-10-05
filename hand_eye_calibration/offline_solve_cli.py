"""Re-solve a recorded hand-eye dataset and compare solvers on the same data.

  ros2 run hand_eye_calibration handeye_offline_solve ~/.ros/hand_eye_calibration_runs/<run>
      [--intrinsics] [--joint-offsets] [--bootstrap 30] [--yaml out.yaml] [--json]

Prints closed-form (AX=XB), AX=XB+refinement and reprojection results side by
side with the held-out evidence (leave-one-out, validation views, board spread).
With --yaml the reprojection result is written in the same format as the
online node (hand_eye_calibration.yaml), so it can be previewed or applied.
"""
import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from . import acceptance
from . import calibration_dataset
from .calibration_backend import CalibrationBackend
from .reprojection_calibration import (Camera, Dataset, JointOffsetModel, calibrate, pose_matrix)


def _delta(a, b):
    A, B = pose_matrix(a), pose_matrix(b)
    return (float(np.linalg.norm(A[:3, 3] - B[:3, 3])),
            math.degrees(Rotation.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude()))


def solve(data, intrinsics=False, joint_offsets=False, bootstrap=30, limits=None):
    samples = data['samples']
    training = [s for s in samples if s.get('role', 'training') == 'training']
    validation = [s for s in samples if s.get('role') == 'validation']
    robot = [s['robot'] for s in training]
    tracking = [s['tracking'] for s in training]
    out = {'training_samples': len(training), 'validation_samples': len(validation)}

    algorithm, rot, tr = CalibrationBackend._select_algorithm(robot, tracking, list(range(len(robot))))
    out['closed_form'] = {'algorithm': algorithm, 'transform': CalibrationBackend._rot_tr_to_list(rot, tr)}
    detail = CalibrationBackend.compute_calibration_detailed(robot, tracking)
    out['axxb_refined'] = {'transform': detail['transform'], 'residuals': detail['residuals'],
                           'rejected': detail['rejected_indices'], 'algorithm_selection': detail['algorithm_selection']}

    camera = Camera.from_dict(data['camera'])
    ordered = training + validation
    dataset = Dataset(ordered, camera, data['calibration_type'])
    if len(dataset) != len(ordered):
        raise ValueError('Some samples have no raw corners; cannot run the reprojection solver.')
    model = None
    if joint_offsets:
        from .visibility import CameraKinematics
        if not data.get('robot_description') or not data.get('robot_joint_names'):
            raise ValueError('Dataset has no robot_description/joint names; joint offsets unavailable.')
        root = ET.fromstring(data['robot_description'])
        names = data['robot_joint_names']
        kin = CameraKinematics(root, data['frames']['robot_base_frame'], data['frames']['robot_effector_frame'],
                               names, np.eye(4))
        model = JointOffsetModel(kin, names)
    report = calibrate(dataset, pose_matrix(detail['transform']), estimate_intrinsics=intrinsics,
                       joint_model=model, validation_indices=list(range(len(training), len(ordered))),
                       bootstrap_samples=bootstrap, excluded_indices=detail['rejected_indices'],
                       compare_refinement_cv=True)
    report.pop('_solution')
    out['reprojection'] = report
    out['acceptance'] = acceptance.evaluate(report, len(training), limits)
    out['difference_reprojection_vs_axxb'] = dict(zip(('translation_m', 'rotation_deg'),
                                                      _delta(report['transform'], detail['transform'])))
    online = (data.get('online_result') or {}).get('transform')
    if online:
        t = online
        online = [t['tx'], t['ty'], t['tz'], t['qx'], t['qy'], t['qz'], t['qw']] if isinstance(t, dict) else t
        out['difference_reprojection_vs_online'] = dict(zip(('translation_m', 'rotation_deg'),
                                                            _delta(report['transform'], online)))
    return out


def _mm(v):
    return 'n/a' if v is None else f'{v * 1000:.2f} mm'


def print_report(out):
    r = out['reprojection']
    print(f"samples: {out['training_samples']} training, {out['validation_samples']} validation")
    print(f"closed form ({out['closed_form']['algorithm']}): {np.round(out['closed_form']['transform'], 5).tolist()}")
    ax = out['axxb_refined']
    print(f"AX=XB refined: {np.round(ax['transform'], 5).tolist()}  mean pair residual "
          f"{_mm(ax['residuals']['mean_translation_m'])} / {ax['residuals']['mean_rotation_deg']:.2f} deg")
    print(f"reprojection:  {np.round(r['transform'], 5).tolist()}  RMS {r['reprojection_rms_px']:.2f} px")
    d = out['difference_reprojection_vs_axxb']
    print(f"  reprojection vs AX=XB: {_mm(d['translation_m'])} / {d['rotation_deg']:.3f} deg")
    if 'difference_reprojection_vs_online' in out:
        d = out['difference_reprojection_vs_online']
        print(f"  reprojection vs online result: {_mm(d['translation_m'])} / {d['rotation_deg']:.3f} deg")
    print(f"board spread (static board seen from every pose): {_mm(r['board_spread']['position_rms_m'])} RMS, "
          f"{_mm(r['board_spread']['position_max_m'])} max")
    if r.get('leave_one_out'):
        loo = r['leave_one_out']
        print(f"leave-one-out: board position {_mm(loo['position_rms_m'])} RMS / {_mm(loo['position_max_m'])} max, "
              f"reprojection {loo['reprojection_rms_px']:.2f} px RMS / {loo['reprojection_max_px']:.2f} px max")
    if r.get('validation_views'):
        v = r['validation_views']
        print(f"validation views ({v['views']}): board position {_mm(v['position_rms_m'])} RMS / "
              f"{_mm(v['position_max_m'])} max, reprojection {v['reprojection_rms_px']:.2f} px")
    if r.get('uncertainty'):
        u = r['uncertainty']
        print(f"bootstrap sigma (worst direction): {_mm(u['worst_direction_sigma_m'])}")
    if r.get('intrinsics'):
        print(f"intrinsics: {r['intrinsics']}")
    if r.get('joint_offsets'):
        print('joint offsets [deg]: ' + ', '.join(f'{k}={math.degrees(v):+.3f}' for k, v in r['joint_offsets'].items()))
    if r.get('rejected_views'):
        print(f"rejected views: {r['rejected_views']}")
    print(('ACCEPTED: ' if out['acceptance']['passed'] else 'REJECTED: ') + out['acceptance']['summary'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset')
    parser.add_argument('--intrinsics', action='store_true', help='also refine fx, fy, cx, cy')
    parser.add_argument('--joint-offsets', action='store_true', help='also estimate joint zero offsets (eye-in-hand)')
    parser.add_argument('--bootstrap', type=int, default=30)
    parser.add_argument('--yaml', help='write the reprojection result as a calibration YAML')
    parser.add_argument('--json', action='store_true', help='print the full report as JSON')
    args = parser.parse_args(argv)
    data = calibration_dataset.load(args.dataset)
    out = solve(data, args.intrinsics, args.joint_offsets, args.bootstrap)
    if args.json:
        json.dump(calibration_dataset.plain(out), sys.stdout, indent=1)
        print()
    else:
        print_report(out)
    if args.yaml:
        t = out['reprojection']['transform']
        frames = data['frames']
        result = {
            'calibration_type': data['calibration_type'], **frames,
            'calibrated_child_frame': frames['tracking_base_frame'],
            'sample_count': len(data['samples']), 'training_sample_count': out['training_samples'],
            'transform': dict(zip(('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'), t)),
            'solver': 'reprojection (offline)', 'reprojection': out['reprojection'],
            'acceptance': out['acceptance'], 'dataset_path': args.dataset,
        }
        with open(args.yaml, 'w') as f:
            yaml.safe_dump(calibration_dataset.plain(result), f, sort_keys=False)
        print(f'wrote {args.yaml}')
    return 0 if out['acceptance']['passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
