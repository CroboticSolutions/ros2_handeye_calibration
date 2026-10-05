"""Re-solve a recorded tool-TCP dataset and separate robot error from TCP error.

  ros2 run hand_eye_calibration tcp_offline_solve ~/.ros/tool_tcp_calibration_runs/<run> [--joint-offsets]

Prints the pivot fit, orientation diversity, leave-one-out, held-out
(reorientation) touches and the acceptance verdict. With --joint-offsets it
also fits zero offsets of the middle joints together with the tip and the spike
point (FK from the recorded URDF and joint positions). If the residual drops a
lot with offsets, the robot's kinematic model, not the TCP, limits accuracy.
"""
import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

from . import tcp_quality
from .pivot_backend import PivotCalibrationBackend


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'dataset.json')
    with open(path) as f:
        data = json.load(f)
    if data.get('kind') != 'tool_tcp':
        raise ValueError('Not a tool-TCP dataset.')
    return data


def joint_offset_fit(kinematics, names, joints, prior_sigma_rad=math.radians(0.5), initial=None):
    """Fit tip t, spike p and offsets of joints 2..n-1 to the tip touches.

    Offsets of the first joint are absorbed by the spike point and of the last
    by the tip offset, so they are not estimated."""
    free = list(range(1, len(names) - 1))
    q = [np.array([j[n] for n in names], float) for j in joints]
    x0 = np.zeros(6 + len(free)) if initial is None else np.r_[initial, np.zeros(len(free))]

    def residual(x):
        t, p, d = x[:3], x[3:6], x[6:]
        out = []
        for qi in q:
            qq = qi.copy()
            qq[free] += d
            flange = kinematics.camera(qq)
            out.append((flange[:3, :3] @ t + flange[:3, 3] - p) / 0.001)
        out.append(d / prior_sigma_rad)
        return np.concatenate(out)

    result = least_squares(residual, x0, loss='huber', f_scale=1.0)
    r = residual(result.x)[:-len(free)].reshape(-1, 3) * 0.001
    per = np.linalg.norm(r, axis=1)
    return {'tcp_translation': result.x[:3].tolist(), 'fixed_point': result.x[3:6].tolist(),
            'offsets_deg': {names[j]: float(math.degrees(v)) for j, v in zip(free, result.x[6:])},
            'rms_residual_m': float(np.sqrt(np.mean(per ** 2))), 'max_residual_m': float(per.max())}


def solve(data, joint_offsets=False, limits=None):
    raw = data['raw_samples']
    tip = raw['tip']
    pivot = PivotCalibrationBackend.compute_pivot(tip)
    validation = PivotCalibrationBackend.validate_pivot(raw.get('validation') or [], pivot)
    spans = tcp_quality.rotation_spans_deg(tip)
    axis_mode = bool(raw.get('axis_alignment'))
    axis = None
    cad = None
    if axis_mode:
        axis = PivotCalibrationBackend.compute_axis_from_alignment(
            raw['axis_alignment'], data.get('spike_axis_base') or (0, 0, 1), pivot['tcp_translation'])
        cad = tcp_quality.cad_axis_deviation_deg(axis['axis_dir'], (limits or {}).get('cad_axis_angle_deg', 35.0))
    out = {'pivot': pivot, 'rotation_spans_deg': spans, 'validation': validation, 'axis': axis,
           'cad_axis_deviation_deg': cad,
           'acceptance': tcp_quality.evaluate(pivot, spans, validation, axis=axis, axis_mode=axis_mode,
                                              cad_deviation=cad, limits=limits)}
    if joint_offsets:
        joints = [m.get('joints') for m in raw.get('tip_capture_metadata') or []]
        names = raw.get('joint_names') or []
        if not data.get('robot_description') or not names or len(joints) != len(tip) or any(j is None for j in joints):
            raise ValueError('Dataset lacks URDF, joint names or joint positions for every tip touch.')
        from .visibility import CameraKinematics
        kin = CameraKinematics(ET.fromstring(data['robot_description']), data['robot_base_frame'],
                               data['robot_flange_frame'], names, np.eye(4))
        out['joint_offsets'] = joint_offset_fit(kin, names, joints,
                                                initial=np.r_[pivot['tcp_translation'], pivot['fixed_point']])
    return out


def _mm(v):
    return 'n/a' if v is None else f'{v * 1000:.2f} mm'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset')
    parser.add_argument('--joint-offsets', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    out = solve(load(args.dataset), args.joint_offsets)
    if args.json:
        from .calibration_dataset import plain
        json.dump(plain(out), sys.stdout, indent=1)
        print()
    else:
        p = out['pivot']
        print(f"tip touches: {len(p['per_sample_residuals_m'])}, TCP {np.round(p['tcp_translation'], 5).tolist()} m")
        print(f"fit RMS {_mm(p['rms_residual_m'])}, max {_mm(p['max_residual_m'])}, "
              f"leave-one-out max shift {_mm(p['max_loo_tcp_shift_m'])}")
        s = out['rotation_spans_deg']
        print(f"rotation spans along principal axes: {s[0]:.1f}° / {s[1]:.1f}° / {s[2]:.1f}°")
        if out['validation']:
            v = out['validation']
            print(f"reorientation touches ({v['sample_count']}): RMS {_mm(v['rms_residual_m'])}, max {_mm(v['max_residual_m'])}")
        if out['axis']:
            print(f"tool axis from {out['axis']['sample_count']} alignments, spread {out['axis']['alignment_spread_deg']:.2f}°, "
                  f"CAD deviation {out['cad_axis_deviation_deg']:.1f}°")
        if 'joint_offsets' in out:
            j = out['joint_offsets']
            print(f"with joint offsets: RMS {_mm(j['rms_residual_m'])}, TCP {np.round(j['tcp_translation'], 5).tolist()} m, "
                  + ', '.join(f'{k} {v:+.3f}°' for k, v in j['offsets_deg'].items()))
        print(('ACCEPTED: ' if out['acceptance']['passed'] else 'REJECTED: ') + out['acceptance']['summary'])
    return 0 if out['acceptance']['passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
