"""Raw calibration datasets: everything needed to re-solve a run offline.

Layout of one run directory (<dataset_dir>/<UTC stamp>/):
  dataset.json      samples (robot pose, board pose, raw corners per frame,
                    joints, role), camera model, board, frames, latency, URDF
  calibration.yaml  the result as computed online (accepted or not)
  images/NN.png     one colour image per sample (optional)
"""
import json
import math
import os
from datetime import datetime, timezone

import numpy as np

FORMAT_VERSION = 1


def plain(value):
    """numpy/tuple-free structure for JSON and yaml.safe_dump."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build(node):
    samples = []
    for i in range(len(node.robot_samples)):
        samples.append({
            'index': i,
            'role': node.sample_roles[i],
            'robot': node.robot_samples[i],
            'tracking': node.tracking_samples[i],
            'joints': node.sample_joints[i],
            'frames': node.sample_frames[i],
            'metrics': node.sample_metrics[i],
        })
    automatic = getattr(node, 'automatic', None)
    return {
        'format_version': FORMAT_VERSION,
        'created': datetime.now(timezone.utc).isoformat(),
        'calibration_type': node.calibration_type,
        'frames': {
            'robot_base_frame': node.robot_base_frame,
            'robot_effector_frame': node.robot_effector_frame,
            'tracking_base_frame': node.tracking_base_frame,
            'tracking_marker_frame': node.tracking_marker_frame,
        },
        'camera': node._camera_model,
        'board': node._board_spec_seen,
        'camera_latency': node.latency.stats(),
        'robot_joint_names': list(getattr(automatic, 'names', []) or []),
        'robot_description': getattr(automatic, 'robot_description', None),
        'samples': samples,
    }


def write_run(root, node, result=None):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    run = os.path.join(root, stamp)
    suffix = 1
    while os.path.exists(run):
        run = os.path.join(root, f'{stamp}_{suffix}')
        suffix += 1
    os.makedirs(os.path.join(run, 'images'))
    data = build(node)
    for i, image in enumerate(node.sample_images):
        if image:
            name = f'images/{i:02d}.png'
            with open(os.path.join(run, name), 'wb') as f:
                f.write(image['png'])
            data['samples'][i]['image'] = {'path': name, 'stamp_ns': image['stamp_ns'],
                                           'matches_frame': image['matches_frame']}
    if result is not None:
        data['online_result'] = {k: result.get(k) for k in ('transform', 'solver', 'acceptance', 'reprojection')}
    temporary = os.path.join(run, 'dataset.json.tmp')
    with open(temporary, 'w') as f:
        json.dump(plain(data), f)
    os.replace(temporary, os.path.join(run, 'dataset.json'))
    return run


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'dataset.json')
    with open(path) as f:
        data = json.load(f)
    if data.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'Unsupported dataset format {data.get("format_version")!r}.')
    return data
