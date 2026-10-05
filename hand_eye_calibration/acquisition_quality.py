"""Geometry and common-board estimates for acquisition, not final acceptance."""
import numpy as np
from scipy.spatial.transform import Rotation


def bootstrap_geometry(poses):
    """Require real multi-axis excitation; more near-duplicate poses do not suffice."""
    if len(poses) < 2:
        return {'ready': False, 'samples': len(poses), 'max_rotation_deg': 0., 'second_span_deg': 0.}
    vectors = np.array([Rotation.from_matrix(poses[0][:3, :3].T @ p[:3, :3]).as_rotvec() for p in poses])
    _, _, axes = np.linalg.svd(vectors - vectors.mean(axis=0), full_matrices=False)
    spans = np.degrees(np.ptp(vectors @ axes.T, axis=0))
    second_span = float(spans[1]) if len(spans) > 1 else 0.
    rotations = np.array([p[:3, :3] for p in poses])
    cosines = (np.einsum('aij,bij->ab', rotations, rotations) - 1.) / 2.
    maximum = float(np.degrees(np.arccos(np.clip(cosines, -1., 1.))).max())
    return {'ready': bool(len(poses) >= 6 and maximum >= 20. and second_span >= 10.), 'samples': len(poses),
            'max_rotation_deg': float(maximum), 'second_span_deg': second_span}


def common_board(robots, tracking, mount, centre, report=None):
    """Use the joint pixel fit when available; otherwise all-view robust PnP consensus.

    Both the residual and translation consensus refer to the same physical board centre.
    Never anchor the planner to a single endpoint observation.
    """
    boards = np.array([r @ mount @ t for r, t in zip(robots, tracking)])
    c = np.r_[centre, 1.]
    if report is not None and report.get('board_in_base') is not None:
        pose = report['board_in_base']
        board = np.eye(4)
        board[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        board[:3, 3] = pose[:3]
    else:
        board = np.eye(4)
        board[:3, :3] = Rotation.from_matrix(boards[:, :3, :3]).mean().as_matrix()
        board[:3, 3] = np.median((boards @ c)[:, :3], axis=0) - board[:3, :3] @ c[:3]
    positions = np.linalg.norm((boards @ c)[:, :3] - (board @ c)[:3], axis=1)
    angles = np.degrees((Rotation.from_matrix(board[:3, :3]).inv() *
                         Rotation.from_matrix(boards[:, :3, :3])).magnitude())
    return board, {'max_translation_m': float(positions.max()), 'max_rotation_deg': float(angles.max()),
                   'poses': len(robots), 'reference_point': 'board_center'}


def coverage_score(geometry):
    """Continuous progress toward both initialization targets, capped per target.

    Rewarding the *endpoint* avoids alternating small steps solely because their
    instantaneous joint direction points along an undersampled axis.
    """
    return (min(geometry['max_rotation_deg'] / 20., 1.)
            + min(geometry['second_span_deg'] / 10., 1.))
