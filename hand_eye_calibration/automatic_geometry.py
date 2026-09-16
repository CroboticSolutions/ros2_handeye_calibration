"""Camera orbits around the observed target; independent of ROS and world pose."""
import numpy as np
from scipy.spatial.transform import Rotation


def camera_targets(base_camera, camera_board):
    """Keep the initial target bearing while exciting all three rotation axes."""
    bearing = camera_board[:3, 3]
    if not np.isfinite(bearing).all() or bearing[2] <= 0.08:
        raise ValueError('Board must be in front of the camera, at least 8 cm away.')
    pivot = (base_camera @ camera_board)[:3, 3]
    offsets = [(0, 0, 0)]
    for angle in (10, 18):
        offsets += [(angle, 0, 0), (-angle, 0, 0), (0, angle, 0), (0, -angle, 0),
                    (angle, angle, 0), (-angle, -angle, 0),
                    (angle, -angle, 0), (-angle, angle, 0)]
    offsets += [(0, 0, 18), (0, 0, -18), (10, 0, 15), (-10, 0, -15)]
    # Last three are reserved for independent validation.
    offsets += [(7, -13, 8), (-13, 7, -8), (12, 9, 12)]
    result = []
    for i, angles in enumerate(offsets):
        pose = base_camera.copy()
        pose[:3, :3] = base_camera[:3, :3] @ Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
        scale = 1.0 if i == 0 else (1.0, 0.95, 1.05)[i % 3]
        pose[:3, 3] = pivot - pose[:3, :3] @ (bearing * scale)
        result.append(pose)
    return result


def validation_error(robot_samples, tracking_samples, transform, holdouts):
    """Compare held-out board poses with the training board pose in base frame."""
    train = [r @ transform @ t for r, t in zip(robot_samples, tracking_samples)]
    centre = np.median([m[:3, 3] for m in train], axis=0)
    orientation = Rotation.from_matrix(np.array([m[:3, :3] for m in train])).mean()
    positions, angles = [], []
    for r, t in holdouts:
        actual = r @ transform @ t
        positions.append(float(np.linalg.norm(actual[:3, 3] - centre)))
        angles.append(float((orientation.inv() * Rotation.from_matrix(actual[:3, :3])).magnitude() * 180 / np.pi))
    return {'max_translation_m': max(positions), 'max_rotation_deg': max(angles), 'poses': len(holdouts)}


def pose_variants(initial, target):
    """Try smaller orbits and wrist-only rotations near workspace boundaries."""
    delta = Rotation.from_matrix(initial[:3, :3].T @ target[:3, :3]).as_rotvec()
    for scale, orbit in ((1.0, 1.0), (0.6, 1.0), (0.35, 1.0), (0.2, 1.0),
                         (1.0, 0.0), (0.6, 0.0), (0.35, 0.0), (0.2, 0.0)):
        pose = initial.copy()
        pose[:3, :3] = initial[:3, :3] @ Rotation.from_rotvec(delta * scale).as_matrix()
        pose[:3, 3] += orbit * scale * (target[:3, 3] - initial[:3, 3])
        yield pose


def distinct_view(candidate, previous):
    for pose in previous:
        angle = Rotation.from_matrix(pose[:3, :3].T @ candidate[:3, :3]).magnitude()
        if angle < np.deg2rad(2.5) and np.linalg.norm(pose[:3, 3] - candidate[:3, 3]) < 0.01:
            return False
    return True


def has_rotation_diversity(poses):
    if len(poses) < 3:
        return False
    rotations = np.array([Rotation.from_matrix(poses[0][:3, :3].T @ p[:3, :3]).as_rotvec() for p in poses[1:]])
    singular = np.linalg.svd(rotations, compute_uv=False)
    return bool(singular[1] > 0.08)
