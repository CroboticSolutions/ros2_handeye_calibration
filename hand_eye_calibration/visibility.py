"""Board framing and joint-path checks independent of ROS transport."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class BoardFraming:
    def __init__(self, spec, info, margin=0.10):
        sx, sy, length = int(spec['squares_x']), int(spec['squares_y']), float(spec['square_length_m'])
        if sx < 2 or sy < 2 or not np.isfinite(length) or length <= 0:
            raise ValueError('Invalid board dimensions.')
        self.width, self.height = int(info.width), int(info.height)
        self.k = np.asarray(info.k, dtype=float).reshape(3, 3)
        self.d = np.asarray(info.d, dtype=float)
        if (self.width <= 0 or self.height <= 0 or not np.isfinite(self.k).all()
                or not np.isfinite(self.d).all() or min(self.k[0, 0], self.k[1, 1]) <= 0):
            raise ValueError('CameraInfo is missing valid intrinsics.')
        if info.distortion_model not in ('', 'plumb_bob', 'rational_polynomial'):
            raise ValueError('Unsupported camera distortion model.')
        self.margin = margin
        # Include the backing's 10 mm white border. Sample edges for distortion.
        x0, y0, x1, y1 = -.01, -.01, sx * length + .01, sy * length + .01
        edge = np.linspace(0, 1, 17)
        self.points = np.array([(x0+(x1-x0)*t, y, 0) for y in (y0, y1) for t in edge]
                               + [(x, y0+(y1-y0)*t, 0) for x in (x0, x1) for t in edge])
        self.center = np.array([sx*length/2, sy*length/2, 0])

    def pixels(self, observation):
        xyz = self.points @ observation[:3, :3].T + observation[:3, 3]
        if not np.isfinite(xyz).all() or np.min(xyz[:, 2]) <= .05:
            return None
        pixels, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), self.k, self.d)
        return pixels.reshape(-1, 2)

    def contains(self, observation, margin=None):
        p = self.pixels(observation)
        if p is None or not np.isfinite(p).all():
            return False
        m = self.margin if margin is None else margin
        size = np.array([self.width, self.height])
        return bool(np.all(p >= size*m) and np.all(p <= size*(1-m)))

    def corrected(self, camera, base_board):
        """Preserve orientation; center the board and back away only as needed."""
        pose = camera.copy()
        center = (base_board @ np.r_[self.center, 1])[:3]
        depth = float((np.linalg.inv(camera) @ np.r_[center, 1])[2])
        if depth <= .05:
            return None
        for scale in (1., 1.05, 1.1, 1.2, 1.3):
            pose[:3, 3] = center - pose[:3, :3] @ np.array([0., 0., depth*scale])
            if self.contains(np.linalg.inv(pose) @ base_board):
                return pose.copy()
        return None


class CameraKinematics:
    """URDF FK to the IK link, followed by the fixed camera extrinsic."""
    def __init__(self, root, base, tip, names, camera_ik):
        by_child = {j.find('child').attrib['link']: j for j in root.findall('joint')}
        self.chain = []
        while tip != base:
            if tip not in by_child:
                raise ValueError(f'No URDF chain from {base} to {tip}.')
            joint = by_child[tip]
            origin = joint.find('origin')
            origin = {} if origin is None else origin.attrib
            tf = np.eye(4)
            tf[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            tf[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            axis = joint.find('axis')
            axis = np.fromstring('1 0 0' if axis is None else axis.attrib.get('xyz', '1 0 0'), sep=' ')
            axis = axis / np.linalg.norm(axis)
            kind, name = joint.attrib['type'], joint.attrib['name']
            if kind not in ('fixed', 'revolute', 'continuous', 'prismatic') or (kind != 'fixed' and name not in names):
                raise ValueError(f'Unsupported or uncontrolled joint in camera chain: {name}.')
            self.chain.append((tf, axis, kind, name))
            tip = joint.find('parent').attrib['link']
        self.names = names
        self.ik_camera = np.linalg.inv(camera_ik)

    def camera(self, q):
        values = dict(zip(self.names, q))
        result = np.eye(4)
        for origin, axis, kind, name in reversed(self.chain):
            motion = np.eye(4)
            if kind in ('revolute', 'continuous'):
                motion[:3, :3] = Rotation.from_rotvec(axis*values[name]).as_matrix()
            elif kind == 'prismatic':
                motion[:3, 3] = axis*values[name]
            result = result @ origin @ motion
        return result @ self.ik_camera


def visible_joint_path(start, end, camera, framing, base_board, margin=None):
    # Controller supplies zero endpoint velocities, so its joint interpolation
    # follows this same line with a common smooth time scaling.
    steps = max(2, int(np.ceil(np.max(np.abs(end-start))/.015))+1)
    return all(framing.contains(np.linalg.inv(camera(start+(end-start)*t)) @ base_board, margin=margin)
               for t in np.linspace(0, 1, steps))
