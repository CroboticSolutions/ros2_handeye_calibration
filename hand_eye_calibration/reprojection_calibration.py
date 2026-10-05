"""Hand-eye calibration by joint minimisation of ChArUco corner reprojection error.

The closed-form AX=XB solvers (calibration_backend.py) fit rigid motions that
were themselves estimated per image with solvePnP, and weight translation
against rotation with hand-picked scales. The final estimate here is instead
the one that best explains what the camera actually measured: the pixel
positions of every board corner in every view.

Unknowns (Koide & Menegatti 2019; ROS-Industrial industrial_calibration):
  X      robot_effector -> camera   (eye-in-hand)   / robot_base -> camera (eye-on-base)
  B      robot_base     -> board    (eye-in-hand)   / robot_effector -> board (eye-on-base)
  opt.   fx, fy, cx, cy             (distortion stays at the CameraInfo values)
  opt.   joint zero offsets         (eye-in-hand, from the URDF chain; regularised)

Model, with A_i the stored robot sample (T_base_effector for eye-in-hand,
T_effector_base for eye-on-base, i.e. OpenCV's gripper2base convention):
  T_camera_board,i = X^-1 * A_i^-1 * B
Observed corners are undistorted once with the CameraInfo distortion, so the
projection itself is a vectorised pinhole model.

Uncertainty is estimated by resampling poses (the independent unit). Per-corner
residuals inside one view share that view's robot/kinematic error, so the
analytic (J^T J)^-1 covariance would be far too optimistic.
"""
import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


MIN_SAMPLES = 4
MIN_CORNERS = 6


def pose_matrix(sample):
    """[tx, ty, tz, qx, qy, qz, qw] -> 4x4."""
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(np.asarray(sample[3:7], dtype=float)).as_matrix()
    m[:3, 3] = np.asarray(sample[:3], dtype=float)
    return m


def pose_list(matrix):
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return [float(v) for v in matrix[:3, 3]] + [float(v) for v in q]


def _vec_to_pose(x):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    m[:3, 3] = x[3:6]
    return m


def _pose_to_vec(m):
    return np.r_[Rotation.from_matrix(m[:3, :3]).as_rotvec(), m[:3, 3]]


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
def merge_frames(frames, min_fraction=0.5):
    """Robustly merge the burst frames of one stationary pose.

    Each corner id seen in at least `min_fraction` of the frames is kept at the
    per-coordinate median of its image positions. Returns (ids, image, object).
    """
    if not frames:
        return None
    seen = {}
    for frame in frames:
        for cid, uv, xyz in zip(frame['ids'], frame['image_points'], frame['object_points']):
            entry = seen.setdefault(int(cid), [[], xyz])
            entry[0].append(uv)
    need = max(1, int(math.ceil(min_fraction * len(frames))))
    ids, image, obj = [], [], []
    for cid in sorted(seen):
        uvs, xyz = seen[cid]
        if len(uvs) < need:
            continue
        ids.append(cid)
        image.append(np.median(np.asarray(uvs, dtype=float), axis=0))
        obj.append(np.asarray(xyz, dtype=float))
    if len(ids) < MIN_CORNERS:
        return None
    return np.asarray(ids), np.asarray(image, dtype=float), np.asarray(obj, dtype=float)


class Camera:
    def __init__(self, k, d, width=0, height=0, distortion_model='plumb_bob'):
        self.k = np.asarray(k, dtype=float).reshape(3, 3)
        self.d = np.asarray(d if d is not None else [], dtype=float).reshape(-1)
        self.width, self.height = int(width or 0), int(height or 0)
        self.distortion_model = distortion_model or 'plumb_bob'
        if not np.isfinite(self.k).all() or min(self.k[0, 0], self.k[1, 1]) <= 0:
            raise ValueError('Camera intrinsics are invalid.')

    @classmethod
    def from_dict(cls, data):
        return cls(data['k'], data.get('d'), data.get('width', 0), data.get('height', 0),
                   data.get('distortion_model', 'plumb_bob'))

    def to_dict(self):
        return {'k': [float(v) for v in self.k.reshape(-1)], 'd': [float(v) for v in self.d],
                'width': self.width, 'height': self.height, 'distortion_model': self.distortion_model}

    def undistort(self, pixels):
        """Distorted pixels -> ideal pinhole pixels with the same K."""
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 1, 2)
        if self.d.size == 0 or not np.any(self.d):
            return pixels.reshape(-1, 2)
        return cv2.undistortPoints(pixels, self.k, self.d, P=self.k).reshape(-1, 2)


class Dataset:
    """Per-pose merged corners plus robot poses, independent of ROS."""

    def __init__(self, samples, camera, calibration_type='eye-in-hand'):
        if calibration_type not in ('eye-in-hand', 'eye-on-base'):
            raise ValueError(f'Invalid calibration_type {calibration_type!r}.')
        self.calibration_type = calibration_type
        self.camera = camera
        self.robot = []        # A_i as 4x4
        self.joints = []       # dict or None
        self.ids, self.image, self.object = [], [], []
        self.tracking = []     # PnP board pose in camera (T_camera_board), 4x4
        self.source_index = []  # index into the caller's sample list
        for index, sample in enumerate(samples):
            merged = merge_frames(sample.get('frames') or [])
            if merged is None:
                continue
            ids, image, obj = merged
            self.source_index.append(index)
            self.robot.append(pose_matrix(sample['robot']))
            self.joints.append(sample.get('joints'))
            self.ids.append(ids)
            self.image.append(camera.undistort(image))
            self.object.append(obj)
            self.tracking.append(pose_matrix(sample['tracking']) if sample.get('tracking') is not None
                                 else self._pnp(obj, image))

    def _pnp(self, obj, image):
        ok, rvec, tvec = cv2.solvePnP(obj, image, self.camera.k, self.camera.d)
        if not ok:
            raise ValueError('solvePnP failed on a stored view.')
        m = np.eye(4)
        m[:3, :3] = Rotation.from_rotvec(rvec.reshape(3)).as_matrix()
        m[:3, 3] = tvec.reshape(3)
        return m

    def __len__(self):
        return len(self.robot)


# ---------------------------------------------------------------------------
# Joint offsets (eye-in-hand only)
# ---------------------------------------------------------------------------
class JointOffsetModel:
    """FK with per-joint zero offsets. The first and last joint are left out:
    an offset of the base joint is absorbed by the board pose and one of the
    last joint by the camera mount, so they are not observable here."""

    def __init__(self, kinematics, names, prior_sigma_rad=math.radians(0.5)):
        self.kinematics = kinematics
        self.names = list(names)
        self.free = list(range(1, len(self.names) - 1))
        self.prior_sigma = float(prior_sigma_rad)

    def robot(self, joints, offsets):
        q = np.array([joints[n] for n in self.names], dtype=float)
        q[self.free] += offsets
        return self.kinematics.camera(q)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def _initial_board(dataset, X, indices):
    boards = [dataset.robot[i] @ X @ dataset.tracking[i] for i in indices]
    board = np.eye(4)
    board[:3, 3] = np.median([b[:3, 3] for b in boards], axis=0)
    board[:3, :3] = Rotation.from_matrix(np.array([b[:3, :3] for b in boards])).mean().as_matrix()
    return board


class Solution:
    def __init__(self, X, board, k, offsets, indices, rms_px, per_view_rms_px, cost, converged, nfev):
        self.X, self.board, self.k = X, board, k
        self.offsets = offsets
        self.indices = list(indices)
        self.rms_px = rms_px
        self.per_view_rms_px = per_view_rms_px
        self.cost, self.converged, self.nfev = cost, converged, nfev

    @property
    def transform(self):
        return pose_list(self.X)


class ReprojectionSolver:
    def __init__(self, dataset, estimate_intrinsics=False, joint_model=None, loss_scale_px=1.0):
        if joint_model is not None and dataset.calibration_type != 'eye-in-hand':
            raise ValueError('Joint offsets are only supported for eye-in-hand.')
        if joint_model is not None and any(j is None for j in dataset.joints):
            raise ValueError('Joint offsets need joint positions for every sample.')
        self.d = dataset
        self.estimate_intrinsics = bool(estimate_intrinsics)
        self.joint_model = joint_model
        self.loss_scale_px = float(loss_scale_px)

    # -- parameter vector: X(6) B(6) [fx fy cx cy] [offsets] --------------
    def _unpack(self, x):
        X, B = _vec_to_pose(x[:6]), _vec_to_pose(x[6:12])
        pos = 12
        k = self.d.camera.k.copy()
        if self.estimate_intrinsics:
            k[0, 0], k[1, 1], k[0, 2], k[1, 2] = x[pos:pos + 4]
            pos += 4
        offsets = x[pos:] if self.joint_model is not None else None
        return X, B, k, offsets

    def _robot(self, i, offsets):
        if self.joint_model is None:
            return self.d.robot[i]
        return self.joint_model.robot(self.d.joints[i], offsets)

    def _stack(self, indices):
        """Concatenated corners of `indices`, cached per index tuple."""
        key = tuple(indices)
        cache = getattr(self, '_stack_cache', None)
        if cache is None or cache[0] != key:
            obj = np.vstack([self.d.object[i] for i in indices])
            img = np.vstack([self.d.image[i] for i in indices])
            view = np.concatenate([np.full(len(self.d.ids[i]), n) for n, i in enumerate(indices)])
            inv_robot = np.array([np.linalg.inv(self.d.robot[i]) for i in indices])
            cache = (key, obj, img, view, inv_robot)
            self._stack_cache = cache
        return cache[1:]

    def _project_all(self, indices, X, B, k, offsets):
        obj, img, view, inv_robot = self._stack(indices)
        if self.joint_model is not None:
            inv_robot = np.array([np.linalg.inv(self._robot(i, offsets)) for i in indices])
        T = np.linalg.inv(X)[None] @ inv_robot @ B[None]
        R, t = T[view, :3, :3], T[view, :3, 3]
        p = np.einsum('nij,nj->ni', R, obj) + t
        z = p[:, 2]
        bad = z <= 1e-6
        z = np.where(bad, 1.0, z)
        uv = np.column_stack((k[0, 0] * p[:, 0] / z + k[0, 2], k[1, 1] * p[:, 1] / z + k[1, 2]))
        err = uv - img
        err[bad] = 1e3
        return err, view

    def _project(self, i, X, B, k, offsets):
        err, _ = self._project_all([i], X, B, k, offsets)
        return None if np.any(np.abs(err) >= 1e3) else err + self.d.image[i]

    def _residual(self, x, indices):
        X, B, k, offsets = self._unpack(x)
        err, _ = self._project_all(indices, X, B, k, offsets)
        parts = [err.ravel()]
        if self.joint_model is not None:
            # Weak prior: pulls unobservable offset combinations to zero. The
            # weight equals roughly one view's worth of corners.
            weight = math.sqrt(max(1, np.median([len(self.d.ids[i]) for i in indices])))
            parts.append(weight * offsets / self.joint_model.prior_sigma)
        return np.concatenate(parts)

    def per_view_rms(self, solution_or_params, indices):
        if isinstance(solution_or_params, Solution):
            X, B, k, offsets = solution_or_params.X, solution_or_params.board, solution_or_params.k, solution_or_params.offsets
        else:
            X, B, k, offsets = solution_or_params
        indices = list(indices)
        err, view = self._project_all(indices, X, B, k, offsets)
        sq = np.sum(err ** 2, axis=1)
        return [float(np.sqrt(np.mean(sq[view == n]))) for n in range(len(indices))]

    def solve(self, initial_X, indices=None, initial_board=None, initial_offsets=None):
        indices = list(range(len(self.d))) if indices is None else list(indices)
        if len(indices) < MIN_SAMPLES:
            raise ValueError(f'Need at least {MIN_SAMPLES} views, got {len(indices)}.')
        board = _initial_board(self.d, initial_X, indices) if initial_board is None else initial_board
        x0 = [_pose_to_vec(initial_X), _pose_to_vec(board)]
        k = self.d.camera.k
        if self.estimate_intrinsics:
            x0.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
        if self.joint_model is not None:
            x0.append(np.zeros(len(self.joint_model.free)) if initial_offsets is None else initial_offsets)
        x0 = np.concatenate(x0)
        result = least_squares(self._residual, x0, args=(indices,), method='trf', loss='huber',
                               f_scale=self.loss_scale_px, x_scale='jac', max_nfev=200)
        X, B, k, offsets = self._unpack(result.x)
        per_view = self.per_view_rms((X, B, k, offsets), indices)
        total = sum(len(self.d.ids[i]) for i in indices)
        rms = math.sqrt(sum(r ** 2 * len(self.d.ids[i]) for r, i in zip(per_view, indices)) / total)
        return Solution(X, B, k, offsets, indices, rms, per_view, float(result.cost),
                        bool(result.success), int(result.nfev))

    def solve_robust(self, initial_X, indices=None, max_reject_fraction=0.2, mad_k=3.5, floor_px=1.0):
        """Solve, then drop whole views whose reprojection RMS is an outlier."""
        indices = list(range(len(self.d))) if indices is None else list(indices)
        solution = self.solve(initial_X, indices)
        rejected = []
        max_reject = int(len(indices) * max_reject_fraction)
        while len(rejected) < max_reject and len(solution.indices) > MIN_SAMPLES + 1:
            rms = np.asarray(solution.per_view_rms_px)
            med = float(np.median(rms))
            mad = float(np.median(np.abs(rms - med))) + 1e-9
            worst = int(np.argmax(rms))
            if rms[worst] <= max(floor_px, med + mad_k * 1.4826 * mad):
                break
            rejected.append(solution.indices[worst])
            kept = [i for i in solution.indices if i != solution.indices[worst]]
            solution = self.solve(solution.X, kept, initial_board=solution.board, initial_offsets=solution.offsets)
        return solution, rejected

    # -- diagnostics -------------------------------------------------------
    def board_in_base(self, i, X, offsets=None):
        return self._robot(i, offsets) @ X @ self.d.tracking[i]

    def board_spread(self, solution):
        """How far each view's own board pose (PnP) lands from the fitted board."""
        centre_local = np.r_[np.mean(np.vstack([self.d.object[i] for i in solution.indices]), axis=0), 1]
        centre = (solution.board @ centre_local)[:3]
        positions, angles = [], []
        for i in solution.indices:
            b = self.board_in_base(i, solution.X, solution.offsets)
            positions.append(float(np.linalg.norm((b @ centre_local)[:3] - centre)))
            angles.append(math.degrees(Rotation.from_matrix(solution.board[:3, :3].T @ b[:3, :3]).magnitude()))
        return _stats(positions, angles, 'board_spread')

    def holdout_errors(self, solution, holdout):
        """Score views that were not in `solution`'s fit: pixels and board pose in base."""
        centre_local = np.r_[np.mean(np.vstack([self.d.object[i] for i in solution.indices]), axis=0), 1]
        centre = (solution.board @ centre_local)[:3]
        rms = self.per_view_rms(solution, holdout)
        positions, angles = [], []
        for i in holdout:
            b = self.board_in_base(i, solution.X, solution.offsets)
            positions.append(float(np.linalg.norm((b @ centre_local)[:3] - centre)))
            angles.append(math.degrees(Rotation.from_matrix(solution.board[:3, :3].T @ b[:3, :3]).magnitude()))
        return rms, positions, angles

    def leave_one_out(self, solution):
        """Refit without each view and predict it. This is the honest accuracy
        proxy available without external hardware: the view never influenced
        the calibration that scores it."""
        rms, positions, angles, shifts = [], [], [], []
        for i in solution.indices:
            others = [j for j in solution.indices if j != i]
            if len(others) < MIN_SAMPLES:
                return None
            fit = self.solve(solution.X, others, initial_board=solution.board, initial_offsets=solution.offsets)
            r, p, a = self.holdout_errors(fit, [i])
            rms += r; positions += p; angles += a
            shifts.append(float(np.linalg.norm(fit.X[:3, 3] - solution.X[:3, 3])))
        out = _stats(positions, angles, 'loo')
        out.update({
            'reprojection_rms_px': float(math.sqrt(np.mean(np.square(rms)))),
            'reprojection_max_px': float(np.max(rms)),
            'per_view_reprojection_px': [float(v) for v in rms],
            'max_transform_shift_m': float(np.max(shifts)),
            'views': len(rms),
        })
        return out

    def bootstrap(self, solution, n=30, seed=0):
        rng = np.random.default_rng(seed)
        idx = np.asarray(solution.indices)
        fits = []
        attempts = 0
        while len(fits) < n and attempts < 4 * n:
            attempts += 1
            pick = rng.choice(idx, len(idx), replace=True)
            if len(set(pick.tolist())) < MIN_SAMPLES + 1:
                continue
            try:
                fits.append(self.solve(solution.X, pick.tolist(), initial_board=solution.board,
                                       initial_offsets=solution.offsets).X)
            except (ValueError, np.linalg.LinAlgError):
                continue
        if len(fits) < max(8, n // 4):
            return None
        t = np.array([f[:3, 3] for f in fits])
        rot = Rotation.from_matrix(solution.X[:3, :3])
        dev = np.atleast_2d((rot.inv() * Rotation.from_matrix(np.array([f[:3, :3] for f in fits]))).as_rotvec())
        sigma = t.std(axis=0, ddof=1)
        rot_sigma = np.degrees(dev.std(axis=0, ddof=1))
        values, vectors = np.linalg.eigh(np.cov(t, rowvar=False))
        values = np.clip(values, 0, None)
        worst_axis = vectors[:, -1]
        if worst_axis[np.argmax(np.abs(worst_axis))] < 0:
            worst_axis = -worst_axis
        worst, best = float(math.sqrt(values[-1])), float(math.sqrt(values[0]))
        return {
            'method': 'bootstrap_reprojection',
            'n_bootstrap': len(fits),
            'n_requested': int(n),
            'sample_count': int(len(idx)),
            'translation_sigma_m': [float(v) for v in sigma],
            'rotation_sigma_deg': [float(v) for v in rot_sigma],
            'translation_sigma_rms_m': float(np.sqrt(np.mean(sigma ** 2))),
            'rotation_sigma_rms_deg': float(np.sqrt(np.mean(rot_sigma ** 2))),
            'worst_direction_sigma_m': worst,
            'worst_direction_axis': [float(v) for v in worst_axis],
            'best_direction_sigma_m': best,
            'anisotropy': float(worst / best) if best > 1e-12 else float('inf'),
        }


def _stats(positions, angles, prefix):
    positions, angles = np.asarray(positions, float), np.asarray(angles, float)
    return {
        'position_rms_m': float(np.sqrt(np.mean(positions ** 2))),
        'position_max_m': float(np.max(positions)),
        'rotation_rms_deg': float(np.sqrt(np.mean(angles ** 2))),
        'rotation_max_deg': float(np.max(angles)),
        'per_view_position_m': [float(v) for v in positions],
    }


def compare_refinement(solver, training):
    """Nested pose CV: rebuild AX=XB selection/rejection in each fold, then
    compare AX=XB and pixel refinement on the SAME omitted pose. No globally
    rejected view is silently removed from this evaluation. Dedicated
    validation poses are not used for this model-selection diagnostic.
    """
    from .calibration_backend import CalibrationBackend
    result = {'method': 'nested_leave_one_pose_out', 'passed': False, 'folds': []}
    if len(training) <= MIN_SAMPLES:
        result['reason'] = 'Need at least 5 training poses for refinement comparison'
        return result
    metrics = {'axxb': [[], [], []], 'reprojection': [[], [], []]}
    shifts = []
    for held in training:
        train = [i for i in training if i != held]
        try:
            detail = CalibrationBackend.compute_calibration_detailed(
                [pose_list(solver.d.robot[i]) for i in train],
                [pose_list(solver.d.tracking[i]) for i in train])
            kept = [train[i] for i in detail['kept_indices']]
            X = pose_matrix(detail['transform'])
            B = _initial_board(solver.d, X, kept)
            offsets = np.zeros(len(solver.joint_model.free)) if solver.joint_model is not None else None
            baseline = Solution(X, B, solver.d.camera.k, offsets, kept, 0., [], 0., True, 0)
            refined, rejected = solver.solve_robust(X, kept)
            if not refined.converged:
                raise ValueError('Refinement did not converge')
            row = {'held_out': int(solver.d.source_index[held]), 'algorithm': detail['algorithm_used'],
                   'fit_indices': [int(solver.d.source_index[i]) for i in refined.indices],
                   'axxb_transform': baseline.transform, 'refined_transform': refined.transform}
            for name, fit in [('axxb', baseline), ('reprojection', refined)]:
                values = solver.holdout_errors(fit, [held])
                if not all(np.isfinite(v).all() for v in values):
                    raise ValueError('Non-finite holdout errors')
                for bucket, v in zip(metrics[name], values):
                    bucket.extend(v)
                row[name] = dict(zip(('reprojection_px', 'position_m', 'rotation_deg'),
                                     [float(v[0]) for v in values]))
            result['folds'].append(row)
            shifts.append(float(np.linalg.norm(refined.X[:3, 3] - X[:3, 3])))
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            result['reason'] = f'Fold {solver.d.source_index[held]} failed: {exc}'
            return result
    for name, (pixels, positions, angles) in metrics.items():
        result[name] = {**_stats(positions, angles, name),
                       'reprojection_rms_px': float(np.sqrt(np.mean(np.square(pixels)))),
                       'reprojection_max_px': float(np.max(pixels)),
                       'per_view_reprojection_px': [float(v) for v in pixels], 'views': len(pixels)}
    # Non-regression margins cover small numerical/noise differences, not the
    # absolute accuracy requirements (which remain independently enforced).
    tolerances = {'reprojection_rms_px': .05, 'position_rms_m': .0005, 'rotation_rms_deg': .1}
    checks = []
    for key, absolute in tolerances.items():
        limit = result['axxb'][key] * 1.05 + absolute
        checks.append({'name': key, 'value': result['reprojection'][key], 'limit': limit,
                       'passed': result['reprojection'][key] <= limit})
    result['checks'] = checks
    result['passed'] = all(c['passed'] for c in checks)
    result['reason'] = 'No material held-out regression' if result['passed'] else 'Refinement worsens held-out predictions'
    result['max_refinement_shift_m'] = max(shifts)
    return result


# ---------------------------------------------------------------------------
# One-call pipeline used by the node and the offline CLI
# ---------------------------------------------------------------------------
def calibrate(dataset, initial_X, estimate_intrinsics=False, joint_model=None,
              validation_indices=(), bootstrap_samples=30, leave_one_out=True,
              excluded_indices=(), compare_refinement_cv=False):
    """Fit on all non-validation views, then report validation evidence.

    Returns a JSON-serialisable dict; 'transform' is X as [tx..qw]."""
    solver = ReprojectionSolver(dataset, estimate_intrinsics=estimate_intrinsics, joint_model=joint_model)
    validation = [i for i in validation_indices if 0 <= i < len(dataset)]
    all_training = [i for i in range(len(dataset)) if i not in validation]
    excluded = set(excluded_indices)
    if any(i < 0 or i >= len(dataset) for i in excluded) or excluded.intersection(validation):
        raise ValueError('Excluded training indices are invalid or overlap validation')
    training = [i for i in all_training if i not in excluded]
    solution, rejected = solver.solve_robust(initial_X, training)
    newly_rejected = list(rejected)
    rejected = sorted(excluded.union(rejected))
    report = {
        'solver': 'reprojection',
        'transform': solution.transform,
        'board_in_base': pose_list(solution.board),
        'converged': solution.converged,
        'training_views': len(solution.indices),
        'rejected_views': [int(dataset.source_index[i]) for i in rejected],
        'axxb_rejected_views': [int(dataset.source_index[i]) for i in sorted(excluded)],
        'reprojection_rejected_views': [int(dataset.source_index[i]) for i in newly_rejected],
        'kept_views': [int(dataset.source_index[i]) for i in solution.indices],
        'reprojection_rms_px': float(solution.rms_px),
        'per_view_reprojection_px': [float(v) for v in solution.per_view_rms_px],
        'board_spread': solver.board_spread(solution),
        'initial_delta_translation_m': float(np.linalg.norm(solution.X[:3, 3] - initial_X[:3, 3])),
        'initial_delta_rotation_deg': float(math.degrees(Rotation.from_matrix(
            initial_X[:3, :3].T @ solution.X[:3, :3]).magnitude())),
        'estimate_intrinsics': bool(estimate_intrinsics),
        'joint_offsets': None,
    }
    if estimate_intrinsics:
        k = solution.k
        report['intrinsics'] = {'fx': float(k[0, 0]), 'fy': float(k[1, 1]), 'cx': float(k[0, 2]), 'cy': float(k[1, 2]),
                                'camera_info_k': [float(v) for v in dataset.camera.k.reshape(-1)]}
    if joint_model is not None:
        report['joint_offsets'] = {joint_model.names[j]: float(v) for j, v in zip(joint_model.free, solution.offsets)}
    if leave_one_out and not compare_refinement_cv and len(solution.indices) > MIN_SAMPLES:
        report['leave_one_out'] = solver.leave_one_out(solution)
    if validation:
        rms, positions, angles = solver.holdout_errors(solution, validation)
        report['validation_views'] = {
            **_stats(positions, angles, 'validation'),
            'reprojection_rms_px': float(math.sqrt(np.mean(np.square(rms)))),
            'reprojection_max_px': float(np.max(rms)),
            'views': len(validation),
        }
    if compare_refinement_cv:
        report['refinement_comparison'] = compare_refinement(solver, all_training)
        comparison = report['refinement_comparison']
        if 'reprojection' in comparison:
            report['leave_one_out'] = dict(comparison['reprojection'])
            report['leave_one_out']['method'] = 'nested_leave_one_pose_out_all_training_views'
    if bootstrap_samples > 0:
        report['uncertainty'] = solver.bootstrap(solution, n=bootstrap_samples)
    report['_solution'] = solution
    return report
