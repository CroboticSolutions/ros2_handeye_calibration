# Copyright 2026 Intrinsic Innovation LLC
# Modifications: ROS/Python adaptation, 2026.
# Licensed under the Apache License, Version 2.0. See THIRD_PARTY_NOTICES.md.
"""Shah + joint pose least squares, adapted from Intrinsic camera_to_robot_calibration.

Same pose residual and data-derived rotation/translation weighting; SciPy replaces
Ceres. All input views participate with squared loss. No pose rejection or pixel fit.
Robot poses use the collector convention (inverted FK for eye-on-base).
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

UPSTREAM = 'c61bf075f2335371c6367b61117e8a62bb960c3b'


def matrix(p):
    p = np.asarray(p, float)
    if p.shape != (7,) or not np.isfinite(p).all() or np.linalg.norm(p[3:]) < 1e-10:
        raise ValueError('Invalid calibration pose')
    m = np.eye(4); m[:3,:3] = R.from_quat(p[3:]).as_matrix(); m[:3,3] = p[:3]
    return m


def pose(m):
    return np.r_[m[:3,3], R.from_matrix(m[:3,:3]).as_quat()].tolist()


def pack(m):
    return np.r_[m[:3,3], R.from_matrix(m[:3,:3]).as_rotvec()]


def unpack(v):
    m = np.eye(4); m[:3,3] = v[:3]; m[:3,:3] = R.from_rotvec(v[3:]).as_matrix()
    return m


def geometry(robots):
    rotations = np.array([m[:3,:3] for m in robots])
    relative = np.einsum('ji,njk->nik', rotations[0], rotations) - np.eye(3)
    eigen = np.linalg.eigvalsh(np.einsum('nji,njk->ik', relative, relative))
    return {'ready': bool(len(robots) >= 6 and eigen[0] > 1e-4),
            'information_eigenvalues': eigen.tolist(), 'samples': len(robots)}


def metrics(robots, tracking, X, Y):
    expected = np.linalg.inv(X)[None] @ np.linalg.inv(robots) @ Y
    pos = np.linalg.norm(expected[:,:3,3] - tracking[:,:3,3], axis=1)
    rot = np.degrees(R.from_matrix(np.swapaxes(expected[:,:3,:3],1,2) @ tracking[:,:3,:3]).magnitude())
    return {'position_rms_m': float(np.sqrt(np.mean(pos**2))), 'position_max_m': float(max(pos)),
            'rotation_rms_deg': float(np.sqrt(np.mean(rot**2))), 'rotation_max_deg': float(max(rot)),
            'per_view_position_m': pos.tolist(), 'per_view_rotation_deg': rot.tolist(), 'views':len(robots)}


def solve(robot, tracking, initial=None):
    if len(robot) != len(tracking) or len(robot) < 3:
        raise ValueError('Need at least three matching pose pairs')
    A = np.array([matrix(p) for p in robot]); C = np.array([matrix(p) for p in tracking])
    if initial is None:
        invA = np.linalg.inv(A)
        try:
            ry,ty,rx,tx = cv2.calibrateRobotWorldHandEye(
                list(C[:,:3,:3]), list(C[:,:3,3]), list(invA[:,:3,:3]), list(invA[:,:3,3]),
                method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
        except cv2.error as exc:
            raise ValueError('Shah initialization failed; increase multi-axis pose diversity') from exc
        X = np.eye(4); X[:3,:3]=rx; X[:3,3]=np.asarray(tx).ravel(); X=np.linalg.inv(X)
        Y = np.eye(4); Y[:3,:3]=ry; Y[:3,3]=np.asarray(ty).ravel(); Y=np.linalg.inv(Y)
    else:
        X,Y = initial
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError('Non-finite Shah initialization')
    invC = np.linalg.inv(C)
    def errors(v):
        left = A @ unpack(v[:6]); right = unpack(v[6:])[None] @ invC
        dt = right[:,:3,3] - left[:,:3,3]
        dr = R.from_matrix(np.swapaxes(left[:,:3,:3],1,2) @ right[:,:3,:3]).as_rotvec()
        return dt,dr
    v0 = np.r_[pack(X),pack(Y)]; dt,dr = errors(v0)
    # Equivalent to Intrinsic's weighting up to a common residual scale.
    scale = np.sqrt(max(np.mean(dt**2),1e-24)/max(np.mean(dr**2),1e-24))
    def residual(v):
        dt,dr = errors(v)
        return np.c_[dt/scale,dr].ravel()
    fit = least_squares(residual,v0,loss='linear',max_nfev=200,ftol=1e-11,xtol=1e-11,gtol=1e-11)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('Joint camera/board optimization did not converge')
    X,Y=unpack(fit.x[:6]),unpack(fit.x[6:])
    return {'solver':'intrinsic_pose','algorithm':'SHAH+NONLINEAR', 'upstream_commit':UPSTREAM,
            'optimizer':'scipy_least_squares', 'transform':pose(X), 'board_in_base':pose(Y),
            'converged':True,'training_views':len(A),'kept_views':list(range(len(A))),
            'rejected_views':[], 'geometry':geometry(A), 'pose_metrics':metrics(A,C,X,Y),
            'translation_rotation_scale_m_per_rad':float(scale)}


def uncertainty(robot, tracking, report, n=40):
    if n < 8 or len(robot)<6:
        return None
    rng=np.random.default_rng(42); xs=[]
    for _ in range(n):
        ids=rng.integers(0,len(robot),len(robot))
        if not geometry([matrix(robot[i]) for i in ids])['ready']:
            continue
        try:
            r=solve([robot[i] for i in ids],[tracking[i] for i in ids],
                    initial=(matrix(report['transform']),matrix(report['board_in_base'])))
            xs.append(matrix(r['transform']))
        except ValueError:
            continue
    if len(xs)<max(8,int(.8*n)):
        return None
    xs=np.array(xs);cov=np.cov(xs[:,:3,3].T);e,v=np.linalg.eigh(cov);e=np.maximum(e,0)
    angles=R.from_matrix(matrix(report['transform'])[:3,:3].T@xs[:,:3,:3]).as_rotvec()
    return {'n_bootstrap':len(xs),'translation_sigma_m':np.sqrt(np.maximum(np.diag(cov),0)).tolist(),
            'rotation_sigma_deg':np.degrees(np.std(angles,axis=0,ddof=1)).tolist(),
            'worst_direction_sigma_m':float(np.sqrt(e[-1])), 'best_direction_sigma_m':float(np.sqrt(e[0])),
            'worst_direction_axis':v[:,-1].tolist(), 'guidance':'Pose bootstrap; verify on independent views.'}


def evaluate(report, sigma_limit=.002, min_samples=20, translation_limit=.003, rotation_limit=1., require_validation=False):
    checks=[]
    def add(name,value,limit,passed):
        checks.append(dict(name=name,value=value,limit=limit,passed=bool(passed)))
    if not report:
        add('pose_solver',None,True,False)
    else:
        add('sample_count',report['training_views'],min_samples,report['training_views']>=min_samples)
        add('pose_geometry',report['geometry']['ready'],True,report['geometry']['ready'])
        m=report['pose_metrics']
        for name,limit in [('position_rms_m',translation_limit),('rotation_rms_deg',rotation_limit)]:
            add(name,m[name],limit,np.isfinite(m[name]) and m[name]<=limit)
        sig=(report.get('uncertainty') or {}).get('worst_direction_sigma_m')
        add('position_sigma_m',sig,sigma_limit,sig is not None and np.isfinite(sig) and sig<=sigma_limit)
        validation=report.get('validation_views')
        if require_validation or validation:
            add('validation_views',(validation or {}).get('views',0),3,(validation or {}).get('views',0)>=3)
            for name,limit in [('position_max_m',.004),('rotation_max_deg',3.)]:
                value=(validation or {}).get(name)
                add('validation_'+name,value,limit,value is not None and np.isfinite(value) and value<=limit)
    failed=[c['name'] for c in checks if not c['passed']]
    return {'passed':not failed,'checks':checks,'summary':'All pose acceptance checks passed.' if not failed else 'Failed: '+', '.join(failed)}
