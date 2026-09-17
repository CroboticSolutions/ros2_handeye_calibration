"""Predict image sensitivity to camera mount AND unknown board pose.

Used only to rank views, not to report calibrated uncertainty. Equal pixel noise
is a planning approximation; the final gate still uses measured bootstrap fits.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def perturb(pose, axis, amount):
    result=pose.copy()
    if axis<3:
        result[axis,3]+=amount*.01
    else:
        vector=np.zeros(3);vector[axis-3]=amount*.05
        result[:3,:3]=pose[:3,:3]@Rotation.from_rotvec(vector).as_matrix()
    return result


def view_information(robot, mount, board, framing):
    def pixels(x,b):
        result=framing.pixels(np.linalg.inv(robot@x)@b)
        if result is None or not np.isfinite(result).all():
            raise ValueError('Invalid projected calibration view.')
        return result.ravel()
    step=1e-4
    columns=[]
    for index in range(12):
        if index<6:
            plus=pixels(perturb(mount,index,step),board)
            minus=pixels(perturb(mount,index,-step),board)
        else:
            plus=pixels(mount,perturb(board,index-6,step))
            minus=pixels(mount,perturb(board,index-6,-step))
        columns.append((plus-minus)/(2*step))
    jacobian=np.column_stack(columns)
    return jacobian.T@jacobian


def mount_log_information(information):
    # Marginalize the uncertain board pose instead of treating it as ground truth.
    matrix=information+np.eye(12)*1e-6
    marginal=matrix[:6,:6]-matrix[:6,6:]@np.linalg.solve(matrix[6:,6:],matrix[6:,:6])
    values=np.linalg.eigvalsh((marginal+marginal.T)*.5)
    return float(np.log(np.maximum(values,1e-10)).sum())


class ViewInformation:
    def __init__(self, poses, mount, board, framing):
        self.mount,self.board,self.framing=mount,board,framing
        self.information=sum((view_information(p,mount,board,framing) for p in poses),np.zeros((12,12)))
        self.baseline=mount_log_information(self.information)

    def gain(self, pose):
        added=view_information(pose,self.mount,self.board,self.framing)
        return max(0.,mount_log_information(self.information+added)-self.baseline)
