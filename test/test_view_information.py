import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo
from hand_eye_calibration.visibility import BoardFraming
from hand_eye_calibration.view_information import view_information,ViewInformation


def test_image_information_accounts_for_board_nuisance_parameters():
    framing=BoardFraming({'squares_x':13,'squares_y':9,'square_length_m':.015},
        CameraInfo(width=640,height=480,k=[600.,0.,320.,0.,600.,240.,0.,0.,1.]))
    mount=np.eye(4);board=np.eye(4);board[:3,3]=[-.0975,-.0675,.65]
    poses=[np.eye(4)]
    for angle in [-.15,.15]:
        p=np.eye(4);p[:3,:3]=Rotation.from_rotvec([angle,0,0]).as_matrix();poses.append(p)
    information=view_information(poses[0],mount,board,framing)
    assert information.shape==(12,12)
    np.testing.assert_allclose(information,information.T,atol=1e-8)
    assert np.linalg.eigvalsh(information).min()>-1e-6
    selection=ViewInformation(poses,mount,board,framing)
    novel=np.eye(4);novel[:3,:3]=Rotation.from_rotvec([0,.15,0]).as_matrix()
    assert selection.gain(novel)>selection.gain(poses[0])
