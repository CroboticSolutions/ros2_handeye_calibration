import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from hand_eye_calibration import welding_gun_apply_cli as cli

FEMTO = '''<robot><xacro:femto_bolt prefix="camera" parent="${parent_link}">
      <origin xyz="${'1 2 3' if use_nominal_extrinsics else '4 5 6'}"
              rpy="${'0 0 0' if use_nominal_extrinsics else '0 0 0'}"/>
    </xacro:femto_bolt></robot>'''

GUN = '''<joint name="arm_tcp_joint" type="fixed">
    <xacro:if value="$(arg simulation)">
      <origin xyz="0.05360585785 0 0.2085160675" rpy="0 -0.9599310886 0"/>
    </xacro:if>
    <xacro:unless value="$(arg simulation)">
      <origin xyz="0 0 0" rpy="-0.03 -0.9 0.02"/>
    </xacro:unless>
  </joint>'''

# TCP-v1 (2026-09-23) and the arm_tcp_joint origin that was baked by hand from it.
TCP_V1 = {'tx': 0.05971650727739351, 'ty': -0.002810430193906347, 'tz': 0.19015803585545335,
          'qx': 0.3245431126388426, 'qy': 0.008110963748323693, 'qz': 0.9458320329677071,
          'qw': 0.002783112994304413}
TCP_V1_RPY = [-0.027929237913, -0.909524738174, 0.022044250167]


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_tcp_real_branch_reproduces_hand_baked_origin(tmp_path):
    gun = tmp_path / 'gun.xacro'
    gun.write_text(GUN)
    _, T = cli.load_calibration(write(tmp_path, 'tcp.yaml', {'transform': TCP_V1, 'parent_frame': 'link6'}))
    xyz, rpy = cli.bake_tcp(str(gun), T)
    np.testing.assert_allclose(rpy, TCP_V1_RPY, atol=1e-9)
    np.testing.assert_allclose(xyz, [TCP_V1['tx'], TCP_V1['ty'], TCP_V1['tz']])
    text = gun.read_text()
    assert 'rpy="0 -0.9599310886 0"' in text  # simulation branch untouched
    # arm_tcp +X is the calibrated tool +Z.
    tool_z = T[:3, :3] @ [0, 0, 1]
    tcp_x = Rotation.from_euler('xyz', rpy).as_matrix() @ [1, 0, 0]
    np.testing.assert_allclose(tcp_x, tool_z, atol=1e-9)


def test_camera_real_origin_is_calibration_times_inverse_inner_chain(tmp_path):
    femto = tmp_path / 'femto.xacro'
    femto.write_text(FEMTO)
    T = cli._matrix([-.11, -.03, .01], [.3, -.2, 1.5])
    inner = cli._matrix([.01, .02, .03], [-1.57, 0, -1.57])
    mount = cli.bake_camera(str(femto), T, inner)
    np.testing.assert_allclose(mount @ inner, T, atol=1e-12)
    text = femto.read_text()
    assert "'1 2 3' if use_nominal_extrinsics" in text  # nominal kept by default
    assert "'4 5 6'" not in text


def test_failed_acceptance_is_refused(tmp_path):
    path = write(tmp_path, 'cal.yaml', {'transform': TCP_V1, 'acceptance': {'passed': False, 'summary': 'x'}})
    with pytest.raises(SystemExit):
        cli.load_calibration(path)
    cli.load_calibration(path, force=True)
