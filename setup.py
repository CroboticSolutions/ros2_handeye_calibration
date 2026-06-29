import os

from glob import glob
from setuptools import setup

package_name = 'hand_eye_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install under share/<pkg>/launch/ only so `ros2 launch` does not find duplicates
        # (flat install + launch/ subfolder both matched calibration.launch.py).
        (
            os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*')),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='todo@todo.todo',
    description='Minimal ROS2 hand-eye calibration package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_eye_calibration = hand_eye_calibration.node:main',
            'charuco_detector = hand_eye_calibration.charuco_detector:main',
            'hand_eye_calibration_publisher = hand_eye_calibration.publish_node:main',
            'handeye_depthai_mount_args = hand_eye_calibration.depthai_mount_cli:main',
        ],
    },
)
