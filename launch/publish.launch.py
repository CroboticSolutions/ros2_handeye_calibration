import os

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():
    calibration_file = DeclareLaunchArgument(
        'calibration_file',
        default_value=os.path.expanduser('~/.ros/hand_eye_calibration.yaml'),
        description='Path to the hand-eye calibration YAML file',
    )

    publish_node = Node(
        package='hand_eye_calibration',
        executable='hand_eye_calibration_publisher',
        name='hand_eye_calibration_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': launch.substitutions.LaunchConfiguration('use_sim_time', default='true')},
            {'calibration_file': launch.substitutions.LaunchConfiguration('calibration_file')},
        ],
    )

    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time for TF',
    )

    return LaunchDescription([
        use_sim_time,
        calibration_file,
        publish_node,
    ])
