import os

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():

    calibration_file = DeclareLaunchArgument(
        'calibration_file',
        default_value=os.path.expanduser('~/.ros/hand_eye_calibration.yaml'),
        description='Path to save/load hand-eye calibration YAML',
    )
    tracking_base_frame = DeclareLaunchArgument('tracking_base_frame', 
                                                 default_value="camera_optical_frame",
                                                 description="Camera frame (where ArUco marker is detected)")
    tracking_marker_frame = DeclareLaunchArgument('tracking_marker_frame',
                                                  default_value="aruco_marker_0",
                                                  description="ArUco marker frame (matches aruco_ros single.launch.py marker_frame)")
    robot_base_frame = DeclareLaunchArgument('robot_base_frame',
                                             default_value="base_link",
                                             description="Robot base frame")
    robot_effector_frame = DeclareLaunchArgument('robot_effector_frame',
                                                 default_value="link6",
                                                 description="Robot end effector frame (where camera is mounted)")
    calibration_type = DeclareLaunchArgument('calibration_type',
                                             default_value="eye-in-hand",
                                             description="Options are eye-in-hand or eye-on-base")

    # Expect aruco_ros single.launch.py (or ros2_aruco) to be running separately; do not start ArUco here
    calibration_node = Node(
            package='hand_eye_calibration',
            executable='hand_eye_calibration',
            name='hand_eye_calibration',
            output='screen',
            parameters=[
                {'use_sim_time': True},  # Use simulation time for TF
                {'tracking_base_frame': launch.substitutions.LaunchConfiguration('tracking_base_frame')},
                {'tracking_marker_frame': launch.substitutions.LaunchConfiguration('tracking_marker_frame')},
                {'robot_base_frame': launch.substitutions.LaunchConfiguration('robot_base_frame')},
                {'robot_effector_frame': launch.substitutions.LaunchConfiguration('robot_effector_frame')},
                {'calibration_type': launch.substitutions.LaunchConfiguration('calibration_type')},
                {'calibration_file': launch.substitutions.LaunchConfiguration('calibration_file')},
            ]
    )

    ll = list()
    ll.append(calibration_file)
    ll.append(tracking_base_frame)
    ll.append(tracking_marker_frame)
    ll.append(robot_base_frame)
    ll.append(robot_effector_frame)
    ll.append(calibration_type)
    ll.append(calibration_node)

    return LaunchDescription(ll)