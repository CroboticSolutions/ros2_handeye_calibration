import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def _launch_calibration_setup(context, *_args, **_kwargs):
    lc = context.launch_configurations
    use_sim_str = lc.get('use_sim_time', 'false').lower()
    use_sim_time = use_sim_str in ('true', '1', 'yes')

    # Expect aruco_markers (or aruco_ros) running separately. Use marker_size in **meters** (e.g. 47 mm -> 0.047);
    # wrong size breaks PnP and invalidates saved hand-eye YAML until you recapture samples.
    calibration_node = Node(
        package='hand_eye_calibration',
        executable='hand_eye_calibration',
        name='hand_eye_calibration',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'tracking_base_frame': lc['tracking_base_frame']},
            {'tracking_marker_frame': lc['tracking_marker_frame']},
            {'robot_base_frame': lc['robot_base_frame']},
            {'robot_effector_frame': lc['robot_effector_frame']},
            {'calibration_type': lc['calibration_type']},
            {'calibration_file': lc['calibration_file']},
        ],
    )
    return [calibration_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'calibration_file',
                default_value=os.path.expanduser('~/.ros/hand_eye_calibration.yaml'),
                description='Path to save/load hand-eye calibration YAML',
            ),
            DeclareLaunchArgument(
                'tracking_base_frame',
                default_value='camera_optical_frame',
                description='Camera frame (where ArUco marker is detected)',
            ),
            DeclareLaunchArgument(
                'tracking_marker_frame',
                default_value='aruco_marker_0',
                description='ArUco marker frame (matches aruco_ros single.launch.py marker_frame)',
            ),
            DeclareLaunchArgument(
                'robot_base_frame',
                default_value='base_link',
                description='Robot base frame',
            ),
            DeclareLaunchArgument(
                'robot_effector_frame',
                default_value='link6',
                description='Robot end effector frame (where camera is mounted)',
            ),
            DeclareLaunchArgument(
                'calibration_type',
                default_value='eye-in-hand',
                description='Options are eye-in-hand or eye-on-base',
            ),
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='false',
                description='true only when using Gazebo/sim and /clock is published; false for real robots',
            ),
            OpaqueFunction(function=_launch_calibration_setup),
        ]
    )
