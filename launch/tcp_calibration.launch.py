import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def _launch_tcp_calibration_setup(context, *_args, **_kwargs):
    lc = context.launch_configurations
    use_sim_str = lc.get('use_sim_time', 'false').lower()
    use_sim_time = use_sim_str in ('true', '1', 'yes')

    pivot_node = Node(
        package='hand_eye_calibration',
        executable='tool_tcp_calibration',
        name='tool_tcp_calibration',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_base_frame': lc['robot_base_frame']},
            {'robot_flange_frame': lc['robot_flange_frame']},
            {'tcp_name': lc['tcp_name']},
            {'calibration_file': lc['calibration_file']},
            {'spike_axis_base': lc['spike_axis_base']},
        ],
    )
    return [pivot_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'calibration_file',
                default_value=os.path.expanduser('~/.ros/tool_tcp_calibration.yaml'),
                description='Path to save the tool TCP (pivot) calibration YAML',
            ),
            DeclareLaunchArgument(
                'robot_base_frame',
                default_value='base_link',
                description='Robot base frame',
            ),
            DeclareLaunchArgument(
                'robot_flange_frame',
                default_value='link6',
                description='Robot flange frame (the samples are captured here, not at the current eef_link)',
            ),
            DeclareLaunchArgument(
                'tcp_name',
                default_value='tool_tcp',
                description='Name of the calibrated tool TCP frame/link to save into the YAML',
            ),
            DeclareLaunchArgument(
                'spike_axis_base',
                default_value='0,0,1',
                description='Known spike direction in the base frame for the axis-alignment '
                            'round (vertical when the base is level)',
            ),
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='false',
                description='true only when using Gazebo/sim and /clock is published; false for real robots',
            ),
            OpaqueFunction(function=_launch_tcp_calibration_setup),
        ]
    )
