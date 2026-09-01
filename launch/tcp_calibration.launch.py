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
            {'robot_firmware_version': lc['robot_firmware_version']},
            {'capture_burst_duration_s': float(lc['capture_burst_duration_s'])},
            {'capture_burst_samples': int(lc['capture_burst_samples'])},
            {'capture_burst_min_samples': int(lc['capture_burst_min_samples'])},
            {'capture_translation_p95_limit_m': float(lc['capture_translation_p95_limit_m'])},
            {'capture_rotation_p95_limit_deg': float(lc['capture_rotation_p95_limit_deg'])},
            {'duplicate_orientation_limit_deg': float(lc['duplicate_orientation_limit_deg'])},
            {'max_tf_age_s': float(lc['max_tf_age_s'])},
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
            DeclareLaunchArgument(
                'robot_firmware_version',
                default_value='unknown',
                description='Metadata saved with the calibration; no CAN query is issued',
            ),
            DeclareLaunchArgument('capture_burst_duration_s', default_value='1.2'),
            DeclareLaunchArgument('capture_burst_samples', default_value='50'),
            DeclareLaunchArgument('capture_burst_min_samples', default_value='15'),
            DeclareLaunchArgument('capture_translation_p95_limit_m', default_value='0.0005'),
            DeclareLaunchArgument('capture_rotation_p95_limit_deg', default_value='0.20'),
            DeclareLaunchArgument('duplicate_orientation_limit_deg', default_value='5.0'),
            DeclareLaunchArgument('max_tf_age_s', default_value='0.25'),
            OpaqueFunction(function=_launch_tcp_calibration_setup),
        ]
    )
