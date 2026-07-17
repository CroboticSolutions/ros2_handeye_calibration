import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def _launch_calibration_setup(context, *_args, **_kwargs):
    lc = context.launch_configurations
    use_sim_str = lc.get('use_sim_time', 'false').lower()
    use_sim_time = use_sim_str in ('true', '1', 'yes')

    # ChArUco board detector: detects the printed board, publishes its pose as
    # TF (tracking_base_frame -> tracking_marker_frame) plus a chessboard_visible
    # Bool for the GUI. Replaces the external single-ArUco-marker detector.
    charuco_detector = Node(
        package='hand_eye_calibration',
        executable='charuco_detector',
        name='charuco_detector',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'image_topic': lc['image_topic']},
            {'camera_info_topic': lc['camera_info_topic']},
            {'board_frame': lc['tracking_marker_frame']},
            {'camera_optical_frame': lc['tracking_base_frame']},
            {'squares_x': int(lc['squares_x'])},
            {'squares_y': int(lc['squares_y'])},
            {'square_length_m': float(lc['square_length_m'])},
            {'marker_length_m': float(lc['marker_length_m'])},
            {'aruco_dictionary': lc['aruco_dictionary']},
        ],
    )

    # Collector: looks up robot + board TF on capture_point, runs hand-eye solve,
    # saves YAML on save_calibration.
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
            {'image_topic': lc['image_topic']},
            {'camera_info_topic': lc['camera_info_topic']},
            {'marker_size': float(lc['square_length_m'])},
        ],
    )
    return [charuco_detector, calibration_node]


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
                default_value='oak_rgb_camera_optical_frame',
                description=(
                    'Camera optical frame in TF (must exist). Piper + OAK-D Pro W RGB: '
                    'oak_rgb_camera_optical_frame. Verify with: ros2 run tf2_ros tf2_monitor'
                ),
            ),
            DeclareLaunchArgument(
                'tracking_marker_frame',
                default_value='charuco_board',
                description='Frame the ChArUco detector broadcasts for the board pose',
            ),
            DeclareLaunchArgument(
                'robot_base_frame',
                default_value='base_link',
                description='Robot base frame',
            ),
            DeclareLaunchArgument(
                'robot_effector_frame',
                default_value='link6',
                description='Robot end effector frame (where the camera is mounted)',
            ),
            DeclareLaunchArgument(
                'calibration_type',
                default_value='eye-in-hand',
                description='Options are eye-in-hand or eye-on-base',
            ),
            DeclareLaunchArgument(
                'image_topic',
                default_value='/oak/rgb/image_raw',
                description='RGB image topic the ChArUco detector subscribes to',
            ),
            DeclareLaunchArgument(
                'camera_info_topic',
                default_value='/oak/rgb/camera_info',
                description='CameraInfo topic providing intrinsics for board pose',
            ),
            DeclareLaunchArgument(
                'squares_x',
                default_value='13',
                description='ChArUco squares in X (printed board is 13 wide x 9 high; X/Y are not swappable)',
            ),
            DeclareLaunchArgument(
                'squares_y',
                default_value='9',
                description='ChArUco squares in Y (printed board is 13 wide x 9 high; X/Y are not swappable)',
            ),
            DeclareLaunchArgument(
                'square_length_m',
                default_value='0.015',
                description='ChArUco square length in meters (15 mm board)',
            ),
            DeclareLaunchArgument(
                'marker_length_m',
                default_value='0.011',
                description='ChArUco marker length in meters (11 mm)',
            ),
            DeclareLaunchArgument(
                'aruco_dictionary',
                default_value='DICT_4X4_100',
                description=(
                    'ArUco dictionary of the printed board. A 13x9 board needs 58 markers, '
                    'so DICT_4X4_50 is too small; default DICT_4X4_100. Change to match your print.'
                ),
            ),
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='false',
                description='true only when using Gazebo/sim and /clock is published; false for real robots',
            ),
            OpaqueFunction(function=_launch_calibration_setup),
        ]
    )
