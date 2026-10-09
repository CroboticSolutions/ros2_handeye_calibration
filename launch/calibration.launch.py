import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def _launch_calibration_setup(context, *_args, **_kwargs):
    lc = context.launch_configurations
    use_sim_str = lc.get('use_sim_time', 'false').lower()
    piper_sim = lc['image_topic'].startswith('/piper/camera/')
    use_sim_time = piper_sim or use_sim_str in ('true', '1', 'yes')

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
            {'robot_motion_tip_frame': lc['robot_motion_tip_frame']},
            {'calibration_type': lc['calibration_type']},
            {'calibration_file': lc['calibration_file']},
            {'auto_enabled': lc.get('auto_enabled', 'auto').lower() == 'true' or (lc.get('auto_enabled', 'auto') == 'auto' and piper_sim)},
            {'auto_check_collisions': lc['auto_check_collisions'].lower() == 'true'},
            {'auto_group': lc['auto_group']},
            {'auto_controller': lc['auto_controller']},
            {'auto_max_position_sigma_m': float(lc['auto_max_position_sigma_m'])},
            {'auto_min_training_samples': int(lc['auto_min_training_samples'])},
            {'auto_max_training_samples': int(lc['auto_max_training_samples'])},
            {'auto_validation_views': int(lc['auto_validation_views'])},
            {'intrinsic_return_to_base': lc['intrinsic_return_to_base'].lower() == 'true'},
            {'auto_heartbeat_timeout_s': float(lc['auto_heartbeat_timeout_s'])},
            {'auto_max_camera_excursion_m': float(lc['auto_max_camera_excursion_m'])},
            {'camera_latency_s': float(lc['camera_latency_s'])},
            {'solver': lc['solver']},
            {'acceptance_mode': lc['acceptance_mode']},
            {'dataset_dir': os.path.expanduser(lc['dataset_dir'])},
            {'pointcloud_topic': lc['pointcloud_topic']},
            {'image_topic': lc['image_topic']},
            {'depth_topic': lc['depth_topic']},
            {'depth_info_topic': lc['depth_info_topic']},
            {'camera_info_topic': lc['camera_info_topic']},
            {'marker_size': float(lc['square_length_m'])},
            {'squares_x': int(lc['squares_x'])},
            {'squares_y': int(lc['squares_y'])},
            {'square_length_m': float(lc['square_length_m'])},
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
                'robot_motion_tip_frame',
                default_value='',
                description='End of the moved joint chain; empty uses robot_effector_frame '
                            '(set link6 when the camera is mounted on link5)',
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
            DeclareLaunchArgument('auto_group', default_value='', description='MoveIt group; empty discovers the group matching the robot chain'),
            DeclareLaunchArgument('auto_controller', default_value='', description='FollowJointTrajectory action; empty discovers the active chain controller'),
            DeclareLaunchArgument('auto_enabled', default_value='auto', description='Enable automatic eye-in-hand calibration'),
            DeclareLaunchArgument('auto_check_collisions', default_value='true', description='Check joint paths against the MoveIt scene before motion'),
            DeclareLaunchArgument('pointcloud_topic', default_value='', description='Optional cloud topic for preflight diagnostics'),
            DeclareLaunchArgument('depth_topic', default_value='', description='Depth image for the depth calibration sweep (colour-aligned, or native with depth_info_topic)'),
            DeclareLaunchArgument('depth_info_topic', default_value='', description='CameraInfo of a native (not colour-aligned) depth_topic'),
            DeclareLaunchArgument('auto_max_position_sigma_m', default_value='0.002', description='Maximum bootstrap position uncertainty (1 sigma, metres) for automatic saving'),
            DeclareLaunchArgument('intrinsic_return_to_base', default_value='true', description='Return along a planned path to the reference pose between captures'),
            DeclareLaunchArgument('auto_validation_views', default_value='5', description='Held-out poses collected after training; never used in the solve'),
            DeclareLaunchArgument('auto_heartbeat_timeout_s', default_value='0.0', description='>0: stop automatic motion when the GUI heartbeat is silent this long (GUI passes 3)'),
            DeclareLaunchArgument('auto_max_camera_excursion_m', default_value='0.30', description='Maximum camera displacement of targeted views from the starting view'),
            DeclareLaunchArgument('camera_latency_s', default_value='-1.0', description='<0 measures camera latency from image stamps; >=0 fixes it'),
            DeclareLaunchArgument('solver', default_value='intrinsic_pose', description='intrinsic_pose: Shah plus all-view joint pose optimization; reprojection/axxb: legacy'),
            DeclareLaunchArgument('acceptance_mode', default_value='enforce', description='enforce: failing results go to <file>.rejected.yaml; warn: saved with a warning'),
            DeclareLaunchArgument('dataset_dir', default_value='~/.ros/hand_eye_calibration_runs', description='Raw dataset of every save/failed run'),
            DeclareLaunchArgument('auto_min_training_samples', default_value='20', description='Intrinsic: main poses after eight initial poses; legacy: minimum training poses'),
            DeclareLaunchArgument('auto_max_training_samples', default_value='30', description='Maximum training poses; continue until quality passes, then collect held-out views'),
            OpaqueFunction(function=_launch_calibration_setup),
        ]
    )
