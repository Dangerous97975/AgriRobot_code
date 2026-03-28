from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def generate_launch_description():

    agri_bot_pkg = get_package_share_directory('agri_bot')

    sensor_position_path = os.path.join(
        agri_bot_pkg, 'config', 'sensor_positions.yaml')
    realsense_param_path = os.path.join(
        agri_bot_pkg, 'config', 'realsense_param.json')

    with open(sensor_position_path, 'r') as file:
        sensor_positions = yaml.safe_load(file)

    d455f_pos = sensor_positions['D455f']
    # 定義相機參數
    d455f_x_arg = DeclareLaunchArgument(
        'd455f_x', default_value=str(d455f_pos['position']['x']))
    d455f_y_arg = DeclareLaunchArgument(
        'd455f_y', default_value=str(d455f_pos['position']['y']))
    d455f_z_arg = DeclareLaunchArgument(
        'd455f_z', default_value=str(d455f_pos['position']['z']))
    d455f_roll_arg = DeclareLaunchArgument(
        'd455f_roll', default_value=str(d455f_pos['orientation']['roll_deg']))
    d455f_pitch_arg = DeclareLaunchArgument(
        'd455f_pitch', default_value=str(d455f_pos['orientation']['pitch_deg']))
    d455f_yaw_arg = DeclareLaunchArgument(
        'd455f_yaw', default_value=str(d455f_pos['orientation']['yaw_deg']))

    # 引用agri_bot的robot_description並傳遞參數
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('agri_bot'),
                'launch',
                'robot_description.launch.py'
            ])
        ]),
        launch_arguments={
            'camera_x': LaunchConfiguration('d455f_x_arg'),
            'camera_y': LaunchConfiguration('d455f_y_arg'),
            'camera_z': LaunchConfiguration('d455f_z_arg'),
            'camera_roll_deg': LaunchConfiguration('d455f_roll_arg'),
            'camera_pitch_deg': LaunchConfiguration('d455f_pitch_arg'),
            'camera_yaw_deg': LaunchConfiguration('d455f_yaw_arg')
        }.items()
    )

    # 啟動 RealSense 節點
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ]),
        launch_arguments={
            'camera_name': 'D455f',
            'camera_namespace': 'agri_bot',
            'json_file_path': realsense_param_path,
            'initial_reset': 'true',
            'enable_color': 'true',
            'enable_depth': 'true',
            'depth_module.depth_profile': "480x270x15",
            'depth_module.enable_auto_exposure': 'false',
            'depth_module.exposure': '1000',
            'pointcloud.enable': 'true',
            'pointcloud.allow_no_texture_points': 'false',
            'pointcloud.texture_stream': 'any',
            'align_depth.enable': 'true',
            'depth_module.enable_auto_exposure': 'false',
            'clip_distance': '5.14'
        }.items()
    )

    return LaunchDescription([
        d455f_x_arg,
        d455f_y_arg,
        d455f_z_arg,
        d455f_roll_arg,
        d455f_pitch_arg,
        d455f_yaw_arg,
        robot_description_launch,
        realsense_launch,
    ])
