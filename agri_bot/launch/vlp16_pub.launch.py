# vlp16.launch.py
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

    with open(sensor_position_path, 'r') as file:
        sensor_positions = yaml.safe_load(file)

    lidar_pos = sensor_positions['VLP-16']

    # 定義Lidar參數
    lidar_x_arg = DeclareLaunchArgument(
        'lidar_x', default_value=str(lidar_pos['position']['x']))
    lidar_y_arg = DeclareLaunchArgument(
        'lidar_y', default_value=str(lidar_pos['position']['y']))
    lidar_z_arg = DeclareLaunchArgument(
        'lidar_z', default_value=str(lidar_pos['position']['z']))
    lidar_roll_arg = DeclareLaunchArgument(
        'lidar_roll', default_value=str(lidar_pos['orientation']['roll_deg']))
    lidar_pitch_arg = DeclareLaunchArgument(
        'lidar_pitch', default_value=str(lidar_pos['orientation']['pitch_deg']))
    lidar_yaw_arg = DeclareLaunchArgument(
        'lidar_yaw', default_value=str(lidar_pos['orientation']['yaw_deg']))

    # 引用robot_description.launch.py並傳遞Lidar參數
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('agri_bot'),
                'launch',
                'robot_description.launch.py'
            ])
        ]),
        launch_arguments={
            'lidar_x': LaunchConfiguration('lidar_x'),
            'lidar_y': LaunchConfiguration('lidar_y'),
            'lidar_z': LaunchConfiguration('lidar_z'),
            'lidar_roll_deg': LaunchConfiguration('lidar_roll_deg'),
            'lidar_pitch_deg': LaunchConfiguration('lidar_pitch_deg'),
            'lidar_yaw_deg': LaunchConfiguration('lidar_yaw_deg')
        }.items()
    )

    # 啟動VLP-16 launch
    velodyne_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('velodyne'),
                'launch',
                'velodyne-all-nodes-VLP16-launch.py'
            ])
        ]),
    )

    return LaunchDescription([
        lidar_x_arg,
        lidar_y_arg,
        lidar_z_arg,
        lidar_roll_arg,
        lidar_pitch_arg,
        lidar_yaw_arg,
        robot_description_launch,
        velodyne_launch,
    ])
