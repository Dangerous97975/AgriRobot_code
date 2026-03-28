import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_share = get_package_share_directory('farm_navigation')
    ekf_config_path = os.path.join(
        pkg_share, 'config', 'ekf_config.yaml')

    return LaunchDescription([
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_config_path],  # 調整路徑
            remappings=[
                ('/odometry/filtered', '/odom_filtered'),
            ]
        )
    ])
