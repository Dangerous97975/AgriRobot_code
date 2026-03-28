from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('farm_navigation')

    config_file = os.path.join(
        get_package_share_directory('farm_navigation'),
        'config',
        'param.yaml'
    )

    start_fsm_controller = Node(
        package='farm_navigation',
        executable='nav_fsm_controller',
        name='fsm_controller',
        output='screen',
        parameters=[config_file],
    )

    start_in_out_row_detector = Node(
        package='farm_navigation',
        executable='in_out_row_detector',
        name='in_out_row_detector',
        output='screen',
        parameters=[config_file],
    )

    visual_nav_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'visual_nav_system.launch.py'))
    )

    ld = LaunchDescription()
    ld.add_action(start_fsm_controller)
    ld.add_action(start_in_out_row_detector)
    ld.add_action(visual_nav_system)

    return ld
