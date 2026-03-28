from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    camera_inner = os.path.join(
        get_package_share_directory('agri_bot'),
        'config',
        'D455f_inner.yaml'
    )

    deduplication_threshold_arg = DeclareLaunchArgument(
        'deduplication_threshold',
        default_value='0.1',
        description='去重距離閾值 (公尺)'
    )

    plant_distribution_mapping_node = Node(
        package='plant_detection',
        executable='plant_distribution_mapping',
        name="plant_distribution_mapping",
        parameters=[
            camera_inner,
            {'deduplication_threshold': LaunchConfiguration(
                'deduplication_threshold'
            )}  # 去重距離閾值
        ],
        output='screen'
    )

    return LaunchDescription([
        deduplication_threshold_arg,
        plant_distribution_mapping_node
    ])
