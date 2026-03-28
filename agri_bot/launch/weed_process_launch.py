from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    plant_detection_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory(
                'plant_detection'), '/launch/', 'plant_detection_launch.py'
        ])
    )

    delta_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory(
                'delta_robot'), '/launch/', 'delta_robot_launch.py'
        ]),
    )

    weed_control_node = Node(
        package='delta_robot',
        executable='weeding_operation',
        name='weeding_operation',
        output='screen'
    )

    launch_description = LaunchDescription([
        plant_detection_launch,
        delta_robot_launch,
        weed_control_node
    ])

    return launch_description
