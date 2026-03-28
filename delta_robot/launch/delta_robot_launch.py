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

    camera_ext = os.path.join(
        get_package_share_directory('agri_bot'),
        'config',
        'camera_extrinsics.yaml'
    )

    delta_working_level_arg = DeclareLaunchArgument(
        'delta_working_level',
        default_value='[-550.0, -590.0, -550.0]',
        description='Delta手臂工作高度'
    )

    # delta_robot_api = Node(
    #     package='delta_robot',
    #     executable='delta_robot_api',
    #     name="delta_robot_api",
    #     output='screen'
    # )

    trajectory_plan_node = Node(
        package='delta_robot',
        executable='trajectory_plan',
        name="trajectory_plan",
        parameters=[
            {'delta_working_level': LaunchConfiguration(
                'delta_working_level')},
            camera_inner,  # 直接傳遞檔案路徑
            camera_ext     # 直接傳遞檔案路徑
        ],
        output='screen'
    )

    launch_description = LaunchDescription([
        delta_working_level_arg,
        # delta_robot_api,
        trajectory_plan_node
    ])

    return launch_description
