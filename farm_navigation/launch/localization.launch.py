import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# College_of_Engineering_3F
# greenhouse_map
# No3_greenhouse_map


def generate_launch_description():
    rtabmap_launch_dir = get_package_share_directory('rtabmap_launch')

    pkg_share = get_package_share_directory('farm_navigation')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    rviz2_config_dir = os.path.join(
        nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz')

    # 取得參數
    namespace = LaunchConfiguration('namespace')
    map_yaml_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')

    # 宣告參數
    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')

    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(
            pkg_share, 'maps', 'greenhouse_map.yaml'),
        description='Full path to map yaml file to load')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_share, 'config', 'amcl_config.yaml'),
        description='Full path to the ROS 2 parameters file to use')

    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    # 開啟 map_server
    start_map_server_cmd = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace=namespace,
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'yaml_filename': map_yaml_file}],
    )

    # 開啟 amcl
    start_amcl_cmd = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        namespace=namespace,
        output='screen',
        parameters=[params_file],
    )

    # start_rtabmap_icp = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(rtabmap_launch_dir,
    #                      'launch', 'icp_odom.launch.py')),
    #     launch_arguments={
    #         'use_sim_time': 'false',
    #         'autostart': 'true',
    #     }.items()
    # )

    # 開啟 lifecycle_manager
    start_lifecycle_manager_cmd = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        namespace=namespace,
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': autostart},
                    {'node_names': ['map_server', 'amcl']}])

    start_rviz2_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{
                'use_sim_time': use_sim_time,
                'yaml_filename': params_file
        }],
        arguments=['-d', rviz2_config_dir]
    )

    # 創建並返回 launch description
    ld = LaunchDescription()

    # 宣告 launch 參數
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_map_yaml_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_params_file_cmd)
    # ld.add_action(start_rtabmap_icp)

    # 添加節點
    ld.add_action(start_map_server_cmd)
    ld.add_action(start_amcl_cmd)
    ld.add_action(start_lifecycle_manager_cmd)
    ld.add_action(start_rviz2_cmd)

    return ld
