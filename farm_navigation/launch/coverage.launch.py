import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 獲取包路徑
    pkg_share = get_package_share_directory('farm_navigation')
    maps_dir = os.path.join(pkg_share, 'maps')

    # 聲明參數
    map_yaml_file = LaunchConfiguration('map')

    # 參數聲明
    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(maps_dir, 'greenhouse_map.yaml'),
        description='Full path to map yaml file'
    )

    # 地圖服務器節點
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml_file}],
        remappings=[
            ('/map', 'map'),
        ])

    # 靜態TF發布器（如果您沒有真實的機器人）
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link'])

    # 生命週期管理器（只管理地圖服務器）
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {'use_sim_time': False},
            {'autostart': True},
            {'node_names': ['map_server']}
        ])

    # RViz
    rviz_config_file = os.path.join(pkg_share, 'config', 'ccpp_view.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file])

    # CCPP路徑生成節點
    ccpp_generator_node = Node(
        package='farm_navigation',
        executable='ccpp_generator',
        name='ccpp_generator',
        output='screen',
        parameters=[{
            'field_length': 26.0,
            'field_width': 12.0,
            'no_go_zone': 4.0,
            'lane_width': 1.2,
            'safety_margin': 0.5
        }])

    return LaunchDescription([
        declare_map_yaml_cmd,

        # 啟動順序很重要
        map_server_node,
        static_tf_node,
        lifecycle_manager_node,

        # 稍微延遲啟動RViz和CCPP生成器
        LogInfo(msg="Waiting for map server to start..."),
        rviz_node,
        ccpp_generator_node
    ])
