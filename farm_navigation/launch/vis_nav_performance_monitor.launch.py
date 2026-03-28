from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():

    # 聲明 launch 參數
    performance_monitor_arg = DeclareLaunchArgument(
        'enable_performance_monitor',
        default_value='true',
        description='是否啟用性能監控器'
    )

    # 視覺導航線檢測節點
    vis_nav_node = Node(
        package='farm_navigation',
        executable='vis_nav_line_detector_ridge',
        name='vis_nav_line_detector',
        output='screen',
        parameters=[
            {'voxel_size': 0.01},
            {'slice_thickness': 0.05},
            {'slice_minimum': -0.42},
            {'slice_maximum': 0.23},
            {'y_minimum': -0.55},
            {'y_maximum': 0.55},
            {'window_width': 0.8},
            {'window_step': 0.025}
        ]
    )

    # 性能監控器節點 (條件啟動)
    performance_monitor_node = Node(
        package='farm_navigation',
        executable='performance_monitor',
        name='performance_monitor',
        output='screen',
        condition=IfCondition(LaunchConfiguration(
            'enable_performance_monitor'))
    )

    return LaunchDescription([
        performance_monitor_arg,
        LogInfo(
            msg="啟動視覺導航性能監控系統"
        ),
        vis_nav_node,
        performance_monitor_node,
        LogInfo(
            msg=[
                "系統已啟動，使用以下服務控制:\n",
                "  - 啟用視覺導航: ros2 service call /vision_navigation_control std_srvs/srv/SetBool '{data: true}'\n",
                "  - 停用視覺導航: ros2 service call /vision_navigation_control std_srvs/srv/SetBool '{data: false}'\n",
                "性能數據將在節點日誌中顯示"
            ]
        )
    ])
