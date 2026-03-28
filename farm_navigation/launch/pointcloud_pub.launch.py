"""
!! 測試用
用於發佈現有的點雲檔到ROS2，實驗室測試用
"""


from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # 宣告參數
        DeclareLaunchArgument(
            'ply_file_path',
            default_value='',
            description='PLY 點雲檔案路徑'
        ),

        DeclareLaunchArgument(
            'frame_id',
            default_value='map',
            description='點雲座標系統'
        ),

        DeclareLaunchArgument(
            'publish_period',
            default_value='1.0',
            description='點雲發布頻率(秒)'
        ),

        # 啟動點雲發布節點
        Node(
            package='farm_navigation',
            executable='pointcloud_pub_test',
            name='pointcloud_publisher',
            parameters=[{
                'ply_file_path': LaunchConfiguration('ply_file_path'),
                'frame_id': LaunchConfiguration('frame_id'),
                'publish_period': LaunchConfiguration('publish_period')
            }],
            output='screen'
        )
    ])
