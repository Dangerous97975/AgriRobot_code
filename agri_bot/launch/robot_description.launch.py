# robot_description.launch.py (在agri_bot套件中)
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


'''
D455 90度朝下
camera_x: 0
camera_z: 0.17
camera_pitch_deg: 90

D455 45斜視
camera_x: 0.32
camera_z: 0.26 # 0.26+0.58=0.84
camera_pitch_deg: 40
'''


def generate_launch_description():
    # 獲取包路徑
    pkg_share = get_package_share_directory('agri_bot')

    # 定義相機參數
    camera_x_arg = DeclareLaunchArgument(
        'camera_x', default_value='-0.11',
        description='Camera x position in meters'
    )
    camera_y_arg = DeclareLaunchArgument(
        'camera_y', default_value='0.0',
        description='Camera y position in meters'
    )
    camera_z_arg = DeclareLaunchArgument(
        'camera_z', default_value='0.17',
        description='Camera height in meters'
    )
    camera_roll_arg = DeclareLaunchArgument(
        'camera_roll_deg', default_value='0.0',
        description='Camera roll angle in degrees'
    )
    camera_pitch_arg = DeclareLaunchArgument(
        'camera_pitch_deg', default_value='90.0',
        description='Camera pitch angle in deg rees'
    )  # 45
    camera_yaw_arg = DeclareLaunchArgument(
        'camera_yaw_deg', default_value='0.0',
        description='Camera yaw angle in degrees'
    )

    # 定義VLP-16 LiDAR參數
    lidar_x_arg = DeclareLaunchArgument(
        'lidar_x', default_value='0.0',
        description='Lidar x position in meters'
    )
    lidar_y_arg = DeclareLaunchArgument(
        'lidar_y', default_value='0.0',
        description='Lidar y position in meters'
    )
    lidar_z_arg = DeclareLaunchArgument(
        'lidar_z', default_value='0.87',
        description='Lidar height in meters'
    )
    lidar_roll_arg = DeclareLaunchArgument(
        'lidar_roll_deg', default_value='0.0',
        description='Lidar roll angle in degrees'
    )
    lidar_pitch_arg = DeclareLaunchArgument(
        'lidar_pitch_deg', default_value='0.0',
        description='Lidar pitch angle in degrees'
    )
    lidar_yaw_arg = DeclareLaunchArgument(
        'lidar_yaw_deg', default_value='0.0',
        description='Lidar yaw angle in degrees'
    )

    # 設置XACRO文件路徑
    xacro_file = os.path.join(pkg_share, 'urdf', 'agri_bot.urdf.xacro')

    # 創建robot_description參數命令
    robot_description_command = Command([
        'xacro ', xacro_file,
        ' camera_x:=', LaunchConfiguration('camera_x'),
        ' camera_y:=', LaunchConfiguration('camera_y'),
        ' camera_z:=', LaunchConfiguration('camera_z'),
        ' camera_roll_deg:=', LaunchConfiguration('camera_roll_deg'),
        ' camera_pitch_deg:=', LaunchConfiguration('camera_pitch_deg'),
        ' camera_yaw_deg:=', LaunchConfiguration('camera_yaw_deg'),
        ' lidar_x:=', LaunchConfiguration('lidar_x'),
        ' lidar_y:=', LaunchConfiguration('lidar_y'),
        ' lidar_z:=', LaunchConfiguration('lidar_z'),
        ' lidar_roll_deg:=', LaunchConfiguration('lidar_roll_deg'),
        ' lidar_pitch_deg:=', LaunchConfiguration('lidar_pitch_deg'),
        ' lidar_yaw_deg:=', LaunchConfiguration('lidar_yaw_deg')
    ])

    # 啟動 robot_state_publisher 節點
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_command}]
    )

    # 啟動 joint_state_publisher (如果需要的話)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen'
    )

    return LaunchDescription([
        camera_x_arg,
        camera_y_arg,
        camera_z_arg,
        camera_roll_arg,
        camera_pitch_arg,
        camera_yaw_arg,
        lidar_x_arg,
        lidar_y_arg,
        lidar_z_arg,
        lidar_roll_arg,
        lidar_pitch_arg,
        lidar_yaw_arg,
        robot_state_publisher,
        joint_state_publisher
    ])
