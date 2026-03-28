from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    # 取得套件 (package) 的分享目錄路徑
    agri_bot_pkg = get_package_share_directory('agri_bot')
    rtabmap_launch_dir = get_package_share_directory('rtabmap_launch')

    # 設定設定檔 (config) 的絕對路徑
    # sensor_positions.yaml: 存放感測器在機器人上的相對位置與姿態
    sensor_position_path = os.path.join(
        agri_bot_pkg, 'config', 'sensor_positions.yaml')
    # realsense_param.json: 存放 RealSense 相機的進階硬體參數設定檔
    realsense_param_path = os.path.join(
        agri_bot_pkg, 'config', 'realsense_param.json')

    # 讀取 YAML 檔案中的感測器位置資訊
    with open(sensor_position_path, 'r') as file:
        sensor_positions = yaml.safe_load(file)

    # 分別取得 LiDAR (VLP-16) 與 相機 (D455f) 的位置與姿態資料
    lidar_pos = sensor_positions['VLP-16']
    d455f_pos = sensor_positions['D455f']

    # ==========================================
    # 宣告 Launch 參數 (Launch Arguments)
    # 這些參數可以在終端機執行 launch 指令時被覆寫 (例如：ros2 launch ... lidar_x:=1.0)
    # 預設值皆來自剛剛讀取的 YAML 檔案
    # ==========================================
    
    # LiDAR (VLP-16) 的 X, Y, Z 平移參數
    lidar_x_arg = DeclareLaunchArgument(
        'lidar_x', default_value=str(lidar_pos['position']['x']))
    lidar_y_arg = DeclareLaunchArgument(
        'lidar_y', default_value=str(lidar_pos['position']['y']))
    lidar_z_arg = DeclareLaunchArgument(
        'lidar_z', default_value=str(lidar_pos['position']['z']))
    # LiDAR (VLP-16) 的 Roll, Pitch, Yaw 旋轉參數 (單位：度)
    lidar_roll_arg = DeclareLaunchArgument(
        'lidar_roll', default_value=str(lidar_pos['orientation']['roll_deg']))
    lidar_pitch_arg = DeclareLaunchArgument(
        'lidar_pitch', default_value=str(lidar_pos['orientation']['pitch_deg']))
    lidar_yaw_arg = DeclareLaunchArgument(
        'lidar_yaw', default_value=str(lidar_pos['orientation']['yaw_deg']))

    # 深度相機 (D455f) 的 X, Y, Z 平移參數
    d455f_x_arg = DeclareLaunchArgument(
        'd455f_x', default_value=str(d455f_pos['position']['x']))
    d455f_y_arg = DeclareLaunchArgument(
        'd455f_y', default_value=str(d455f_pos['position']['y']))
    d455f_z_arg = DeclareLaunchArgument(
        'd455f_z', default_value=str(d455f_pos['position']['z']))
    # 深度相機 (D455f) 的 Roll, Pitch, Yaw 旋轉參數 (單位：度)
    d455f_roll_arg = DeclareLaunchArgument(
        'd455f_roll', default_value=str(d455f_pos['orientation']['roll_deg']))
    d455f_pitch_arg = DeclareLaunchArgument(
        'd455f_pitch', default_value=str(d455f_pos['orientation']['pitch_deg']))
    d455f_yaw_arg = DeclareLaunchArgument(
        'd455f_yaw', default_value=str(d455f_pos['orientation']['yaw_deg']))

    # ==========================================
    # 載入其他 Launch 檔案 (Include Launch Descriptions)
    # ==========================================

    # 1. 啟動機器人描述檔 (Robot Description)
    # 負責發布機器人的 URDF/Xacro 模型，建立 TF (座標系統) 樹
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('agri_bot'),
                'launch',
                'robot_description.launch.py'
            ])
        ]),
        # 將上方宣告的感測器位置參數傳遞給 robot_description.launch.py
        launch_arguments={
            'lidar_x': LaunchConfiguration('lidar_x'),
            'lidar_y': LaunchConfiguration('lidar_y'),
            'lidar_z': LaunchConfiguration('lidar_z'),
            'lidar_roll_deg': LaunchConfiguration('lidar_roll'),
            'lidar_pitch_deg': LaunchConfiguration('lidar_pitch'),
            'lidar_yaw_deg': LaunchConfiguration('lidar_yaw'),
            'camera_x': LaunchConfiguration('d455f_x'),
            'camera_y': LaunchConfiguration('d455f_y'),
            'camera_z': LaunchConfiguration('d455f_z'),
            'camera_roll_deg': LaunchConfiguration('d455f_roll'),
            'camera_pitch_deg': LaunchConfiguration('d455f_pitch'),
            'camera_yaw_deg': LaunchConfiguration('d455f_yaw')
        }.items()
    )

    # 2. 感測器啟動檔
    # 啟動 Velodyne VLP-16 LiDAR 驅動節點
    velodyne_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('velodyne'),
                'launch',
                'velodyne-all-nodes-VLP16-launch.py'
            ])
        ]),
    )

    # 3. 啟動 RealSense 深度相機驅動節點
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ]),
        # 設定 RealSense 的各項參數
        launch_arguments={
            'camera_name': 'D455f',                    # 命名相機節點
            'camera_namespace': 'agri_bot',            # 設定相機的命名空間 (namespace)
            'json_file_path': realsense_param_path,    # 匯入額外的硬體設定檔 (JSON)
            'initial_reset': 'true',                   # 啟動時先重置相機硬體 (避免卡死)
            'enable_color': 'true',                    # 啟用彩色影像流
            'enable_depth': 'true',                    # 啟用深度影像流
            'depth_module.depth_profile': "480x270x15",# 設定深度影像解析度與幀率 (寬x高xFPS)
            'depth_module.enable_auto_exposure': 'false', # 關閉深度相機自動曝光 (這裡寫了兩次，下方還有一行)
            'depth_module.exposure': '1000',           # 手動設定深度曝光值
            'pointcloud.enable': 'true',               # 啟用點雲 (Pointcloud) 生成
            'pointcloud.allow_no_texture_points': 'false', # 不允許沒有紋理(顏色)的點雲
            'pointcloud.texture_stream': 'any',        # 點雲紋理來源 (設為任意)
            'align_depth.enable': 'true',              # 啟用深度圖與彩色圖對齊 (Align Depth to Color)
            'clip_distance': '5.14'                    # 設定深度截斷距離 (超過 5.14 公尺的深度資訊將被忽略)
        }.items()
    )

    # 4. 啟動 RTAB-Map 的 ICP (Iterative Closest Point) 里程計
    # 透過 LiDAR 點雲進行特徵匹配來推算機器人位移
    start_rtabmap_icp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rtabmap_launch_dir,
                         'launch', 'icp_odom.launch.py')),
        launch_arguments={
            'use_sim_time': 'false', # 使用真實時間 (非模擬環境)
            'autostart': 'true',     # 自動啟動節點
        }.items()
    )

    # 5. Delta 機器手臂的 API 節點 (目前已被註解停用)
    # delta_robot_api = Node(
    #     package='delta_robot',
    #     executable='delta_robot_api',
    #     name="delta_robot_api",
    #     output='screen' # 將節點輸出的資訊印在終端機畫面上
    # )

    # ==========================================
    # 建立 LaunchDescription 物件並加入所有 Actions
    # ==========================================
    ld = LaunchDescription()

    # 加入所有參數宣告 (讓終端機可以看見並使用這些參數)
    ld.add_action(lidar_x_arg)
    ld.add_action(lidar_y_arg)
    ld.add_action(lidar_z_arg)
    ld.add_action(lidar_roll_arg)
    ld.add_action(lidar_pitch_arg)
    ld.add_action(lidar_yaw_arg)
    ld.add_action(d455f_x_arg)
    ld.add_action(d455f_y_arg)
    ld.add_action(d455f_z_arg)
    ld.add_action(d455f_roll_arg)
    ld.add_action(d455f_pitch_arg)
    ld.add_action(d455f_yaw_arg)

    # 加入所有要執行的子 Launch 檔與節點
    ld.add_action(robot_description_launch) # 啟動機器人模型 TF
    ld.add_action(velodyne_launch)          # 啟動 LiDAR
    ld.add_action(realsense_launch)         # 啟動相機
    ld.add_action(start_rtabmap_icp)        # 啟動 ICP 里程計

    # ld.add_action(delta_robot_api)        # (被註解的 Delta 手臂 API)

    # 回傳完整的 Launch 描述，交由 ROS 2 執行
    return ld