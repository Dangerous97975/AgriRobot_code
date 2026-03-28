from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    # 取得 'farm_navigation' 套件的分享目錄路徑，並組合出設定檔 (param.yaml) 的絕對路徑
    # 集中管理參數：這個 YAML 檔案將會被底下兩個節點共同讀取，確保演算法與控制器的參數設定保持一致
    config_file = os.path.join(
        get_package_share_directory('farm_navigation'),
        'config',
        'param.yaml'
    )

    # 宣告第一個 ROS 2 節點：視覺導航線偵測器 (Line Detector)
    # 功能推演：執行 'visual_nav_line_detector_ridge'，通常負責處理攝影機影像，辨識農田中的田壟 (ridge) 或作物的導航特徵線
    start_navigation_line_detector_cmd = Node(
        package='farm_navigation',                    # 節點所屬的套件名稱
        executable='visual_nav_line_detector_ridge',  # 實際執行的程式檔 (可執行檔) 名稱
        name='visual_nav_line_detector',              # 系統中運行時的節點名稱 (Node Name)
        parameters=[config_file],                     # 載入上方定義的 param.yaml 參數設定檔
        output='screen'                               # 將此節點的標準輸出 (Log, Info, Error) 顯示在終端機上
    )

    # 宣告第二個 ROS 2 節點：視覺導航控制器 (Navigation Controller)
    # 功能推演：執行 'visual_nav_controller'，通常負責接收偵測器發布的特徵線資訊，計算出速度與轉向的控制指令 (cmd_vel) 發送給底盤
    start_visual_nav_controller_cmd = Node(
        package='farm_navigation',                    # 節點所屬的套件名稱
        executable='visual_nav_controller',           # 實際執行的程式檔 (可執行檔) 名稱
        name='visual_navigation_controller',          # 系統中運行時的節點名稱 (Node Name)
        parameters=[config_file],                     # 載入同一份 param.yaml 參數設定檔
        output='screen'                               # 將此節點的標準輸出顯示在終端機上
    )

    # 建立 LaunchDescription 物件，這是裝載所有啟動命令 (Actions) 的容器
    ld = LaunchDescription()

    # 將剛才宣告好的兩個節點依序加入啟動清單中
    # ROS 2 會在執行這個 Launch 檔時，非同步將它們喚醒
    ld.add_action(start_navigation_line_detector_cmd)
    ld.add_action(start_visual_nav_controller_cmd)

    # 回傳完整的 Launch 描述，交給 ROS 2 系統執行
    return ld
