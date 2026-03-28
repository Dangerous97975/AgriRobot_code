# 視覺深度導航線v2
#此程式碼用於農業機器人的視覺導航，透過深度相機的點雲數據檢測土畦（Ridge），
#並生成導航線（Navigation Line）以控制機器人沿著土畦行駛。

import rclpy
from rclpy.node import Node
import tf2_ros
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped, Vector3
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor, LinearRegression
import os
import time


class VisNavLineDetector(Node):
    """
    視覺導航線檢測節點
    功能：
    1. 訂閱深度相機的點雲數據
    2. 過濾並轉換點雲座標系
    3. 使用滑動窗口算法檢測土畦中心
    4. 使用 RANSAC 算法擬合導航線
    5. 發佈導航路徑 (Pose) 與視覺化標記 (Markers)
    """
    def __init__(self):
        super().__init__('vis_nav_line_detector')

        # 初始化 TF 監聽器，用於座標轉換 (camera_link -> base_footprint)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.target_frame = "base_footprint"  # 目標座標系：機器人底盤中心

        # 初始化參數
        self.declare_custom_parameters()
        self.get_parameters()

        # 狀態標誌
        self.visualize = False
        self.vision_navigation_enabled = False  # 預設不啟用導航計算，需透過 Service 開啟

        # 建立服務：控制視覺導航的開/關
        self.vision_nav_service = self.create_service(
            SetBool,
            '/vision_navigation_control',
            self.vision_navigation_callback
        )

        # 訂閱：深度相機點雲
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/agri_bot/D455f/depth/color/points',
            self.pointcloud_callback,
            10
        )

        # 發佈：畦溝/土畦點的視覺化標記
        self.furrow_points_pub = self.create_publisher(
            MarkerArray,
            '/furrow_points_markers',
            10
        )

        # 發佈：導航線與相關標記
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/navigation_markers',
            10
        )

        # 發佈：最終計算出的導航位姿 (位置與航向)
        self.navigation_pose_pub = self.create_publisher(
            PoseStamped,
            '/navigation_pose',
            10
        )

        # 發佈：導航系統狀態
        self.navigation_status_pub = self.create_publisher(
            Bool,
            '/vision_navigation_status',
            10
        )

        # 初始化效能監控字典：用於記錄各階段處理時間
        self.processing_times = {
            'total': [],                  # 總處理時間
            'tf_transform': [],           # TF座標轉換時間
            'point_processing': [],       # 點雲前處理時間
            'sliding_window': [],         # 滑動窗口算法時間
            'navigation_generation': [],  # 線性擬合生成導航線時間
            'visualization': []           # Rviz 標記發佈時間
        }

        # 初始化點雲統計字典：監控數據量變化
        self.pointcloud_stats = {
            'original_points': [],  # 原始點數
            'filtered_points': [],  # 範圍過濾後點數
            'processed_points': []  # 座標轉換後實際處理點數
        }

        self.get_logger().info('Navigation Line Detector Node initialized')

    def declare_custom_parameters(self):
        """宣告ROS2參數server參數 (可透過 yaml 或 rqt 動態調整)"""
        self.declare_parameter('voxel_size', 0.01)   # 體素濾波大小 1cm (降採樣用)
        self.declare_parameter('slice_thickness', 0.05)  # 切片厚度 5cm (X軸方向切片)
        self.declare_parameter('slice_minimum', -0.42)   # ROI X軸最小值 (機器人前方/後方範圍)
        self.declare_parameter('slice_maximum', 0.23)    # ROI X軸最大值
        self.declare_parameter('y_minimum', -0.55)  # ROI Y軸最小值 (左右寬度)
        self.declare_parameter('y_maximum', 0.55)   # ROI Y軸最大值
        self.declare_parameter('window_width', 0.8)  # 滑動窗口寬度 80 cm (用於直方圖分析)
        self.declare_parameter('window_step', 0.025)   # 滑動步長 2.5cm

    def get_parameters(self):
        """從ROS參數server抓取參數並存入變數"""
        self.voxel_size = self.get_parameter('voxel_size').value
        self.slice_thickness = self.get_parameter('slice_thickness').value
        self.slice_minimum = self.get_parameter('slice_minimum').value
        self.slice_maximum = self.get_parameter('slice_maximum').value
        self.y_minimum = self.get_parameter('y_minimum').value
        self.y_maximum = self.get_parameter('y_maximum').value
        self.window_width = self.get_parameter('window_width').value
        self.step = self.get_parameter('window_step').value

    def vision_navigation_callback(self, request, response):
        """
        視覺導航控制服務回調函數
        處理來自外部的開關請求 (SetBool)
        """
        try:
            if request.data:  # 啟用視覺導航
                self.vision_navigation_enabled = True
                self.get_logger().info('視覺導航已啟用')
                response.success = True
                response.message = "視覺導航已啟用"
            else:  # 關閉視覺導航
                self.vision_navigation_enabled = False
                self.get_logger().info('視覺導航已關閉')
                response.success = True
                response.message = "視覺導航已關閉"

            # 發佈當前狀態
            status_msg = Bool()
            status_msg.data = request.data  # True=啟用, False=停用
            self.navigation_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'視覺導航控制服務錯誤: {str(e)}')
            response.success = False
            response.message = f"服務錯誤: {str(e)}"

        return response

    def _log_processing_times(self):
        """
        記錄並輸出處理時間統計
        計算最近50次的平均值、最大最小值與標準差，用於系統效能診斷
        """
        def get_stats(times_list):
            if not times_list:
                return {'mean': 0, 'min': 0, 'max': 0, 'std': 0}
            times = np.array(times_list[-50:])  # 只統計最近50次
            return {
                'mean': np.mean(times) * 1000,  # 轉換為毫秒 (ms)
                'min': np.min(times) * 1000,
                'max': np.max(times) * 1000,
                'std': np.std(times) * 1000
            }

        # 獲取各階段的統計數據
        total_stats = get_stats(self.processing_times['total'])
        tf_stats = get_stats(self.processing_times['tf_transform'])
        point_stats = get_stats(self.processing_times['point_processing'])
        sliding_stats = get_stats(self.processing_times['sliding_window'])
        nav_stats = get_stats(self.processing_times['navigation_generation'])
        viz_stats = get_stats(self.processing_times['visualization'])

        # 點雲數據統計
        def get_point_stats(point_list):
            if not point_list:
                return {'mean': 0, 'min': 0, 'max': 0}
            points = np.array(point_list[-50:])  # 最近50次
            return {
                'mean': int(np.mean(points)),
                'min': int(np.min(points)),
                'max': int(np.max(points))
            }

        original_point_stats = get_point_stats(
            self.pointcloud_stats['original_points'])
        filtered_point_stats = get_point_stats(
            self.pointcloud_stats['filtered_points'])
        processed_point_stats = get_point_stats(
            self.pointcloud_stats['processed_points'])

        # 計算過濾效率 (剔除多少無效點)
        filter_ratio = 0
        if original_point_stats['mean'] > 0:
            filter_ratio = (
                1 - filtered_point_stats['mean'] / original_point_stats['mean']) * 100

        # 輸出詳細日誌
        self.get_logger().info(
            f"========== 處理時間統計 (最近50次) ==========\n"
            f"總處理時間:     平均 {total_stats['mean']:.2f}ms, "
            f"最小 {total_stats['min']:.2f}ms, 最大 {total_stats['max']:.2f}ms, "
            f"標準差 {total_stats['std']:.2f}ms\n"
            f"TF轉換時間:     平均 {tf_stats['mean']:.2f}ms, "
            f"最小 {tf_stats['min']:.2f}ms, 最大 {tf_stats['max']:.2f}ms, "
            f"標準差 {tf_stats['std']:.2f}ms\n"
            f"點雲處理時間:   平均 {point_stats['mean']:.2f}ms, "
            f"最小 {point_stats['min']:.2f}ms, 最大 {point_stats['max']:.2f}ms, "
            f"標準差 {point_stats['std']:.2f}ms\n"
            f"滑動窗口時間:   平均 {sliding_stats['mean']:.2f}ms, "
            f"最小 {sliding_stats['min']:.2f}ms, 最大 {sliding_stats['max']:.2f}ms, "
            f"標準差 {sliding_stats['std']:.2f}ms\n"
            f"導航生成時間:   平均 {nav_stats['mean']:.2f}ms, "
            f"最小 {nav_stats['min']:.2f}ms, 最大 {nav_stats['max']:.2f}ms, "
            f"標準差 {nav_stats['std']:.2f}ms\n"
            f"視覺化時間:     平均 {viz_stats['mean']:.2f}ms, "
            f"最小 {viz_stats['min']:.2f}ms, 最大 {viz_stats['max']:.2f}ms, "
            f"標準差 {viz_stats['std']:.2f}ms\n"
            f"==========================================\n"
            f"點雲數據統計:\n"
            f"  原始點數:     平均 {original_point_stats['mean']}, "
            f"最小 {original_point_stats['min']}, 最大 {original_point_stats['max']}\n"
            f"  過濾後點數:   平均 {filtered_point_stats['mean']}, "
            f"最小 {filtered_point_stats['min']}, 最大 {filtered_point_stats['max']}\n"
            f"  處理點數:     平均 {processed_point_stats['mean']}, "
            f"最小 {processed_point_stats['min']}, 最大 {processed_point_stats['max']}\n"
            f"  過濾效率:     {filter_ratio:.1f}% 點被過濾掉\n"
            f"==========================================\n"
            f"處理頻率: {1000/total_stats['mean']:.1f} Hz"
        )

    def pointcloud_callback(self, msg):
        """
        核心回調函數：處理接收到的點雲數據
        流程：
        1. 檢查是否啟用
        2. TF 轉換查詢
        3. 點雲讀取與ROI過濾 (使用 NumPy 向量化加速)
        4. 座標轉換 (Camera -> Base)
        5. 土畦檢測 (滑動窗口)
        6. 導航線生成 (RANSAC)
        7. 視覺化發佈
        """
        if not self.vision_navigation_enabled:
            return

        start_total = time.time()

        try:
            # 1. TF轉換時間測量 (獲取當下從相機到基底的變換矩陣)
            start_tf = time.time()
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0))
            end_tf = time.time()
            tf_time = end_tf - start_tf

            # 2. 點雲處理時間測量
            start_point_processing = time.time()

            # 優化：使用向量化處理替代 Python 循環讀取點雲
            points_xyz = np.array(list(pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)))
            original_count = len(points_xyz)

            # 初步過濾：移除明顯超出範圍的點 (減少後續運算負擔)
            if len(points_xyz) > 0:
                # 定義感興趣區域 (ROI)
                valid_mask = (
                    (points_xyz[:, 0] >= self.slice_minimum - 0.1) &
                    (points_xyz[:, 0] <= self.slice_maximum + 0.1) &
                    (points_xyz[:, 1] >= self.y_minimum - 0.1) &
                    (points_xyz[:, 1] <= self.y_maximum + 0.1) &
                    (points_xyz[:, 2] >= -1.0) &  # 移除明顯錯誤的 Z 值 (過低或過高)
                    (points_xyz[:, 2] <= 2.0)
                )
                points_xyz = points_xyz[valid_mask]
                filtered_count = len(points_xyz)

                # 添加齊次座標 (Homogeneous coordinates) 用於矩陣乘法 [x, y, z, 1]
                points_homo = np.ones((len(points_xyz), 4))
                points_homo[:, :3] = points_xyz
                points = points_homo
            else:
                filtered_count = 0
                points = np.array([]).reshape(0, 4)

            # 應用 TF 變換矩陣
            TF_points = self._apply_transform(points, transform)
            processed_count = len(TF_points)
            end_point_processing = time.time()
            point_processing_time = end_point_processing - start_point_processing

            new_header = msg.header
            new_header.frame_id = self.target_frame

            # 3. 滑動窗口處理時間測量 (核心算法：檢測土畦中心)
            start_sliding_window = time.time()
            ridge_centers = self._process_sliding_window(TF_points[:, 0:3])
            end_sliding_window = time.time()
            sliding_window_time = end_sliding_window - start_sliding_window

            # 4. 導航線生成時間測量 (擬合土畦中心點)
            start_navigation = time.time()
            navigation_line, heading_angle, lateral_offset = self._generate_navigation_from_ridge(
                ridge_centers)
            end_navigation = time.time()
            navigation_time = end_navigation - start_navigation

            # 5. 發佈航向資訊 (PoseStamped)
            self._publish_navigation_pose(
                navigation_line, heading_angle, lateral_offset, new_header)

            # 6. 視覺化時間測量 (Rviz Markers)
            start_viz = time.time()
            self._publish_ridge_markers(
                ridge_centers, navigation_line, new_header)
            end_viz = time.time()
            viz_time = end_viz - start_viz

            end_total = time.time()
            total_time = end_total - start_total

            # 記錄各階段處理時間
            self.processing_times['total'].append(total_time)
            self.processing_times['tf_transform'].append(tf_time)
            self.processing_times['point_processing'].append(
                point_processing_time)
            self.processing_times['sliding_window'].append(sliding_window_time)
            self.processing_times['navigation_generation'].append(
                navigation_time)
            self.processing_times['visualization'].append(viz_time)

            # 記錄點雲數據統計
            self.pointcloud_stats['original_points'].append(original_count)
            self.pointcloud_stats['filtered_points'].append(filtered_count)
            self.pointcloud_stats['processed_points'].append(processed_count)

            # 每50次處理輸出統計結果
            if len(self.processing_times['total']) % 50 == 0:
                self._log_processing_times()

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as ex:
            self.get_logger().error(f'TF2 轉換錯誤: {str(ex)}')
            return

    def _apply_transform(self, points, transform):
        """
        應用轉換到點雲中的所有點
        數學公式： P' = R * P + T
        使用四元數構建旋轉矩陣
        """
        # 從 transform 中提取旋轉矩陣 (四元數) 和平移向量
        q = transform.transform.rotation
        t = transform.transform.translation

        # 建立 4x4 變換矩陣 (基於四元數轉換公式)
        # R = [r00 r01 r02 0]
        #     [r10 r11 r12 0]
        #     [r20 r21 r22 0]
        #     [tx  ty  tz  1]
        r = np.array([
            [1-2*(q.y*q.y+q.z*q.z), 2*(q.x*q.y-q.z*q.w),
                2*(q.x*q.z+q.y*q.w), 0],
            [2*(q.x*q.y+q.z*q.w), 1-2*(q.x*q.x+q.z*q.z),
                2*(q.y*q.z-q.x*q.w), 0],
            [2*(q.x*q.z-q.y*q.w), 2*(q.y*q.z+q.x*q.w),
                1-2*(q.x*q.x+q.y*q.y), 0],
            [0, 0, 0, 1]
        ])

        # 填入平移向量
        r[0, 3] = t.x
        r[1, 3] = t.y
        r[2, 3] = t.z

        # 應用變換到所有點 (矩陣乘法)
        # points 形狀為 (N, 4), r.T 為 (4, 4), 結果為 (N, 4)
        transformed_points = np.dot(points, r.T)

        return transformed_points

    def _process_sliding_window(self, points):
        """
        使用滑動窗口方法處理點雲
        主要流程：
        1. 嚴格 Y 軸過濾
        2. 體素降採樣 (Voxel Downsampling) 減少點數
        3. 沿 X 軸切片 (Slicing)
        4. 對每個切片執行土畦檢測
        """
        start_time = time.time()

        # 過濾掉左右過遠的點
        y_mask = (points[:, 1] >= self.y_minimum) & (
            points[:, 1] <= self.y_maximum)
        filtered_points = points[y_mask]
        filter_time = time.time() - start_time

        # 使用 Open3D 進行體素降採樣 (加速運算)
        voxel_start = time.time()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # 降採樣：合併相近的點
        pcd_voxel = pcd.voxel_down_sample(self.voxel_size)
        voxel_points = np.asarray(pcd_voxel.points)
        voxel_time = time.time() - voxel_start

        # 獲取過濾後的實際 X 範圍
        actual_x_min = np.min(voxel_points[:, 0])
        actual_x_max = np.max(voxel_points[:, 0])

        # 確保切片範圍不超出實際點雲範圍
        slice_x_min = max(self.slice_minimum, actual_x_min)
        slice_x_max = min(self.slice_maximum, actual_x_max)

        # 計算切片數量 (沿著行進方向 X 軸切分)
        slices = int((self.slice_maximum - self.slice_minimum) /
                     self.slice_thickness)

        ridge_centers = []  # 儲存各切片的土畦中心 [x, y]
        slice_processing_times = []

        # 遍歷每個切片
        slice_loop_start = time.time()
        for slice_idx in range(slices):
            slice_start_time = time.time()

            # 定義當前切片的 X 範圍
            x_start = self.slice_minimum + self.slice_thickness * slice_idx
            x_end = x_start + self.slice_thickness

            # 獲取屬於當前切片的點
            slice_mask = (voxel_points[:, 0] >= x_start) & (
                voxel_points[:, 0] < x_end)
            slice_points = voxel_points[slice_mask]

            if len(slice_points) > 10:
                # 對該切片執行滑動窗口分析，找出土畦中心 Y 值
                ridge_center = self._find_ridge_with_sliding_window(
                    slice_points, slice_idx)

                if ridge_center is not None:
                    ridge_centers.append([x_start, ridge_center])

            slice_time = time.time() - slice_start_time
            slice_processing_times.append(slice_time)

        slice_loop_time = time.time() - slice_loop_start
        total_time = time.time() - start_time

        # 詳細日誌記錄 (每100次輸出一次)
        if len(self.processing_times.get('total', [])) % 100 == 0:
            avg_slice_time = np.mean(
                slice_processing_times) * 1000 if slice_processing_times else 0
            self.get_logger().debug(
                f"滑動窗口詳細時間統計:\n"
                f"  - 點過濾時間: {filter_time*1000:.2f}ms\n"
                f"  - 體素化時間: {voxel_time*1000:.2f}ms\n"
                f"  - 切片處理時間: {slice_loop_time*1000:.2f}ms\n"
                f"  - 平均單切片時間: {avg_slice_time:.2f}ms\n"
                f"  - 處理切片數量: {len(slice_processing_times)}\n"
                f"  - 檢測到的土畦中心點數: {len(ridge_centers)}"
            )

        return np.array(ridge_centers)

    def _find_ridge_with_sliding_window(self, slice_points, slice_id):
        """
        單一切片處理：找出該 X 截面下的土畦中心 Y 座標
        結合高度直方圖方法
        """
        start_time = time.time()

        y_center = 0
        y_coords = slice_points[:, 1]
        z_coords = slice_points[:, 2]

        # 1. 進行 Y 軸排序 (為了後續滑動窗口處理)
        sort_start = time.time()
        sorted_indices = np.argsort(y_coords)
        y_coords = y_coords[sorted_indices]
        z_coords = z_coords[sorted_indices]
        sort_time = time.time() - sort_start

        y_min = np.min(y_coords)
        y_max = np.max(y_coords)

        # 2. 執行高度直方圖分析：計算不同 Y 位置的平均高度
        histogram_start = time.time()
        histogram_centers, histogram_heights = self._identify_features_histogram_ros(
            y_coords, z_coords, slice_id)
        histogram_time = time.time() - histogram_start

        if histogram_centers is None or len(histogram_centers) == 0:
            self.get_logger().warn(f'切片 {slice_id}: 滑動窗口分析失敗')
            return None

        # 3. 找出高度最高的點作為土畦中心
        ridge_start = time.time()
        ridge_center_idx = np.argmax(histogram_heights)
        ridge_center_y = histogram_centers[ridge_center_idx]
        ridge_center_height = histogram_heights[ridge_center_idx]

        # 4. 執行土畦基準方法進行導航點檢測 (獲取精確的導航點)
        furrow_info = self._find_navigation_point_improved(
            histogram_centers, histogram_heights, slice_id,
            ridge_center_y, ridge_center_height, y_coords, z_coords)
        ridge_time = time.time() - ridge_start

        total_time = time.time() - start_time

        # 每200次切片輸出詳細統計 (避免過於頻繁)
        if slice_id % 200 == 0:
            self.get_logger().debug(
                f"切片 {slice_id} 處理時間詳情:\n"
                f"  - 排序時間: {sort_time*1000:.3f}ms\n"
                f"  - 直方圖分析時間: {histogram_time*1000:.3f}ms\n"
                f"  - 土畦檢測時間: {ridge_time*1000:.3f}ms\n"
                f"  - 總時間: {total_time*1000:.3f}ms"
            )

        # 返回導航點 Y 座標
        if furrow_info.get('midpoint') is not None:
            return furrow_info['midpoint']
        elif furrow_info.get('navigation_peak'):
            return furrow_info['navigation_peak']['y']
        else:
            return None

    def _identify_features_histogram_ros(self, y_coords, z_coords, slice_id=0):
        """
        ROS版本的高度直方圖生成
        原理：在 Y 軸上移動一個窗口，計算窗口內 Z 值的平均高度
        """
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        y_center = 0

        histogram_centers = []  # 窗口中心 Y
        histogram_heights = []  # 窗口平均高度 Z

        # 使用80cm窗口進行滑動分析
        window_width = 0.80  # 80cm窗口
        step_size = 0.025    # 2.5cm步進

        # 從中心開始，向左右兩邊滑動計算最大偏移量
        max_offset = max(y_center - y_min, y_max - y_center) - window_width/2

        # 檢查是否有足夠的空間進行滑動
        if max_offset <= 0:
            # 空間不足，退回到使用整個範圍計算平均高度
            window_left = y_min + 0.05
            window_right = y_max - 0.05

            mask = (y_coords >= window_left) & (y_coords <= window_right)
            heights_in_bin = z_coords[mask]

            if len(heights_in_bin) > 5:
                histogram_centers.append(y_center)
                histogram_heights.append(round(np.mean(heights_in_bin), 4))

            return np.array(histogram_centers), np.array(histogram_heights) if histogram_centers else None

        # 處理中心窗口 (Offset = 0)
        window_left = y_center - window_width/2
        window_right = y_center + window_width/2

        mask = (y_coords >= window_left) & (y_coords <= window_right)
        heights_in_bin = z_coords[mask]

        if len(heights_in_bin) > 10:
            histogram_centers.append(y_center)
            histogram_heights.append(round(np.mean(heights_in_bin), 4))

        # 使用滑動窗口策略，左右遍歷
        num_steps = int(2 * max_offset / step_size) + 1
        offsets = np.linspace(-max_offset, max_offset, num_steps)

        for offset in offsets:
            if abs(offset) < 0.001:  # 跳過已經計算過的中心點
                continue

            window_center = y_center + offset
            window_left = window_center - window_width/2
            window_right = window_center + window_width/2

            # 確保窗口在數據範圍內
            if window_left < y_min or window_right > y_max:
                continue

            # 計算該窗口內的平均高度
            mask = (y_coords >= window_left) & (y_coords <= window_right)
            heights_in_bin = z_coords[mask]

            if len(heights_in_bin) > 10:
                histogram_centers.append(window_center)
                histogram_heights.append(round(np.mean(heights_in_bin), 4))

        if len(histogram_heights) == 0:
            return None, None

        # 轉換為 numpy 陣列並按 Y 座標排序
        sorted_indices = np.argsort(histogram_centers)
        histogram_centers = np.array(histogram_centers)[sorted_indices]
        histogram_heights = np.array(histogram_heights)[sorted_indices]

        return histogram_centers, histogram_heights

    def _find_navigation_point_improved(self, centers, heights, slice_id,
                                        ridge_center_y, ridge_center_height, y_coords, z_coords):
        """
        從直方圖中找出最佳導航點
        策略：直接尋找直方圖中的全局最高點 (Global Maxima)
        """

        valid_centers = centers
        valid_heights = heights

        if len(valid_centers) == 0:
            return {"midpoint": None, "ridge_center": ridge_center_y, "navigation_peak": None}

        # 直接找直方圖中的最高點作為導航點
        global_max_idx = np.argmax(valid_heights)
        navigation_point_y = valid_centers[global_max_idx]
        navigation_point_height = valid_heights[global_max_idx]

        self.get_logger().debug(
            f'切片 {slice_id}: 導航點位置 Y={navigation_point_y:.3f}m, 高度={navigation_point_height:.3f}m')

        # 將導航點作為"中點"返回
        midpoint_y = navigation_point_y

        return {
            "midpoint": midpoint_y,
            "ridge_center": ridge_center_y,
            "navigation_peak": {"y": navigation_point_y, "height": navigation_point_height}
        }

    def _generate_navigation_from_ridge(self, ridge_centers):
        """
        從檢測到的多個土畦中心點生成一條平滑導航線
        使用線性回歸擬合： y = ax + b
        """
        if len(ridge_centers) < 2:
            self.get_logger().warn('土畦中心點不足，無法生成導航線')
            return [], 0.0, 0.0

        # 使用 RANSAC 進行穩健的線性擬合 (可抗噪點)
        line_params = self._robust_line_fitting(ridge_centers)

        if line_params is None:
            return [], 0.0, 0.0

        # 生成導航線 (用於視覺化)
        x_min = np.min(ridge_centers[:, 0])
        x_max = np.max(ridge_centers[:, 0])
        x_range = np.linspace(x_min, x_max, 100)

        # 計算導航線上的點 (y = slope * x + intercept)
        y_fitted = line_params['slope'] * x_range + line_params['intercept']
        navigation_line = [[x, y, 0.6] for x, y in zip(x_range, y_fitted)]

        # 計算航向角度 (Heading Angle)
        # 角度 = arctan(斜率)
        heading_angle = np.degrees(np.arctan(line_params['slope']))

        # 計算橫向偏移 (Lateral Offset) - 即截距
        # 代表機器人中心距離土畦中心的橫向偏差
        lateral_offset = line_params['intercept']

        # 儲存用於視覺化的資料
        self.ridge_line_fitted = navigation_line
        self.ridge_centers_raw = ridge_centers

        return navigation_line, heading_angle, lateral_offset

    def _robust_line_fitting(self, points):
        """
        使用 RANSAC (Random Sample Consensus) 進行線性擬合
        相比普通線性回歸，RANSAC 能有效忽略異常值 (Outliers)
        """
        if len(points) < 2:
            return None

        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1]

        # 設定 RANSAC 回歸器
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=2,          # 擬合所需最少點數
            residual_threshold=0.1, # 內點閾值 (距離線0.1m內的點視為有效)
            max_trials=100,         # 最大迭代次數
            random_state=42
        )

        ransac.fit(x, y)

        # 獲取內點 (Inliers) 遮罩
        inlier_mask = ransac.inlier_mask_
        outlier_ratio = 1 - np.sum(inlier_mask) / len(points)

        # 如果內點足夠多 (異常值少於50% 且至少有3個有效點)，接受這個結果
        if outlier_ratio < 0.5 and np.sum(inlier_mask) >= 3:
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_

            self.get_logger().debug(
                f'RANSAC successful with threshold={0.1}, '
                f'outlier_ratio={outlier_ratio:.2%}'
            )

            return {
                'slope': slope,
                'intercept': intercept,
                'inliers': points[inlier_mask],
                'outliers': points[~inlier_mask]
            }

        # 如果 RANSAC 失敗 (點太分散)，退回嘗試簡單的線性回歸 (使用所有點)
        self.get_logger().warn(
            'RANSAC failed with all thresholds, falling back to simple linear regression'
        )

        reg = LinearRegression()
        reg.fit(x, y)

        return {
            'slope': reg.coef_[0],
            'intercept': reg.intercept_,
            'inliers': points,
            'outliers': np.array([])
        }

    def _visualize_sliding_window_analysis(self, y_coords, z_coords,
                                           window_centers, height_means,
                                           coeffs, peak_position, peak_height, slice_id):
        """
        視覺化滑動窗口分析結果 (生成PNG圖片用於除錯)
        注意：此函數會進行 I/O 操作，僅在需要深度除錯時調用
        """
        import matplotlib.pyplot as plt

        # 確保輸出目錄存在
        output_dir = "result_output/sliding_window"
        os.makedirs(output_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 第一個子圖：原始點雲和平滑曲線
        ax1.scatter(y_coords, z_coords, c='blue',
                    alpha=0.3, s=2, label='Original')
        ax1.axvline(x=peak_position, color='green', linestyle='--',
                    linewidth=2, label=f'Ridge Center: y={peak_position:.3f}')
        ax1.set_xlabel('Y coordinate (m)')
        ax1.set_ylabel('Z coordinate (m)')
        ax1.set_title(f'Slice {slice_id}: Point Cloud Profile')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 第二個子圖：滑動窗口分析結果
        ax2.scatter(window_centers, height_means, c='blue',
                    s=30, label='Window Averages')

        # 繪製擬合的二次曲線
        y_fit = np.linspace(min(window_centers), max(window_centers), 200)
        z_fit = np.polyval(coeffs, y_fit)
        ax2.plot(y_fit, z_fit, 'r-', linewidth=2, label='Quadratic Fit')

        # 標記峰值
        ax2.scatter(peak_position, peak_height, c='green', s=100, marker='*',
                    zorder=10, label=f'Peak: ({peak_position:.3f}, {peak_height:.3f})')

        ax2.set_xlabel('Y coordinate (m)')
        ax2.set_ylabel('Average Height (m)')
        ax2.set_title('Sliding Window Analysis with Quadratic Fitting')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存圖像
        filename = os.path.join(
            output_dir, f'sliding_window_slice_{slice_id:03d}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        self.get_logger().info(
            f'切片 {slice_id}: 找到土畦中心 y={peak_position:.3f}, 高度={peak_height:.3f}')

    def _publish_ridge_markers(self, ridge_centers, navigation_line, header):
        """
        發佈土畦相關的視覺化標記 (MarkerArray) 到 Rviz
        1. 綠色球體：檢測到的土畦中心點
        2. 黃色線條：擬合後的導航線
        """
        marker_array = MarkerArray()

        # 土畦中心點 - 綠色球體
        if len(ridge_centers) > 0:
            for i, point in enumerate(ridge_centers):
                marker = self._create_sphere_marker(
                    point, 0.1, [0.0, 1.0, 0.0, 1.0], i, header)
                marker_array.markers.append(marker)

        # 導航線 - 黃色
        if len(navigation_line) > 0:
            nav_marker = self._create_line_marker(
                navigation_line, [1.0, 1.0, 0.0, 1.0], 100, header)
            marker_array.markers.append(nav_marker)

        self.marker_pub.publish(marker_array)

    def get_current_processing_stats(self):
        """獲取當前處理時間統計資訊 (供外部調用或除錯)"""
        if not self.processing_times['total']:
            return None

        recent_total = self.processing_times['total'][-10:]  # 最近10次
        recent_tf = self.processing_times['tf_transform'][-10:]
        recent_point = self.processing_times['point_processing'][-10:]
        recent_sliding = self.processing_times['sliding_window'][-10:]
        recent_nav = self.processing_times['navigation_generation'][-10:]
        recent_viz = self.processing_times['visualization'][-10:]

        stats = {
            'total_ms': np.mean(recent_total) * 1000,
            'tf_transform_ms': np.mean(recent_tf) * 1000,
            'point_processing_ms': np.mean(recent_point) * 1000,
            'sliding_window_ms': np.mean(recent_sliding) * 1000,
            'navigation_generation_ms': np.mean(recent_nav) * 1000,
            'visualization_ms': np.mean(recent_viz) * 1000,
            'frequency_hz': 1.0 / np.mean(recent_total),
            'sample_count': len(recent_total)
        }

        return stats

    def _publish_furrow_points_markers(self, left_points, right_points, header):
        """發佈畦溝點標記 (左紅/右藍)"""
        marker_array = MarkerArray()

        # 左畦溝點 - 紅色
        if len(left_points) > 0:
            for i, point in enumerate(left_points):
                marker = self._create_sphere_marker(
                    point, 0.08, [1.0, 0.0, 0.0, 1.0], i, header)
                marker_array.markers.append(marker)

        # 右畦溝點 - 藍色
        if len(right_points) > 0:
            for i, point in enumerate(right_points):
                marker = self._create_sphere_marker(
                    point, 0.08, [0.0, 0.0, 1.0, 1.0], i + len(left_points), header)
                marker_array.markers.append(marker)

        self.furrow_points_pub.publish(marker_array)

    def _publish_navigation_pose(self, navigation_line, angle, lateral_offset, header):
        """
        發布導航線的位置和角度資訊 (PoseStamped)
        這通常是控制系統 (Controller) 訂閱的目標話題
        """
        if len(navigation_line) == 0:
            return

        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = self.target_frame

        # 使用導航線的中點作為位置
        mid_point = navigation_line[len(navigation_line)//2]
        pose_msg.pose.position.x = 0.0
        pose_msg.pose.position.y = lateral_offset  # 橫向偏差
        pose_msg.pose.position.z = 0.6

        # 將歐拉角 (Angle) 轉換為四元數 (Quaternion)
        # 僅考慮繞 Z 軸旋轉 (Yaw)
        angle_rad = np.radians(angle)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = np.sin(angle_rad / 2.0)
        pose_msg.pose.orientation.w = np.cos(angle_rad / 2.0)

        self.navigation_pose_pub.publish(pose_msg)

    def _create_sphere_marker(self, point, size, color, marker_id, header):
        """Helper 函數：創建特徵質量球體標記"""
        marker = Marker()
        marker.header = header
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # 設置球體位置
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = 0.1  # 設置在地面上方一點
        marker.pose.orientation.w = 1.0

        # 設置球體大小
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = size

        # 設置顏色
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        return marker

    def _create_line_marker(self, points, color, marker_id, header):
        """Helper 函數：創建線條標記"""
        marker = Marker()
        marker.header = header
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 0.5

        # 設置線條粗細
        marker.scale.x = 0.05  # 線寬

        # 設置顏色
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        # 添加點
        for point in points:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.5  # 設置略高於地面的高度
            marker.points.append(p)

        return marker

    def _publish_markers(self, left_furrow, right_furrow, center_from_furrows, header):
        """發佈可視化標記 (包含左/右畦溝線與中心線)"""
        marker_array = MarkerArray()

        # 左畦溝線 - 紅色
        if len(left_furrow) > 0:
            left_marker = self._create_line_marker(
                left_furrow, [1.0, 0.0, 0.0, 1.0], 1, header)
            marker_array.markers.append(left_marker)

        # 右畦溝線 - 藍色
        if len(right_furrow) > 0:
            right_marker = self._create_line_marker(
                right_furrow, [0.0, 0.0, 1.0, 1.0], 2, header)
            marker_array.markers.append(right_marker)

        # 從畦溝計算的中間線 - 黃色
        if len(center_from_furrows) > 0:
            center_from_furrows_marker = self._create_line_marker(
                center_from_furrows, [1.0, 1.0, 0.0, 1.0], 3, header)
            marker_array.markers.append(center_from_furrows_marker)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)

    node = VisNavLineDetector()

    # 添加性能監控提示
    node.get_logger().info(
        "視覺導航線檢測節點已啟動\n"
        "性能監控功能已啟用:\n"
        "  - 每50次處理會輸出平均處理時間統計\n"
        "  - 每100次處理會輸出滑動窗口詳細統計\n"
        "  - 每200個切片會輸出切片處理詳情\n"
        "使用 '/vision_navigation_control' 服務來啟用/停用視覺導航"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 在退出前輸出最終統計
        node.get_logger().info("正在關閉節點...")
        if node.processing_times['total']:
            node._log_processing_times()
            node.get_logger().info(
                f"總共處理了 {len(node.processing_times['total'])} 次點雲數據")
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()