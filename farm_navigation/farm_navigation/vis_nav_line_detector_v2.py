# 視覺深度導航線v2
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
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import RANSACRegressor, LinearRegression
import os
import time
from scipy import interpolate
import matplotlib.pyplot as plt

# 確保輸出目錄存在
output_dir = "result_output/navigation_line"
os.makedirs(output_dir, exist_ok=True)

plt.rcParams.update({
    'font.size': 14,          # 基本字型大小
    'axes.titlesize': 16,     # 圖表標題字型大小
    'axes.labelsize': 14,     # 軸標籤字型大小
    'xtick.labelsize': 12,    # X軸刻度字型大小
    'ytick.labelsize': 12,    # Y軸刻度字型大小
    'legend.fontsize': 12,    # 圖例字型大小
    'figure.titlesize': 18    # 整體圖形標題字型大小
})


class VisNavLineDetector(Node):
    def __init__(self):
        super().__init__('vis_nav_line_detector')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.target_frame = "base_footprint"

        self._declare_parameters()
        self._init_variables()

        self.visualize = False
        self.vision_navigation_enabled = False

        # 訂閱狀態控制
        self.vision_nav_service = self.create_service(
            SetBool,
            '/vision_navigation_control',
            self.vision_navigation_callback
        )

        self.navigation_status_pub = self.create_publisher(
            Bool,
            '/vision_navigation_status',
            10
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/agri_bot/D455f/depth/color/points',
            self.pointcloud_callback,
            10
        )

        self.furrow_points_pub = self.create_publisher(
            MarkerArray,
            '/furrow_points_markers',
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/navigation_markers',
            10
        )

        self.navigation_pose_pub = self.create_publisher(
            PoseStamped,
            '/navigation_pose',
            10
        )

        self.get_logger().info('Navigation Line Detector Node initialized')

    def _declare_parameters(self):
        """宣告ROS2參數server參數"""
        self.declare_parameter('voxel_size', 0.01)  # 1cm
        self.declare_parameter('+++', 0.025)  # 5cm
        self.declare_parameter('x_minimum', -0.42)
        self.declare_parameter('x_maximum', 0.23)
        self.declare_parameter('y_minimum', -7)  # Y軸最小值
        self.declare_parameter('y_maximum', 0.7)   # Y軸最大值
        self.declare_parameter('gaussian_filter_sigma', 1.0)
        self.declare_parameter('window_width', 0.10)  # 10 cm
        self.declare_parameter('window_step', 0.025)   # 2.5cm

    def _init_variables(self):
        """從ROS參數server抓取參數"""
        self.voxel_size = self.get_parameter('voxel_size').value
        self.slice_thickness = self.get_parameter('slice_thickness').value
        self.x_minimum = self.get_parameter('x_minimum').value
        self.x_maximum = self.get_parameter('x_maximum').value
        self.y_minimum = self.get_parameter('y_minimum').value
        self.y_maximum = self.get_parameter('y_maximum').value
        self.sigma = self.get_parameter('gaussian_filter_sigma').value
        self.window_width = self.get_parameter('window_width').value
        self.step = self.get_parameter('window_step').value

    def vision_navigation_callback(self, request, response):
        """視覺導航控制服務回調"""
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

            status_msg = Bool()
            status_msg.data = request.data  # True=啟用, False=停用
            self.navigation_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'視覺導航控制服務錯誤: {str(e)}')
            response.success = False
            response.message = f"服務錯誤: {str(e)}"

        return response

    def pointcloud_callback(self, msg):
        if not self.vision_navigation_enabled:
            return  # 不處理點雲資料

        # 頻率控制，每3-4個點雲處理一次
        if not hasattr(self, '_frame_counter'):
            self._frame_counter = 0

        self._frame_counter += 1
        if self._frame_counter % 3 != 0:  # 每3幀處理一次，將頻率從12Hz降到4Hz
            return

        # 性能監控
        # start_time = time.time()

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0))

            # tf_time = time.time()

            points = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                p = np.array([point[0], point[1], point[2], 1.0])
                points.append(p)

            points_array = np.array(points)
            # read_time = time.time()

            TF_points = self._apply_transform(points_array, transform)
            # transform_time = time.time()

            new_header = msg.header
            new_header.frame_id = self.target_frame

            left_points, right_points = self._furrow_point_process(
                TF_points[:, 0:3])
            # process_time = time.time()

            navigation_line, heading_angle, lateral_offset = self._nav_line_fitting(
                left_points, right_points)

            # * 發佈航向資訊
            self._publish_navigation_pose(
                navigation_line, heading_angle, lateral_offset, new_header)

            self._publish_markers(
                navigation_line,
                new_header)

            if len(left_points) > 0 or len(right_points) > 0:
                self._publish_furrow_points_markers(
                    left_points, right_points, new_header)

            # 性能日誌
            # total_time = process_time - start_time
            # self.get_logger().info(
            #     f'處理時間: 總計={total_time:.3f}s, '
            #     f'TF查找={tf_time-start_time:.3f}s, '
            #     f'點讀取={read_time-tf_time:.3f}s, '
            #     f'轉換={transform_time-read_time:.3f}s, '
            #     f'處理={process_time-transform_time:.3f}s, '
            #     f'點數={len(points_array)}'
            # )

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as ex:
            self.get_logger().error(f'TF2 轉換錯誤: {str(ex)}')
            return

    def _apply_transform(self, points, transform):
        """應用轉換到點雲中的所有點"""
        # 從 transform 中提取旋轉矩陣和平移向量
        q = transform.transform.rotation
        t = transform.transform.translation

        # 建立旋轉矩陣 (四元數轉換)
        r = np.array([
            [1-2*(q.y*q.y+q.z*q.z), 2*(q.x*q.y-q.z*q.w), 2*(q.x*q.z+q.y*q.w), 0],
            [2*(q.x*q.y+q.z*q.w), 1-2*(q.x*q.x+q.z*q.z), 2*(q.y*q.z-q.x*q.w), 0],
            [2*(q.x*q.z-q.y*q.w), 2*(q.y*q.z+q.x*q.w), 1-2*(q.x*q.x+q.y*q.y), 0],
            [0, 0, 0, 1]
        ])

        # 建立平移矩陣
        r[0, 3] = t.x
        r[1, 3] = t.y
        r[2, 3] = t.z

        # 應用變換到所有點
        transformed_points = np.dot(points, r.T)

        return transformed_points

    def _furrow_point_process(self, points):
        y_mask = (points[:, 1] >= self.y_minimum) & (
            points[:, 1] <= self.y_maximum)
        x_mask = (points[:, 0] >= self.x_minimum) & (
            points[:, 0] <= self.x_maximum)

        # 合併兩個遮罩，一次性過濾
        combined_mask = y_mask & x_mask
        filtered_points = points[combined_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)

        pcd_voxel = pcd.voxel_down_sample(self.voxel_size)
        voxel_points = np.asarray(pcd_voxel.points)

        # 獲取過濾後的實際 X 範圍
        slice_x_min = np.min(voxel_points[:, 0])
        slice_x_max = np.max(voxel_points[:, 0])

        num_slices = int(
            np.ceil((slice_x_max - slice_x_min) / self.slice_thickness))

        left_furrow_points = []
        right_furrow_points = []

        # 預先計算所有切片的索引，避免重複計算
        x_coords = voxel_points[:, 0]
        slice_indices_list = []

        for slice_idx in range(num_slices):
            x_start = slice_x_min + self.slice_thickness * slice_idx
            x_end = x_start + self.slice_thickness

            slice_mask = (x_coords >= x_start) & (x_coords < x_end)
            slice_indices = np.where(slice_mask)[0]

            if len(slice_indices) > 100:  # 只處理有足夠點的切片
                slice_indices_list.append((slice_idx, slice_indices, x_start))

        # 批量處理切片
        for slice_idx, slice_indices, x_start in slice_indices_list:
            slice_points = voxel_points[slice_indices]

            # 按 Y 座標排序
            sorted_indices = np.argsort(slice_points[:, 1])
            sorted_points = slice_points[sorted_indices]

            left_furrow_y, right_furrow_y = self._identify_features_histogram(
                sorted_points[:, 1], sorted_points[:, 2], slice_id=slice_idx)

            if left_furrow_y is not None:
                left_furrow_points.append([x_start, left_furrow_y])
            if right_furrow_y is not None:
                right_furrow_points.append([x_start, right_furrow_y])

        return np.array(left_furrow_points),  np.array(right_furrow_points)

    def _identify_features_histogram(self, y_coords, z_coords, slice_id=0):
        """使用高度直方圖方法識別畦溝特徵"""

        y_min, y_max = np.min(y_coords), np.max(y_coords)
        y_center = 0

        histogram_centers = []
        histogram_heights = []
        histogram_stats = []

        # 從中心開始，向左右兩邊滑動
        max_offset = max(y_center - y_min, y_max -
                         y_center) - self.window_width/2

        # 檢查是否有足夠的空間進行滑動
        if max_offset <= 0:
            if self.visualize:
                self.get_logger().warn(f'切片 {slice_id}: 數據範圍太小，無法進行滑動窗口分析')
            return None, None

        # 處理中心窗口
        window_left = y_center - self.window_width/2
        window_right = y_center + self.window_width/2

        # 找出窗口內的點
        mask = (y_coords >= window_left) & (y_coords <= window_right)
        heights_in_bin = z_coords[mask]

        # 檢查是否有足夠的點
        if len(heights_in_bin) > 5:  # 至少需要5個點才能計算統計
            stats = {
                'center': y_center,
                'mean': np.mean(heights_in_bin),
                'median': np.median(heights_in_bin),
                'std': np.std(heights_in_bin),
                'min': np.min(heights_in_bin),
                'max': np.max(heights_in_bin),
                'count': len(heights_in_bin)
            }

            histogram_centers.append(y_center)
            histogram_heights.append(round(stats['mean'], 4))
            histogram_stats.append(stats)

        offset = self.step
        while offset <= max_offset:
            # 向左滑動
            left_center = y_center - offset
            if left_center - self.window_width/2 >= y_min:
                window_left = left_center - self.window_width/2
                window_right = left_center + self.window_width/2

                mask = (y_coords >= window_left) & (y_coords <= window_right)
                heights_in_bin = z_coords[mask]

                # 檢查是否有足夠的點
                if len(heights_in_bin) > 5:
                    stats = {
                        'center': left_center,
                        'mean': np.mean(heights_in_bin),
                        'median': np.median(heights_in_bin),
                        'std': np.std(heights_in_bin),
                        'min': np.min(heights_in_bin),
                        'max': np.max(heights_in_bin),
                        'count': len(heights_in_bin)
                    }

                    histogram_centers.append(left_center)
                    histogram_heights.append(round(stats['mean'], 4))
                    histogram_stats.append(stats)

            # 向右滑動
            right_center = y_center + offset
            if right_center + self.window_width/2 <= y_max:
                window_left = right_center - self.window_width/2
                window_right = right_center + self.window_width/2

                mask = (y_coords >= window_left) & (y_coords <= window_right)
                heights_in_bin = z_coords[mask]

                # 檢查是否有足夠的點
                if len(heights_in_bin) > 5:
                    stats = {
                        'center': right_center,
                        'mean': np.mean(heights_in_bin),
                        'median': np.median(heights_in_bin),
                        'std': np.std(heights_in_bin),
                        'min': np.min(heights_in_bin),
                        'max': np.max(heights_in_bin),
                        'count': len(heights_in_bin)
                    }

                    histogram_centers.append(right_center)
                    histogram_heights.append(round(stats['median'], 4))
                    histogram_stats.append(stats)

            offset += self.step

        # 檢查是否找到有效數據
        if len(histogram_heights) == 0:
            if self.visualize:
                self.get_logger().warn(f'切片 {slice_id}: 沒有找到有效的直方圖數據')
            return None, None

        # 轉換為 numpy 陣列
        sorted_indices = np.argsort(histogram_centers)
        histogram_centers = np.array(histogram_centers)[sorted_indices]
        histogram_heights = np.array(histogram_heights)[sorted_indices]

        # 在直方圖數據上找畦溝
        # histogram_furrow_info = self._find_furrows_from_histogram(
        #     histogram_centers, histogram_heights, histogram_stats)

        ridge_furrow_info = self._find_furrows_from_ridge(
            histogram_centers, histogram_heights, histogram_stats)

        # left_furrow, right_furrow = self._fuse_furrow_detection_results(
        #     histogram_furrow_info, ridge_furrow_info, slice_id)

        # fused_furrow_info = {
        #     'left_furrow': left_furrow,
        #     'right_furrow': right_furrow,
        #     'valleys': self._get_fused_valleys(left_furrow, right_furrow, histogram_centers),
        #     'heights_smooth': ridge_furrow_info['heights_smooth'],
        #     'histogram_info': histogram_furrow_info,
        #     'ridge_info': ridge_furrow_info
        # }

        left_furrow = ridge_furrow_info['left_furrow']
        right_furrow = ridge_furrow_info['right_furrow']

        # 視覺化直方圖
        if self.visualize:
            self._visualize_histogram(
                y_coords, z_coords,
                histogram_centers, histogram_heights, histogram_stats,
                ridge_furrow_info, slice_id)

        return left_furrow, right_furrow

    def _find_furrows_from_histogram(self, centers, heights, stats):
        """從高度直方圖中識別畦溝"""

        # 對直方圖進行額外平滑
        heights_smooth = gaussian_filter1d(heights, sigma=1.0)

        # 過濾掉heights_smooth=0的位置
        valid_mask = heights_smooth != 0
        valid_centers = centers[valid_mask]
        valid_heights = heights_smooth[valid_mask]

        if len(valid_centers) == 0:
            return {"left_furrow": None,
                    "right_furrow": None,
                    "valleys": np.array([]),
                    "heights_smooth": heights_smooth}

        center_y = 0.0  # 假設機器人在中心

        # 找左邊的最低點
        left_mask = valid_centers < center_y
        left_furrow = None
        if np.any(left_mask):
            left_heights = valid_heights[left_mask]
            left_centers = valid_centers[left_mask]
            min_idx = np.argmin(left_heights)
            left_furrow = left_centers[min_idx]

        # 找右邊的最低點
        right_mask = valid_centers > center_y
        right_furrow = None
        if np.any(right_mask):
            right_heights = valid_heights[right_mask]
            right_centers = valid_centers[right_mask]
            min_idx = np.argmin(right_heights)
            right_furrow = right_centers[min_idx]

        # 找出原始數組中對應的valleys索引用於可視化
        valleys = []
        if left_furrow is not None:
            left_idx = np.where(centers == left_furrow)[0]
            if len(left_idx) > 0:
                valleys.append(left_idx[0])
        if right_furrow is not None:
            right_idx = np.where(centers == right_furrow)[0]
            if len(right_idx) > 0:
                valleys.append(right_idx[0])

        return {"left_furrow": left_furrow,
                "right_furrow": right_furrow,
                "valleys": np.array(valleys),
                "heights_smooth": heights_smooth}

    def _find_furrows_from_ridge(self, centers, heights, stats):
        """基於土畦參考高度來找畦溝"""

        heights_smooth = gaussian_filter1d(heights, sigma=self.sigma)

        # 檢查數據點數量，決定插值方法
        if len(heights_smooth) >= 4:
            # 數據點足夠，使用 cubic 插值
            original_indices = np.arange(len(heights_smooth))
            new_indices = np.linspace(
                0, len(heights_smooth)-1, len(heights_smooth)*2)

            height_interpolator = interpolate.interp1d(
                original_indices, heights_smooth, kind='cubic')
            heights_smooth = height_interpolator(new_indices)

            center_interpolator = interpolate.interp1d(
                original_indices, centers, kind='cubic')
            centers = center_interpolator(new_indices)

        elif len(heights_smooth) >= 2:
            # 數據點較少，使用 linear 插值
            original_indices = np.arange(len(heights_smooth))
            new_indices = np.linspace(
                0, len(heights_smooth)-1, len(heights_smooth)*2)

            height_interpolator = interpolate.interp1d(
                original_indices, heights_smooth, kind='linear')
            heights_smooth = height_interpolator(new_indices)

            center_interpolator = interpolate.interp1d(
                original_indices, centers, kind='linear')
            centers = center_interpolator(new_indices)

        else:
            # 數據點太少，跳過插值
            self.get_logger().warn("數據點太少，跳過插值處理")

        # 過濾無效數據
        valid_mask = heights_smooth != 0
        valid_centers = centers[valid_mask]
        valid_heights = heights_smooth[valid_mask]

        if len(valid_centers) == 0:
            return {"left_furrow": None,
                    "right_furrow": None,
                    "valleys": np.array([]),
                    "heights_smooth": heights_smooth}

        center_y = 0.0
        ridge_width = 0.4  # 土畦半寬度
        ridge_reference_range = 0.1  # 參考高度計算範圍
        height_threshold_min = 0.08  # 10cm
        height_threshold_max = 0.13  # 20cm
        search_range = 0.6  # 搜索範圍

        # 1. 獲取土畦中心區域的參考高度
        ridge_reference_height = self._get_ridge_height(
            valid_centers,
            valid_heights,
            center_y,
            ridge_reference_range)

        # 2. 定義畦溝目標高度範圍
        furrow_height_min = ridge_reference_height - height_threshold_max
        furrow_height_max = ridge_reference_height - height_threshold_min

        # 3. 搜索左右畦溝
        left_furrow = self._search_furrow(
            valid_centers, valid_heights, center_y, 'left',
            furrow_height_min, furrow_height_max, search_range)

        right_furrow = self._search_furrow(
            valid_centers, valid_heights, center_y, 'right',
            furrow_height_min, furrow_height_max, search_range)

        # 找出對應的索引用於可視化
        valley_indices = []
        if left_furrow is not None:
            left_idx = np.where(centers == left_furrow)[0]
            if len(left_idx) > 0:
                valley_indices.append(left_idx[0])
        if right_furrow is not None:
            right_idx = np.where(centers == right_furrow)[0]
            if len(right_idx) > 0:
                valley_indices.append(right_idx[0])

        if self.visualize:
            self.get_logger().info(
                f'土畦參考高度: {ridge_reference_height:.3f}m, '
                f'畦溝搜索範圍: {furrow_height_min:.3f}~{furrow_height_max:.3f}m'
                f"原始 heights 的點數量: {len(heights)},"
                f'直方圖 heights_smooth 的點數量: {len(heights_smooth)}')

        return {"left_furrow": left_furrow,
                "right_furrow": right_furrow,
                "valleys": np.array(valley_indices),
                "heights_smooth": heights_smooth,
                "center_inter": centers}

    def _get_ridge_height(self, centers, heights, center_y, reference_range):
        """獲取強健的土畦參考高度"""
        # 方法1：使用指定範圍內的平均高度
        center_mask = (centers >= center_y - reference_range /
                       2) & (centers <= center_y + reference_range/2)

        if np.sum(center_mask) >= 3:  # 至少需要3個點
            ridge_heights = heights[center_mask]

            # 使用四分位數去除異常值後計算平均
            q25 = np.percentile(ridge_heights, 25)
            q75 = np.percentile(ridge_heights, 75)
            iqr_mask = (ridge_heights >= q25) & (ridge_heights <= q75)

            if np.sum(iqr_mask) >= 2:
                robust_height = np.mean(ridge_heights[iqr_mask])
                self.get_logger().debug(
                    f'使用中心範圍平均高度: {robust_height:.3f}m (基於{np.sum(iqr_mask)}個點)')
                return robust_height

        # 方法2：如果中心範圍數據不足，使用更大範圍的最高點
        extended_range = 0.6  # 擴大搜索範圍
        extended_mask = (centers >= center_y - extended_range /
                         2) & (centers <= center_y + extended_range/2)

        if np.sum(extended_mask) > 0:
            extended_heights = heights[extended_mask]
            # 使用前20%的高點平均值
            top_20_threshold = np.percentile(extended_heights, 80)
            top_heights = extended_heights[extended_heights >=
                                           top_20_threshold]
            robust_height = np.mean(top_heights)
            self.get_logger().debug(
                f'使用擴大範圍頂部20%平均高度: {robust_height:.3f}m')
            return robust_height

        # 方法3：最後備選，使用全域最高點
        robust_height = np.max(heights)
        return robust_height

    def _search_furrow(self, centers, heights, center_y, side,
                       height_min, height_max, search_range):
        """在直方圖數據中搜索符合條件的畦溝"""

        if side == 'left':
            search_mask = (centers >= center_y -
                           search_range) & (centers < center_y + 0.2)
        else:  # right
            search_mask = (centers <= center_y +
                           search_range) & (centers > center_y - 0.2)

        if np.sum(search_mask) == 0:
            return None

        search_centers = centers[search_mask]
        search_heights = heights[search_mask]

        # 找到符合高度條件的點
        height_mask = (search_heights >= height_min) & (
            search_heights <= height_max)

        if np.sum(height_mask) == 0:
            return None

        candidate_centers = search_centers[height_mask]

        # 選擇最靠近中心的符合條件的點
        if side == 'left':
            return np.max(candidate_centers)  # 左側選最大（最靠近中心）
        else:
            return np.min(candidate_centers)  # 右側選最小（最靠近中心）

    def _fuse_furrow_detection_results(self, histogram_info, ridge_info, slice_id=0):
        """融合兩種畦溝檢測方法的結果"""

        # 獲取兩種方法的結果
        hist_left = histogram_info['left_furrow']
        hist_right = histogram_info['right_furrow']
        ridge_left = ridge_info['left_furrow']
        ridge_right = ridge_info['right_furrow']

        # 融合參數
        max_deviation = 0.15  # 最大允許偏差 15cm

        # 融合左側畦溝
        final_left = self._fuse_single_furrow(
            hist_left, ridge_left, 'left', max_deviation, slice_id)

        # 融合右側畦溝
        final_right = self._fuse_single_furrow(
            hist_right, ridge_right, 'right', max_deviation, slice_id)

        return final_left, final_right

    def _fuse_single_furrow(self, hist_result, ridge_result, side, max_deviation, slice_id):
        """融合單側畦溝的檢測結果"""

        # 情況1：兩種方法都找到了結果
        if hist_result is not None and ridge_result is not None:
            deviation = abs(hist_result - ridge_result)

            if deviation <= max_deviation:
                # 兩個結果接近，取平均值
                fused_result = (hist_result + ridge_result) / 2
                if self.visualize:
                    self.get_logger().info(
                        f'切片 {slice_id} {side}側: 兩種方法一致 '
                        f'(直方圖={hist_result:.3f}, 土畦={ridge_result:.3f}, '
                        f'融合={fused_result:.3f})')
                return fused_result
            else:
                # 兩個結果差異太大，優先選擇土畦方法（更可靠）
                if self.visualize:
                    self.get_logger().warn(
                        f'切片 {slice_id} {side}側: 方法結果差異過大 '
                        f'(偏差={deviation:.3f}m), 採用土畦方法結果={ridge_result:.3f}')
                return ridge_result

        # 情況2：只有土畦方法找到結果
        elif ridge_result is not None:
            if self.visualize:
                self.get_logger().info(
                    f'切片 {slice_id} {side}側: 僅土畦方法有效，結果={ridge_result:.3f}')
            return ridge_result

        # 情況3：只有直方圖方法找到結果
        elif hist_result is not None:
            if self.visualize:
                self.get_logger().info(
                    f'切片 {slice_id} {side}側: 僅直方圖方法有效，結果={hist_result:.3f}')
            return hist_result

        # 情況4：兩種方法都沒找到
        else:
            if self.visualize:
                self.get_logger().warn(f'切片 {slice_id} {side}側: 兩種方法都未找到畦溝')
            return None

    def _get_fused_valleys(self, left_furrow, right_furrow, centers):
        """獲取融合結果的valleys索引用於可視化"""
        valleys = []

        if left_furrow is not None:
            left_idx = np.where(np.abs(centers - left_furrow) < 0.01)[0]
            if len(left_idx) > 0:
                valleys.append(left_idx[0])

        if right_furrow is not None:
            right_idx = np.where(np.abs(centers - right_furrow) < 0.01)[0]
            if len(right_idx) > 0:
                valleys.append(right_idx[0])

        return np.array(valleys)

    def _nav_line_fitting(self, left_points, right_points):
        """整合的導航線處理：計算左右點的中點，然後進行線性擬合"""

        # 確保左右點數組不為空
        if len(left_points) == 0 or len(right_points) == 0:
            self.get_logger().warn("左右畦溝點不足，無法計算導航線")
            return [], 0.0, 0.0

        # 找出左右點的共同X範圍
        left_x = left_points[:, 0]
        right_x = right_points[:, 0]

        # 對每個X位置，找出對應的左右Y值並計算中點
        center_points = []

        # 使用較小的點集作為基準
        if len(left_points) <= len(right_points):
            base_points = left_points
            other_points = right_points
        else:
            base_points = right_points
            other_points = left_points

        for base_point in base_points:
            base_x = base_point[0]
            base_y = base_point[1]

            # 在另一側找最接近的點
            distances = np.abs(other_points[:, 0] - base_x)
            closest_idx = np.argmin(distances)

            # 如果距離太遠，跳過這個點
            if distances[closest_idx] > 0.1:  # 10cm 容忍度
                continue

            other_y = other_points[closest_idx, 1]

            # 計算中點
            center_x = base_x
            center_y = (base_y + other_y) / 2
            center_points.append([center_x, center_y])

        if len(center_points) < 2:
            self.get_logger().warn("計算出的中點不足，無法進行線性擬合")
            return [], 0.0, 0.0

        center_points = np.array(center_points)

        # 對中點進行線性擬合
        center_line_params = self._robust_line_fitting(center_points)
        # center_line_params = self._simple_line_fitting(center_points)

        if center_line_params is None:
            self.get_logger().warn("中點線性擬合失敗")
            return [], 0.0, 0.0

        # 根據原始點的範圍生成擬合線
        all_x = np.concatenate([left_points[:, 0], right_points[:, 0]])
        x_min, x_max = np.min(all_x), np.max(all_x)

        # 生成密集的點來表示導航線
        x_range = np.linspace(x_min, x_max, 100)
        center_y_fitted = center_line_params['slope'] * \
            x_range + center_line_params['intercept']

        # 建立導航線
        navigation_line = [[x, y, 0] for x, y in zip(x_range, center_y_fitted)]

        # 計算導航角度（基於中點擬合的斜率）
        heading_angle = np.degrees(np.arctan(center_line_params['slope']))

        # 計算橫向偏移（在x=0位置的y值）
        lateral_offset = center_line_params['intercept']

        return navigation_line, heading_angle, lateral_offset

    def _robust_line_fitting(self, points):
        """使用RANSAC進行線性擬合"""
        if len(points) < 2:
            return None

        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1]

        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=2,
            residual_threshold=0.1,
            max_trials=100,
            random_state=42
        )

        ransac.fit(x, y)

        # 獲取內點
        inlier_mask = ransac.inlier_mask_
        outlier_ratio = 1 - np.sum(inlier_mask) / len(points)

        # 如果內點足夠多，接受這個結果
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

        # 如果RANSAC失敗，嘗試簡單的線性回歸（使用所有點）
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

    def _simple_line_fitting(self, points):
        if len(points) < 2:
            return None

        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1]

        reg = LinearRegression()
        reg.fit(x, y)

        return {
            'slope': reg.coef_[0],
            'intercept': reg.intercept_,
            'inliers': points,
            'outliers': np.array([])
        }

    def _visualize_histogram(self, y_coords, z_coords,
                             hist_centers, hist_heights, hist_stats,
                             furrow_info, slice_id):
        """視覺化高度直方圖分析結果"""
        import matplotlib.pyplot as plt

        # 確保輸出目錄存在
        output_dir = "result_output/navigation_line"
        os.makedirs(output_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 第一個子圖：原始點雲和平滑曲線
        ax1.scatter(y_coords, z_coords, c='blue',
                    alpha=0.3, s=1, label='Original')
        # ax1.scatter(y_coords[np.argsort(y_coords)],
        #             z_smooth[np.argsort(y_coords)],
        #             c='red', alpha=0.3, s=1, label='Smoothed')

        # 顯示直方圖的分區
        for i, center in enumerate(hist_centers):
            if i < len(hist_stats):

                ax1.axvspan(center - self.window_width/2, center + self.window_width/2,
                            alpha=0.1, color='gray')

        # ax1.set_xlabel('Y coordinate (m)')
        ax1.set_ylabel('Z coordinate (m)')
        ax1.set_title(f'Slice {slice_id}: Original Data with Histogram Bins')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        heights_smooth = furrow_info['heights_smooth']
        center_inter = furrow_info['center_inter']
        # 第二個子圖：高度直方圖
        bars = ax2.bar(hist_centers, hist_heights, width=self.window_width,
                       alpha=0.7, color='blue', edgecolor='black')

        ax2.scatter(center_inter, heights_smooth, c='red',
                    alpha=1.0, s=20, label='Smoothed')

        ax2.plot(center_inter, heights_smooth, 'r-',
                 linewidth=1, label='Smoothed')

        # 標記峰值和谷值
        valleys = furrow_info['valleys']

        if len(valleys) > 0:
            ax2.scatter(center_inter[valleys], heights_smooth[valleys],
                        c='red', s=100, marker='v', zorder=10, label='Valleys (Furrows)')

        # 標記選中的左右畦溝
        if furrow_info['left_furrow'] is not None:
            ax2.axvline(x=furrow_info['left_furrow'], color='red',
                        linestyle='--', linewidth=2, label='Left Furrow')

        if furrow_info['right_furrow'] is not None:
            ax2.axvline(x=furrow_info['right_furrow'], color='blue',
                        linestyle='--', linewidth=2, label='Right Furrow')

        ax2.set_xlabel('Y coordinate (m)')
        ax2.set_ylabel('Average Height (m)')
        ax2.set_title('Height Histogram Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # plt.tight_layout()

        # 保存圖像
        filename = os.path.join(
            output_dir, f'histogram_slice_{slice_id:03d}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

    def _publish_furrow_points_markers(self, left_points, right_points, header):
        """發佈畦溝點標記"""
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
        """發布導航線的位置和角度資訊"""
        if len(navigation_line) == 0:
            return

        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = self.target_frame

        # 使用導航線的中點作為位置
        mid_point = navigation_line[len(navigation_line)//2]
        pose_msg.pose.position.x = 0.0
        pose_msg.pose.position.y = lateral_offset
        pose_msg.pose.position.z = 0.6

        # 將角度轉換為四元數
        angle_rad = np.radians(angle)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = np.sin(angle_rad / 2.0)
        pose_msg.pose.orientation.w = np.cos(angle_rad / 2.0)

        self.navigation_pose_pub.publish(pose_msg)

    def _create_sphere_marker(self, point, size, color, marker_id, header):
        """創建特徵質量球體標記"""
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
        """創建線條標記"""
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

    def _publish_markers(self, center_from_furrows, header):
        """發佈可視化標記"""
        marker_array = MarkerArray()

        # 從畦溝計算的中間線 - 黃色
        if len(center_from_furrows) > 0:
            center_from_furrows_marker = self._create_line_marker(
                center_from_furrows, [1.0, 1.0, 0.0, 1.0], 3, header)
            marker_array.markers.append(center_from_furrows_marker)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)

    node = VisNavLineDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == 'main':
    main()
