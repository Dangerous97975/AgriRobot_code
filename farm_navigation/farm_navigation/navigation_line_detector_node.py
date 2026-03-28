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

import numpy as np
import open3d as o3d
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
import os
import tempfile


class NavigationLineDetectorNode(Node):
    def __init__(self):
        super().__init__('navigation_line_detector')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.target_frame = "base_footprint"

        self.declare_parameter('voxel_size', 0.01)  # 5 cm
        self.declare_parameter('slice_thickness', 0.05)  # 10 cm
        self.declare_parameter('output_dir', 'result_output/navigation_line')
        self.declare_parameter('feature_quality_threshold', 0.005)  # 特徵質量閾值
        self.declare_parameter('prominence',  0.03)         # 峰值的高度
        self.declare_parameter('history_buffer_size', 5)  # 儲存5次歷史數據
        self.declare_parameter('min_history_for_fitting', 3)  # 至少需要3次歷史數據才進行擬合
        self.declare_parameter('slice_minimum', -0.4)
        self.declare_parameter('slice_maximum', 0.2)

        self.voxel_size = self.get_parameter('voxel_size').value
        self.slice_thickness = self.get_parameter('slice_thickness').value
        self.output_dir = self.get_parameter('output_dir').value
        self.feature_quality_threshold = self.get_parameter(
            'feature_quality_threshold').value
        self.prominence = self.get_parameter('prominence').value
        self.history_buffer_size = self.get_parameter(
            'history_buffer_size').value
        self.min_history_for_fitting = self.get_parameter(
            'min_history_for_fitting').value
        self.slice_minimum = self.get_parameter('slice_minimum').value
        self.slice_maximum = self.get_parameter('slice_maximum').value

        self.visualize = True

        # 歷史數據緩衝區
        self.left_furrow_history = []
        self.right_furrow_history = []
        self.timestamp_history = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/agri_bot/D455f/depth/color/points',
            self.pointcloud_callback,
            10)

        self.navigation_pose_pub = self.create_publisher(
            PoseStamped,
            '/navigation_pose',
            10)

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/navigation_markers',
            10)

        self.feature_quality_pub = self.create_publisher(
            MarkerArray,
            '/feature_quality_markers',
            10)

        self.furrow_points_pub = self.create_publisher(
            MarkerArray,
            '/furrow_points_markers',
            10)

        self.get_logger().info('Navigation Line Detector Node initialized')

    def pointcloud_callback(self, msg):
        self.get_logger().debug('Received point cloud message')
        self.get_logger().debug(f'Point cloud frame_id: {msg.header.frame_id}')

        try:
            # 查詢從點雲座標系到目標座標系的轉換
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0))

            # 將點雲轉換為 numpy 陣列
            points = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                # 對每個點應用轉換
                p = np.array([point[0], point[1], point[2], 1.0])
                points.append(p)

            # 應用轉換到所有點
            points_array = np.array(points)
            transformed_points = self._apply_transform(points_array, transform)

            if transformed_points.shape[0] < 100:
                self.get_logger().warn('Not enough points in the cloud')
                return

            # 修改 header 以使用新的座標系
            new_header = msg.header
            new_header.frame_id = self.target_frame

            # self.get_logger().info(
            #     f'Transformed points: {points}')

            # 繼續處理轉換後的點雲
            self._process_pointcloud_to_navigation_line(
                transformed_points[:, 0:3], new_header)

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

    def _process_pointcloud_to_navigation_line(self, points, header):
        """處理點雲數據並計算導航線"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd_voxel = pcd.voxel_down_sample(self.voxel_size)
        voxel_points = np.asarray(pcd_voxel.points)

        num_slices = int(
            (self.slice_maximum - self.slice_minimum) / self.slice_thickness) + 1

        left_furrow_points = []
        right_furrow_points = []

        # 記錄特徵質量
        left_quality_scores = []
        right_quality_scores = []

        # 儲存每個切片的資料以供後續可視化
        slice_data_for_viz = []

        for i in range(num_slices):
            x_start = self.slice_minimum + self.slice_thickness*i
            x_end = x_start + self.slice_thickness
            x_center = x_start + self.slice_thickness/2

            # 提取當前切片中的點索引
            slice_indices = np.where(
                (voxel_points[:, 0] >= x_start) & (voxel_points[:, 0] < x_end))[0]

            if len(slice_indices) > 500:
                # 選取全部點雲空間的對應的索引點
                slice_points = voxel_points[slice_indices]

                # 按y座標排序
                sorted_indices = np.argsort(slice_points[:, 1])
                sorted_points = slice_points[sorted_indices]

                y_cords = sorted_points[:, 1]
                z_cords = sorted_points[:, 2]

                # @ 識別畦溝點
                left_furrow_y, right_furrow_y, left_quality, right_quality = self._identify_features(
                    y_cords, z_cords, slice_id=i)

                features = (left_furrow_y, right_furrow_y,
                            left_quality, right_quality)

                if self.visualize:
                    self.get_logger().info(
                        f'切片 {i}: 左畦溝={left_furrow_y}, 右畦溝={right_furrow_y}, 點雲總數{len(slice_points)}')
                    self._visualize_slice(
                        slice_points, i, (x_start, x_end), features)
                    slice_data_for_viz.append((x_center, features))

                # 只有當特徵質量超過閾值時才記錄
                if left_furrow_y is not None and left_quality > self.feature_quality_threshold:
                    left_furrow_points.append(
                        [x_start + self.slice_thickness/2, left_furrow_y, 0])
                    left_quality_scores.append(left_quality)

                if right_furrow_y is not None and right_quality > self.feature_quality_threshold:
                    right_furrow_points.append(
                        [x_start + self.slice_thickness/2, right_furrow_y, 0])
                    right_quality_scores.append(right_quality)

        # 創建所有切片的總覽圖
        if self.visualize:
            self._create_all_slices_overview(slice_data_for_viz)

            # 轉換為numpy陣列
        left_points = np.array(left_furrow_points)
        right_points = np.array(right_furrow_points)

        left_furrow_points = self._remove_outliers_by_position(
            left_points, side='left')
        right_furrow_points = self._remove_outliers_by_position(
            right_points, side='right')

        # 更新歷史數據
        current_time = self.get_clock().now()
        self._update_history_buffer(
            left_furrow_points, right_furrow_points, current_time)

        if len(self.left_furrow_history) >= self.min_history_for_fitting:
            left_points_combined, right_points_combined = self._combine_historical_data()
        else:
            left_points_combined = left_furrow_points
            right_points_combined = right_furrow_points

        navigation_line, heading_angle, lateral_offset = self._process_and_fit_navigation_line(
            left_points_combined, right_points_combined)

        # 發佈可視化標記（使用擬合後的線）
        if hasattr(self, 'left_furrow_fitted') and hasattr(self, 'right_furrow_fitted'):
            self._publish_markers(
                self.left_furrow_fitted,
                self.right_furrow_fitted,
                navigation_line,
                header)

        if len(left_points_combined) > 0 or len(right_points_combined) > 0:
            self._publish_furrow_points_markers(
                left_points_combined, right_points_combined, header)

        # 發布導航位置和角度資訊
        if len(navigation_line) > 0:
            self._publish_navigation_pose(
                navigation_line, heading_angle, lateral_offset, header)
            # self.get_logger().info(
            #     f'Navigation angle: {heading_angle:.2f} degrees, lateral offset: {lateral_offset:.2f} m')

        self.get_logger().debug('Navigation lines calculated and published')

    def _identify_features(self, y_cords, z_cords, slice_id=0):
        """識別畦溝特徵並評估其質量"""

        # * Gaussian filter
        z_smooth = gaussian_filter1d(z_cords, sigma=2.0)

        # * Spline
        # y_unique, unique_indices = np.unique(y_cords, return_index=True)
        # z_unique = z_cords[unique_indices]
        # spl = UnivariateSpline(y_unique, z_unique, k=3, s=0)
        # z_smooth = spl(y_cords)

        # 找到畦溝的谷值
        valleys, _ = find_peaks(
            -z_smooth,
            prominence=self.prominence,
            width=2,
        )

        valleys = valleys.astype(int)  # 確保索引是整數

        # 計算每個谷值的寬度
        valley_widths = []
        valley_boundaries = []

        if len(valleys) > 0:
            # 使用 peak_widths 計算寬度
            widths, _, left_ips, right_ips = peak_widths(
                -z_smooth,
                valleys,
                rel_height=0.5
            )

            # 轉換為實際距離
            y_resolution = np.mean(np.diff(y_cords)) if len(
                y_cords) > 1 else 1.0
            actual_widths = widths * y_resolution

            for i in range(len(valleys)):
                # 獲取邊界點的 y 座標
                left_y = np.interp(
                    left_ips[i], np.arange(len(y_cords)), y_cords)
                right_y = np.interp(
                    right_ips[i], np.arange(len(y_cords)), y_cords)

                valley_widths.append(actual_widths[i])
                valley_boundaries.append((left_y, right_y))

        # 計算每個谷值的質量分數
        valley_quality = []
        for v in valleys:
            # 基於深度的質量
            depth_quality = 0.0
            if v > 5 and v < len(z_smooth) - 5:
                # 計算相對深度
                left_range = z_smooth[max(0, v-10):v]
                right_range = z_smooth[v:min(len(z_smooth), v+10)]

                # 合併兩個範圍並找到最大值
                combined_range = np.concatenate([left_range, right_range])
                if len(combined_range) > 0:
                    local_max = np.max(combined_range)
                    depth = local_max - z_smooth[v]
                    depth_quality = depth

            # 基於對稱性的質量
            symmetry_quality = 0.0
            if v > 10 and v < len(z_smooth) - 10:
                left_profile = z_smooth[v-10:v]
                right_profile = z_smooth[v:v+10]
                # 計算左右對稱性
                if len(left_profile) > 0 and len(right_profile) > 0:
                    symmetry = 1.0 - \
                        np.abs(np.mean(left_profile) - np.mean(right_profile))
                    symmetry_quality = max(0, symmetry)

            # 基於寬度的質量（假設合理的畦溝寬度範圍）
            width_quality = 0.0
            if i < len(valley_widths):
                width = valley_widths[i]
                # 假設理想寬度在 0.15m 到 0.4m 之間
                if 0.15 <= width <= 0.4:
                    width_quality = 1.0 - \
                        abs(width - 0.25) / 0.15  # 0.25m 為理想寬度
                else:
                    width_quality = 0.0

            # 總質量分數
            total_quality = depth_quality * 0.5 + \
                symmetry_quality * 0.2 + width_quality * 0.3
            valley_quality.append(total_quality)

        valley_quality = np.array(valley_quality)

        # 獲取對應的y座標
        valley_y = y_cords[valleys]

        # 因為是垂直視角，整體區域的中心應該是最佳分割點
        y_center = (np.min(y_cords) + np.max(y_cords)) / 2

        # 如果有多個谷值，嘗試找到最合理的配對
        left_candidates = []
        right_candidates = []

        for i, val_y in enumerate(valley_y):
            if i < len(valley_quality):  # 確保索引不超出範圍
                if val_y < y_center:
                    left_candidates.append((val_y, valley_quality[i], i))
                else:
                    right_candidates.append((val_y, valley_quality[i], i))
        # 選擇最佳的左右畦溝
        left_furrow_y = None
        right_furrow_y = None
        left_quality = 0.0
        right_quality = 0.0

        # 左畦溝：選擇最左邊的候選（從左側開始找）
        if left_candidates:
            left_candidates.sort(key=lambda x: x[0], reverse=True)
            best_left = left_candidates[0]
            left_furrow_y = best_left[0]
            left_quality = best_left[1]

        # 右畦溝：選擇最右邊的候選（從右側開始找）
        if right_candidates:
            right_candidates.sort(
                key=lambda x: x[0])
            best_right = right_candidates[0]
            right_furrow_y = best_right[0]
            right_quality = best_right[1]

        return left_furrow_y, right_furrow_y, left_quality, right_quality

    def _process_and_fit_navigation_line(self, left_furrow_points, right_furrow_points):
        """整合的導航線處理：異常值移除、線性擬合、向量計算"""

        # 使用RANSAC進行穩健的線性回歸（自動處理異常值）
        left_line_params = self._robust_line_fitting(left_furrow_points)
        right_line_params = self._robust_line_fitting(right_furrow_points)

        # 如果""任一側擬合失敗，返回空結果
        if left_line_params is None or right_line_params is None:
            self.get_logger().warn('Line fitting failed due to too many outliers')
            return [], 0.0, 0.0  # 返回三個值

        # 根據原始點的範圍生成擬合線
        all_x = np.concatenate(
            [left_furrow_points[:, 0], right_furrow_points[:, 0]])
        x_min, x_max = np.min(all_x), np.max(all_x)

        # 生成密集的點來表示擬合線
        x_range = np.linspace(x_min, x_max, 100)

        # 計算左右擬合線上的點
        left_y_fitted = left_line_params['slope'] * \
            x_range + left_line_params['intercept']
        right_y_fitted = right_line_params['slope'] * \
            x_range + right_line_params['intercept']

        # 計算中心導航線
        center_y = (left_y_fitted + right_y_fitted) / 2
        navigation_line = [[x, y, 0] for x, y in zip(x_range, center_y)]

        # 計算導航角度（基於擬合的斜率）
        # 左右線斜率的平均值代表整體方向
        avg_slope = (left_line_params['slope'] +
                     right_line_params['slope']) / 2
        heading_angle = np.degrees(np.arctan(avg_slope))

        # 計算橫向偏移
        # 中心線的參數
        center_intercept = (
            left_line_params['intercept'] + right_line_params['intercept']) / 2
        lateral_offset = center_intercept

        # 更新用於視覺化的點（使用擬合後的線）
        self.left_furrow_fitted = [[x, y, 0]
                                   for x, y in zip(x_range, left_y_fitted)]
        self.right_furrow_fitted = [[x, y, 0]
                                    for x, y in zip(x_range, right_y_fitted)]

        return navigation_line, heading_angle, lateral_offset

    def _remove_outliers_by_position(self, points, side='left'):
        """基於位置分佈移除離群點"""
        if len(points) < 3:
            return points

        y_coords = points[:, 1]

        # 使用四分位距方法檢測Y座標的離群點
        Q1 = np.percentile(y_coords, 25)
        Q3 = np.percentile(y_coords, 75)
        IQR = Q3 - Q1

        # 計算離群點閾值
        outlier_threshold = 1.5 * IQR
        lower_bound = Q1 - outlier_threshold
        upper_bound = Q3 + outlier_threshold

        # 基於邊界的額外約束
        y_center = 0.0  # 假設中心線在Y=0

        if side == 'left':
            # 左側畦溝應該在負Y值區域
            expected_bound = -0.2  # 左側點應該小於-0.2m
            valid_mask = (y_coords >= lower_bound) & (
                y_coords <= upper_bound) & (y_coords < expected_bound)
        else:  # right
            # 右側畦溝應該在正Y值區域
            expected_bound = 0.2   # 右側點應該大於0.2m
            valid_mask = (y_coords >= lower_bound) & (
                y_coords <= upper_bound) & (y_coords > expected_bound)

        filtered_points = points[valid_mask]

        # 記錄移除的離群點
        removed_points = points[~valid_mask]
        if len(removed_points) > 0:
            self.get_logger().debug(f'{side.capitalize()} side removed {len(removed_points)} outliers: '
                                    f'Y coords = {removed_points[:, 1]}')

        return filtered_points

    def _robust_line_fitting(self, points):
        """使用RANSAC進行穩健的線性擬合，自動處理異常值"""
        if len(points) < 2:
            return None

        try:
            from sklearn.linear_model import RANSACRegressor, LinearRegression

            X = points[:, 0].reshape(-1, 1)
            y = points[:, 1]

            # 首先嘗試使用較寬鬆的閾值
            residual_thresholds = [0.1, 0.15, 0.2, 0.3]  # 逐步放寬閾值

            for threshold in residual_thresholds:
                ransac = RANSACRegressor(
                    estimator=LinearRegression(),
                    min_samples=2,
                    residual_threshold=threshold,
                    max_trials=100,
                    random_state=42
                )

                ransac.fit(X, y)

                # 獲取內點
                inlier_mask = ransac.inlier_mask_
                outlier_ratio = 1 - np.sum(inlier_mask) / len(points)

                # 如果內點足夠多，接受這個結果
                if outlier_ratio < 0.5 and np.sum(inlier_mask) >= 3:
                    slope = ransac.estimator_.coef_[0]
                    intercept = ransac.estimator_.intercept_

                    self.get_logger().debug(
                        f'RANSAC successful with threshold={threshold}, '
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
            reg.fit(X, y)

            return {
                'slope': reg.coef_[0],
                'intercept': reg.intercept_,
                'inliers': points,
                'outliers': np.array([])
            }

        except Exception as e:
            self.get_logger().error(f'Error in robust line fitting: {str(e)}')
            return None

    def _update_history_buffer(self, left_points, right_points, timestamp):
        """更新歷史數據緩衝區"""
        # 加入新的數據
        self.left_furrow_history.append(left_points.copy())
        self.right_furrow_history.append(right_points.copy())
        self.timestamp_history.append(timestamp)

        # 維持緩衝區大小
        if len(self.left_furrow_history) > self.history_buffer_size:
            self.left_furrow_history.pop(0)
            self.right_furrow_history.pop(0)
            self.timestamp_history.pop(0)

        self.get_logger().debug(
            f'History buffer updated: {len(self.left_furrow_history)} entries')

    def _combine_historical_data(self):
        """結合歷史數據，給予較新的數據更高權重"""
        if len(self.left_furrow_history) == 0:
            return np.array([]), np.array([])

        # 計算時間權重（較新的數據權重更高）
        current_time = self.timestamp_history[-1]
        weights = []
        for timestamp in self.timestamp_history:
            time_diff = (current_time - timestamp).nanoseconds / 1e9  # 轉換為秒
            weight = np.exp(-time_diff / 2.0)  # 2秒衰減常數
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / np.sum(weights)  # 正規化權重

        # 結合左側畦溝點
        all_left_points = []
        for i, points in enumerate(self.left_furrow_history):
            if len(points) > 0:
                # 為每個點加上時間權重（複製點以反映權重）
                repeat_count = max(1, int(weights[i] * 10))  # 權重轉換為重複次數
                for _ in range(repeat_count):
                    all_left_points.extend(points.tolist())

        # 結合右側畦溝點
        all_right_points = []
        for i, points in enumerate(self.right_furrow_history):
            if len(points) > 0:
                repeat_count = max(1, int(weights[i] * 10))
                for _ in range(repeat_count):
                    all_right_points.extend(points.tolist())

        combined_left = np.array(
            all_left_points) if all_left_points else np.array([])
        combined_right = np.array(
            all_right_points) if all_right_points else np.array([])

        self.get_logger().debug(
            f'Combined historical data - Left: {len(combined_left)}, Right: {len(combined_right)} points')

        return combined_left, combined_right

    def _visualize_slice(self, slice_points, slice_id, slice_x_range, features=None):
        """可視化單個切片中的點和檢測到的特徵"""
        import matplotlib.pyplot as plt

        if len(slice_points) == 0:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

        # 按y座標排序點
        sorted_indices = np.argsort(slice_points[:, 1])
        sorted_points = slice_points[sorted_indices]

        y_cords = sorted_points[:, 1]
        z_cords = sorted_points[:, 2]

        # 上面的子圖：顯示所有點在y-z平面上的分佈
        ax1.scatter(y_cords, z_cords, c='blue', alpha=0.6, s=2)
        ax1.set_xlabel('Y coordinates (m)')
        ax1.set_ylabel('Z coordinates (m)')
        ax1.set_title(
            f'slice {slice_id} (X range: {slice_x_range[0]:.3f} ~ {slice_x_range[1]:.3f}) - point distribution')
        ax1.grid(True, alpha=0.3)

        # 設定 Y 軸範圍為 -1.5 到 1.5 米
        # ax1.set_xlim(-1.5, 1.5)

        # 如果提供了特徵資訊，在圖上標註
        if features is not None:
            left_furrow_y, right_furrow_y, left_quality, right_quality = features

            # 找到對應的z座標
            if left_furrow_y is not None and -1.5 <= left_furrow_y <= 1.5:
                left_idx = np.argmin(np.abs(y_cords - left_furrow_y))
                ax1.axvline(x=left_furrow_y, color='red',
                            linestyle='--', alpha=0.7)
                ax1.scatter([left_furrow_y], [z_cords[left_idx]], c='red', s=100, marker='o',
                            edgecolors='black', linewidth=2, zorder=10)
                ax1.text(left_furrow_y, z_cords[left_idx], f'L({left_quality:.3f})',
                         verticalalignment='bottom', color='red', fontweight='bold')

            if right_furrow_y is not None and -1.5 <= right_furrow_y <= 1.5:
                right_idx = np.argmin(np.abs(y_cords - right_furrow_y))
                ax1.axvline(x=right_furrow_y, color='green',
                            linestyle='--', alpha=0.7)
                ax1.scatter([right_furrow_y], [z_cords[right_idx]], c='green', s=100, marker='o',
                            edgecolors='black', linewidth=2, zorder=10)
                ax1.text(right_furrow_y, z_cords[right_idx], f'R({right_quality:.3f})',
                         verticalalignment='bottom', color='green', fontweight='bold')

        # 下面的子圖：顯示平滑後的高度剖面
        window_length = min(21, len(y_cords) -
                            (1 if len(y_cords) % 2 == 0 else 0))
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 3:
            z_smooth = gaussian_filter1d(z_cords, sigma=2.0)

            ax2.plot(y_cords, z_cords, 'b-', alpha=0.3, label='Original')
            ax2.plot(y_cords, z_smooth, 'r-', linewidth=2, label='smoothed')

            # 標記峰值和谷值
            peaks, _ = find_peaks(z_smooth, prominence=self.prominence)
            valleys, _ = find_peaks(-z_smooth, prominence=self.prominence)

            if len(peaks) > 0:
                # 只顯示在Y軸範圍內的峰值
                valid_peaks = peaks[(-1.5 <= y_cords[peaks]) &
                                    (y_cords[peaks] <= 1.5)]
                if len(valid_peaks) > 0:
                    ax2.scatter(y_cords[valid_peaks], z_smooth[valid_peaks], c='orange', s=100, marker='^',
                                edgecolors='black', linewidth=1, zorder=5, label='peak')

            if len(valleys) > 0:
                # 只顯示在Y軸範圍內的谷值
                valid_valleys = valleys[(-1.5 <= y_cords[valleys])
                                        & (y_cords[valleys] <= 1.5)]
                if len(valid_valleys) > 0:
                    ax2.scatter(y_cords[valid_valleys], z_smooth[valid_valleys], c='purple', s=100, marker='v',
                                edgecolors='black', linewidth=1, zorder=5, label='valley')

            # 標記檢測到的特徵
            if features is not None:
                if left_furrow_y is not None and -1.5 <= left_furrow_y <= 1.5:
                    ax2.axvline(x=left_furrow_y, color='red', linestyle='--', alpha=0.7,
                                label='Left furrow line')
                if right_furrow_y is not None and -1.5 <= right_furrow_y <= 1.5:
                    ax2.axvline(x=right_furrow_y, color='green', linestyle='--', alpha=0.7,
                                label='Right furrow line')

            ax2.legend()

            ax2.set_xlabel('Y coordinates (m)')
            ax2.set_ylabel('Z coordinates (m)')
            ax2.set_title('Smoothed height profile')
            ax2.grid(True, alpha=0.3)

            # 設定 Y 軸範圍為 -1.5 到 1.5 米
            # ax2.set_xlim(-1.5, 1.5)

        # 調整版面
        plt.tight_layout()

        filename = os.path.join(self.output_dir, f'slice_{slice_id:03d}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        self.get_logger().debug(f'Saved slice visualization: {filename}')

    def _create_all_slices_overview(self, slice_data_list):
        """創建所有切片特徵的總覽圖"""
        import matplotlib.pyplot as plt

        if len(slice_data_list) == 0:
            return

        # 準備資料
        x_positions = []
        left_furrow_y = []
        right_furrow_y = []
        left_qualities = []
        right_qualities = []

        for data in slice_data_list:
            x_pos, features = data
            if features is not None:
                left_y, right_y, left_q, right_q = features
                x_positions.append(x_pos)
                left_furrow_y.append(
                    left_y if left_y is not None else np.nan)
                right_furrow_y.append(
                    right_y if right_y is not None else np.nan)
                left_qualities.append(left_q)
                right_qualities.append(right_q)

        # 轉換為numpy數組
        x_positions = np.array(x_positions)
        left_furrow_y = np.array(left_furrow_y)
        right_furrow_y = np.array(right_furrow_y)
        left_qualities = np.array(left_qualities)
        right_qualities = np.array(right_qualities)

        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 上面的子圖：顯示檢測到的畦溝線位置
        ax1.plot(x_positions, left_furrow_y, 'ro-',
                 label='Left furrow line', markersize=5, linewidth=2)
        ax1.plot(x_positions, right_furrow_y, 'go-',
                 label='Right furrow line', markersize=5, linewidth=2)
        ax1.fill_between(x_positions, left_furrow_y, right_furrow_y, alpha=0.2, color='blue',
                         label='Crop row width')

        ax1.set_xlabel('X coordinate (m)')
        ax1.set_ylabel('Y coordinate (m)')
        ax1.set_title('All slice furrow point result')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 下面的子圖：顯示特徵質量
        ax2.plot(x_positions, left_qualities, 'r-',
                 marker='o', label='Left furrow points quality', markersize=5)
        ax2.plot(x_positions, right_qualities, 'g-',
                 marker='o', label='Right furrow points quality', markersize=5)
        ax2.axhline(y=self.feature_quality_threshold, color='black', linestyle='--', alpha=0.7,
                    label=f'Quality threshold ({self.feature_quality_threshold})')

        ax2.set_xlabel('X coordinate (m)')
        ax2.set_ylabel('Quality mass')
        ax2.set_title('Qualiity mass evalute')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 調整版面
        plt.tight_layout()

        # 儲存圖像
        overview_filename = os.path.join(
            self.output_dir, 'all_slices_overview.png')
        plt.savefig(overview_filename, dpi=150, bbox_inches='tight')
        plt.close()

        self.get_logger().info(
            f'Saved overview visualization: {overview_filename}')

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

    def _publish_markers(self, left_furrow, right_furrow, center_from_furrows, header):
        """發佈可視化標記"""
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

    def _publish_feature_quality_markers(self, left_points, right_points,
                                         left_quality, right_quality, header):
        """發佈特徵質量視覺化標記"""
        marker_array = MarkerArray()

        # 左畦溝特徵質量 - 綠色球體，尺寸根據質量
        if len(left_points) > 0 and len(left_quality) > 0:
            for i, point in enumerate(left_points):
                if i < len(left_quality):
                    quality = left_quality[i]
                    marker = self._create_sphere_marker(
                        point, 0.05 + quality * 2.0, [0.0, 1.0, 0.0, 0.7], i, header)
                    marker_array.markers.append(marker)

        # 右畦溝特徵質量 - 青色球體，尺寸根據質量
        if len(right_points) > 0 and len(right_quality) > 0:
            offset = len(left_points)
            for i, point in enumerate(right_points):
                if i < len(right_quality):
                    quality = right_quality[i]
                    marker = self._create_sphere_marker(
                        point, 0.05 + quality * 2.0, [0.0, 1.0, 1.0, 0.7], i + offset, header)
                    marker_array.markers.append(marker)

        self.feature_quality_pub.publish(marker_array)

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


def main(args=None):
    rclpy.init(args=args)

    node = NavigationLineDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == 'main':
    main()
