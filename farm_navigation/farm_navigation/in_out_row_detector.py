#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import open3d as o3d
from collections import deque
import json


class FurrowEndDetector(Node):
    def __init__(self):
        super().__init__('furrow_end_detector')

        # TF2設定
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.target_frame = "base_footprint"

        # 宣告參數
        self._declare_parameters()
        self._init_variables()

        # 檢測狀態
        self.detection_history = deque(maxlen=self.history_size)
        self.current_state = "unknown"  # unknown, in_field, out_field
        self.previous_state = "unknown"

        # 訂閱點雲
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/agri_bot/D455f/depth/color/points',
            self.pointcloud_callback,
            10
        )

        # 發布檢測結果
        self.state_pub = self.create_publisher(
            String,
            '/furrow_state',
            10
        )

        # 發布狀態變化事件
        self.transition_pub = self.create_publisher(
            String,
            '/furrow_transition',
            10
        )

        # 視覺化
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/furrow_end_markers',
            10
        )

        # # 調試資訊
        self.debug_pub = self.create_publisher(
            String,
            '/furrow_debug_info',
            10
        )

        self.get_logger().info('土畦尾檢測節點已初始化')

    def _declare_parameters(self):
        """宣告ROS2參數"""
        # 檢測區域參數
        self.declare_parameter('detection_min_x', 0.2)      # 最近檢測距離
        self.declare_parameter('detection_max_x', 2.0)      # 最遠檢測距離

        # 切片參數
        self.declare_parameter('slice_thickness', 0.1)      # 切片厚度
        self.declare_parameter('voxel_size', 0.02)          # 體素大小

        # 檢測閾值
        self.declare_parameter('iqr_threshold', 0.03)       # IQR閾值（替代標準差閾值）8cm
        self.declare_parameter('min_points_per_slice', 20)  # 每個切片最少點數

        # 狀態判斷參數
        self.declare_parameter('history_size', 8)           # 歷史記錄大小
        self.declare_parameter('confidence_threshold', 0.75)  # 信心度閾值

        # 可視化
        self.declare_parameter('enable_visualization', True)

    def _init_variables(self):
        """獲取ROS2參數"""
        self.detection_min_x = self.get_parameter('detection_min_x').value
        self.detection_max_x = self.get_parameter('detection_max_x').value

        self.slice_thickness = self.get_parameter('slice_thickness').value
        self.voxel_size = self.get_parameter('voxel_size').value

        self.iqr_threshold = self.get_parameter('iqr_threshold').value
        self.min_points_per_slice = self.get_parameter(
            'min_points_per_slice').value

        self.history_size = self.get_parameter('history_size').value
        self.confidence_threshold = self.get_parameter(
            'confidence_threshold').value

        self.enable_visualization = self.get_parameter(
            'enable_visualization').value

    def pointcloud_callback(self, msg):
        """點雲回調函數"""
        try:
            # TF轉換
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0))

            # 提取和轉換點雲
            points = self._extract_and_transform_points(msg, transform)

            if len(points) == 0:
                return

            # 濾除檢測區域
            detection_points = self._filter_detection_area(points)

            if len(detection_points) < self.min_points_per_slice:
                return

            # 進行高度變化分析
            analysis_result = self._analyze_height_variation(detection_points)

            # 更新檢測歷史
            self.detection_history.append(analysis_result)

            # 判斷當前狀態
            new_state = self._determine_state()

            # 檢查狀態變化
            if new_state != self.current_state:
                self._handle_state_transition(new_state)

            self.current_state = new_state

            # 發布結果
            self._publish_detection_results(analysis_result)

            # 視覺化
            if self.enable_visualization:
                self._publish_visualization(
                    detection_points, analysis_result, msg.header)

        except Exception as ex:
            self.get_logger().error(f'點雲處理錯誤: {str(ex)}')

    def _extract_and_transform_points(self, msg, transform):
        """提取並轉換點雲"""
        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            p = np.array([point[0], point[1], point[2], 1.0])
            points.append(p)

        if len(points) == 0:
            return np.array([])

        points_array = np.array(points)
        transformed_points = self._apply_transform(points_array, transform)
        return transformed_points[:, 0:3]  # 只保留x,y,z

    def _apply_transform(self, points, transform):
        """應用TF轉換"""
        q = transform.transform.rotation
        t = transform.transform.translation

        # 四元數轉旋轉矩陣
        r = np.array([
            [1-2*(q.y*q.y+q.z*q.z), 2*(q.x*q.y-q.z*q.w), 2*(q.x*q.z+q.y*q.w), 0],
            [2*(q.x*q.y+q.z*q.w), 1-2*(q.x*q.x+q.z*q.z), 2*(q.y*q.z-q.x*q.w), 0],
            [2*(q.x*q.z-q.y*q.w), 2*(q.y*q.z+q.x*q.w), 1-2*(q.x*q.x+q.y*q.y), 0],
            [0, 0, 0, 1]
        ])

        r[0, 3] = t.x
        r[1, 3] = t.y
        r[2, 3] = t.z

        return np.dot(points, r.T)

    def _filter_detection_area(self, points):
        """濾除檢測區域內的點"""
        # 定義檢測區域邊界
        x_min, x_max = self.detection_min_x, self.detection_max_x

        # 使用整個Y軸範圍
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

        # 濾除條件
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        )

        return points[mask]

    def _analyze_height_variation(self, points):
        """分析高度變化"""
        # 加入數據清理
        if len(points) == 0:
            return self._create_empty_analysis_result()

        # 清理無效數值
        valid_mask = np.isfinite(points).all(axis=1)
        clean_points = points[valid_mask]

        if len(clean_points) == 0:
            return self._create_empty_analysis_result()

        # 使用Open3D進行體素下採樣
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(clean_points)
        pcd_voxel = pcd.voxel_down_sample(self.voxel_size)
        voxel_points = np.asarray(pcd_voxel.points)

        if len(voxel_points) == 0:
            return self._create_empty_analysis_result()

        # 計算切片數量
        x_range = self.detection_max_x - self.detection_min_x
        num_slices = int(x_range / self.slice_thickness)

        slice_analysis = []
        overall_heights = []

        for i in range(num_slices):
            x_start = self.detection_min_x + i * self.slice_thickness
            x_end = x_start + self.slice_thickness

            # 獲取當前切片的點
            slice_mask = (voxel_points[:, 0] >= x_start) & (
                voxel_points[:, 0] < x_end)
            slice_points = voxel_points[slice_mask]

            if len(slice_points) < self.min_points_per_slice:
                continue  # 跳過點數不足的切片

            # 計算高度統計，重點加入四分位距
            heights = slice_points[:, 2]
            q1 = np.percentile(heights, 25)
            q3 = np.percentile(heights, 75)
            iqr = q3 - q1  # 四分位距作為關鍵指標

            slice_stats = {
                'x_center': (x_start + x_end) / 2,
                'mean_height': np.mean(heights),
                'std_height': np.std(heights),
                'q1': q1,
                'q3': q3,
                'iqr': iqr,  # 確保這個鍵存在
                'min_height': np.min(heights),
                'max_height': np.max(heights),
                'height_range': np.max(heights) - np.min(heights),
                'point_count': len(slice_points),
                'points': slice_points
            }

            slice_analysis.append(slice_stats)
            overall_heights.extend(heights)

        if len(slice_analysis) == 0:
            return self._create_empty_analysis_result()

        # 計算整體統計
        overall_stats = {
            'mean_height': np.mean(overall_heights),
            'std_height': np.std(overall_heights),
            'height_range': np.max(overall_heights) - np.min(overall_heights)
        }

        # 計算切片間的高度變化和IQR統計
        mean_heights = [s['mean_height'] for s in slice_analysis]
        iqrs = [s['iqr'] for s in slice_analysis]

        height_variation_score = np.std(mean_heights)  # 切片間平均高度的變化
        avg_iqr = np.mean(iqrs)                       # 平均四分位距（關鍵指標）

        return {
            'slice_analysis': slice_analysis,
            'overall_stats': overall_stats,
            'height_variation_score': height_variation_score,
            'avg_iqr': avg_iqr,
            'total_points': len(voxel_points),
            'valid_slices': len(slice_analysis)
        }

    def _create_empty_analysis_result(self):
        """創建空的分析結果"""
        return {
            'slice_analysis': [],
            'overall_stats': {'mean_height': 0, 'std_height': 0, 'height_range': 0},
            'height_variation_score': 0,
            'avg_iqr': 0,
            'total_points': 0,
            'valid_slices': 0
        }

    def _determine_state(self):
        """根據歷史記錄判斷當前狀態"""
        if len(self.detection_history) < 3:  # 需要足夠的歷史數據
            return self.current_state

        # 統計最近的檢測結果
        recent_results = list(self.detection_history)[-3:]  # 最近3次結果

        in_field_count = 0
        out_field_count = 0

        for result in recent_results:
            avg_iqr = result['avg_iqr']  # 使用avg_iqr而不是iqr

            if avg_iqr > self.iqr_threshold:
                in_field_count += 1  # 有明顯高度變化，在田內
            else:
                out_field_count += 1  # 高度平緩，在田外

        # 根據統計結果判斷
        total_count = len(recent_results)
        in_field_confidence = in_field_count / total_count
        out_field_confidence = out_field_count / total_count

        if in_field_confidence >= self.confidence_threshold:
            return "in_field"
        elif out_field_confidence >= self.confidence_threshold:
            return "out_field"
        else:
            return self.current_state  # 維持當前狀態

    def _handle_state_transition(self, new_state):
        """處理狀態轉換"""
        transition = f"{self.current_state} -> {new_state}"

        # 發布轉換事件
        transition_msg = String()
        transition_msg.data = transition
        self.transition_pub.publish(transition_msg)

        # 記錄重要的轉換
        if self.current_state == "out_field" and new_state == "in_field":
            self.get_logger().info("檢測到入行：開始進入田地")
        elif self.current_state == "in_field" and new_state == "out_field":
            self.get_logger().info("檢測到出行：離開田地")

        self.get_logger().info(f"狀態轉換: {transition}")
        self.previous_state = self.current_state

    def _publish_detection_results(self, analysis_result):
        """發布檢測結果"""
        # 發布當前狀態
        state_msg = String()
        state_msg.data = self.current_state
        self.state_pub.publish(state_msg)

        # 發布調試資訊
        debug_info = {
            'current_state': self.current_state,
            'avg_iqr': analysis_result['avg_iqr'],
            'height_variation_score': analysis_result['height_variation_score'],
            'valid_slices': analysis_result['valid_slices'],
            'total_points': analysis_result['total_points'],
            'threshold': self.iqr_threshold
        }

        debug_msg = String()
        debug_msg.data = json.dumps(debug_info, indent=2)
        self.debug_pub.publish(debug_msg)

    def _publish_visualization(self, detection_points, analysis_result, header):
        """發布視覺化標記"""
        marker_array = MarkerArray()

        # 檢測區域邊界框
        boundary_marker = self._create_boundary_marker(header)
        marker_array.markers.append(boundary_marker)

        # 切片結果視覺化
        for i, slice_data in enumerate(analysis_result['slice_analysis']):
            slice_marker = self._create_slice_marker(slice_data, i, header)
            marker_array.markers.append(slice_marker)

        # 狀態指示
        state_marker = self._create_state_marker(header)
        marker_array.markers.append(state_marker)

        self.marker_pub.publish(marker_array)

    def _create_boundary_marker(self, header):
        """創建檢測區域邊界標記"""
        marker = Marker()
        marker.header = header
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # 使用整個Y軸範圍作為檢測區域寬度
        # 這裡我們用一個合理的預設範圍來繪製邊界
        detection_width = 2.0  # 預設繪製寬度，實際檢測使用完整Y範圍

        # 設定邊界框的四個角點
        corners = [
            [self.detection_min_x, -detection_width/2, 0.0],
            [self.detection_max_x, -detection_width/2, 0.0],
            [self.detection_max_x, detection_width/2, 0.0],
            [self.detection_min_x, detection_width/2, 0.0],
            [self.detection_min_x, -detection_width/2, 0.0]  # 閉合
        ]

        for corner in corners:
            p = Point()
            p.x, p.y, p.z = corner
            marker.points.append(p)

        marker.scale.x = 0.02
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        return marker

    def _create_slice_marker(self, slice_data, slice_id, header):
        """創建切片視覺化標記"""
        marker = Marker()
        marker.header = header
        marker.id = slice_id + 1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # 位置設定
        marker.pose.position.x = slice_data['x_center']
        marker.pose.position.y = 0.0
        marker.pose.position.z = slice_data['mean_height']
        marker.pose.orientation.w = 1.0

        # 大小設定
        marker.scale.x = self.slice_thickness
        marker.scale.y = 2.0  # 使用固定寬度來繪製，實際檢測使用完整Y範圍
        marker.scale.z = 0.02

        # 根據IQR設定顏色
        iqr_ratio = min(slice_data['iqr'] / self.iqr_threshold, 1.0)
        marker.color.r = iqr_ratio
        marker.color.g = 1.0 - iqr_ratio
        marker.color.b = 0.0
        marker.color.a = 0.3

        return marker

    def _create_state_marker(self, header):
        """創建狀態指示標記"""
        marker = Marker()
        marker.header = header
        marker.id = 1000
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = (
            self.detection_min_x + self.detection_max_x) / 2
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.0
        marker.pose.orientation.w = 1.0

        marker.text = f"狀態: {self.current_state}"
        marker.scale.z = 0.2

        # 根據狀態設定顏色
        if self.current_state == "in_field":
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        elif self.current_state == "out_field":
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
        else:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0

        marker.color.a = 1.0

        return marker


def main(args=None):
    rclpy.init(args=args)

    node = FurrowEndDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
