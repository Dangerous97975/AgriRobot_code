#!/usr/bin/env python3
# ccpp_generator.py

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
import time
import cv2  # 使用OpenCV進行更好的邊界檢測


class CCPPGenerator(Node):
    def __init__(self):
        super().__init__('ccpp_generator')

        # 聲明參數
        self.declare_parameter('field_length', 42.0)    # 溫室長
        self.declare_parameter('field_width', 12.0)     # 溫室寬
        self.declare_parameter('no_go_zone', 4.0)       # 忽略區域寬
        self.declare_parameter('num_ridges', 10)       # 土畦數量
        self.declare_parameter('safety_margin', 0.2)
        self.declare_parameter('wait_for_map_timeout', 10.0)

        # 獲取參數
        self.field_length = self.get_parameter('field_length').value
        self.field_width = self.get_parameter('field_width').value
        self.no_go_zone = self.get_parameter('no_go_zone').value
        self.num_ridges = self.get_parameter('num_ridges').value
        self.safety_margin = self.get_parameter('safety_margin').value
        self.wait_timeout = self.get_parameter('wait_for_map_timeout').value

        # 創建GetMap服務客戶端
        self.map_client = self.create_client(GetMap, '/map_server/map')

        # 訂閱地圖主題
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)

        # 發布路徑
        self.path_publisher = self.create_publisher(Path, '/coverage_path', 10)
        self.marker_publisher = self.create_publisher(
            MarkerArray, '/coverage_markers', 10)

        self.map_received = False
        self.get_logger().info('CCPP Generator 已啟動，等待地圖服務...')

        # 創建定時器來嘗試獲取地圖
        self.create_timer(1.0, self.try_get_map)

    def try_get_map(self):
        """嘗試通過服務獲取地圖"""
        if self.map_received:
            return

        if not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('地圖服務尚未就緒，繼續等待...')
            return

        request = GetMap.Request()
        future = self.map_client.call_async(request)
        future.add_done_callback(self.handle_get_map_response)

    def handle_get_map_response(self, future):
        """處理GetMap服務響應"""
        try:
            response = future.result()
            if response is not None:
                self.get_logger().info('成功通過服務獲取地圖！')
                self.process_map(response.map)
        except Exception as e:
            self.get_logger().error(f'獲取地圖服務失敗: {e}')

    def map_callback(self, msg):
        """地圖主題回調"""
        if not self.map_received:
            self.get_logger().info('通過主題接收到地圖')
            self.process_map(msg)

    def process_map(self, map_msg):
        """處理地圖數據"""
        if self.map_received:
            return

        self.map_received = True
        self.map = map_msg

        # 地圖參數
        self.map_origin_x = map_msg.info.origin.position.x
        self.map_origin_y = map_msg.info.origin.position.y
        self.map_resolution = map_msg.info.resolution
        self.map_width = map_msg.info.width
        self.map_height = map_msg.info.height

        self.get_logger().info(
            f'地圖原點: ({self.map_origin_x:.3f}, {self.map_origin_y:.3f})')
        self.get_logger().info(f'地圖解析度: {self.map_resolution:.3f} m/pixel')
        self.get_logger().info(
            f'地圖尺寸: {self.map_width}x{self.map_height} pixels')

        # 找出溫室在地圖中的位置
        self.find_greenhouse_boundaries()

        self.generate_coverage_path()

    def find_greenhouse_boundaries(self):
        """從佔據網格地圖中找出溫室的邊界輪廓"""
        # 將地圖數據轉換為2D陣列
        grid = np.array(self.map.data, dtype=np.int8).reshape(
            self.map_height, self.map_width)

        # 轉換為OpenCV可處理的圖像格式
        img = np.zeros((self.map_height, self.map_width), dtype=np.uint8)
        img.fill(127)  # 未知區域為灰色
        img[grid == 0] = 255  # 自由空間為白色
        img[grid == 100] = 0  # 佔據空間為黑色

        # 保存一份原始圖像用於調試
        self.original_map_img = img.copy()

        # 進行圖像處理以找到邊界
        # 1. 使用高斯模糊減少噪點
        blur = cv2.GaussianBlur(img, (7, 7), 0)  # 增加模糊半徑以更好地處理噪點

        # 2. 使用Canny邊緣檢測找出邊緣
        edges = cv2.Canny(blur, 30, 100)  # 降低閾值以捕獲更多邊緣

        # 保存邊緣圖像用於調試
        self.edges_img = edges.copy()

        # 3. 使用膨脹操作連接邊緣的斷點
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # 4. 找出輪廓
        contours, _ = cv2.findContours(
            dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 保存所有找到的輪廓以進行調試
        self.all_contours = contours

        if len(contours) > 0:
            # 過濾掉面積過小的輪廓
            min_contour_area = 100  # 最小面積閾值
            valid_contours = [
                c for c in contours if cv2.contourArea(c) > min_contour_area]

            if len(valid_contours) > 0:
                # 找出最大的輪廓（假設是溫室的邊界）
                largest_contour = max(valid_contours, key=cv2.contourArea)
                self.largest_contour = largest_contour  # 保存最大輪廓用於調試

                # 5. 使用多邊形近似簡化輪廓
                epsilon = 0.01 * \
                    cv2.arcLength(largest_contour, True)  # 多邊形近似的精度參數
                approx_polygon = cv2.approxPolyDP(
                    largest_contour, epsilon, True)
                self.approx_polygon = approx_polygon  # 保存近似多邊形用於調試

                # 6. 如果頂點數過多，使用最小面積矩形代替
                if len(approx_polygon) > 8:  # 如果頂點太多，可能不是一個簡單的矩形
                    self.get_logger().info(
                        f'多邊形頂點數過多 ({len(approx_polygon)}), 使用最小面積矩形')
                    rect = cv2.minAreaRect(largest_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    self.min_area_rect = box  # 保存最小面積矩形用於調試
                    corners_pixels = box
                else:
                    corners_pixels = approx_polygon.squeeze()
                    if corners_pixels.ndim == 1:
                        corners_pixels = np.expand_dims(corners_pixels, axis=0)
                    self.get_logger().info(
                        f'使用多邊形近似，頂點數: {len(corners_pixels)}')

                # 7. 將像素坐標轉換為世界坐標
                corners_world = []
                for point in corners_pixels:
                    x_world = self.map_origin_x + \
                        int(point[0]) * self.map_resolution
                    y_world = self.map_origin_y + \
                        int(point[1]) * self.map_resolution
                    corners_world.append((x_world, y_world))

                # 存儲角點坐標
                self.greenhouse_corners_pixels = corners_pixels
                self.greenhouse_corners_world = corners_world

                # 輸出角點信息
                self.get_logger().info(f'溫室角點（世界坐標）:')
                for i, corner in enumerate(corners_world):
                    self.get_logger().info(
                        f'  角點 {i+1}: ({corner[0]:.3f}, {corner[1]:.3f})')

                # 8. 計算邊界框（用於後備）
                x, y, w, h = cv2.boundingRect(largest_contour)

                # 將像素坐標轉換為世界坐標
                self.greenhouse_min_x = self.map_origin_x + x * self.map_resolution
                self.greenhouse_max_x = self.map_origin_x + \
                    (x + w) * self.map_resolution
                self.greenhouse_min_y = self.map_origin_y + y * self.map_resolution
                self.greenhouse_max_y = self.map_origin_y + \
                    (y + h) * self.map_resolution

                # 輸出找到的邊界信息
                self.get_logger().info(f'溫室邊界框:')
                self.get_logger().info(
                    f'  X: [{self.greenhouse_min_x:.3f}, {self.greenhouse_max_x:.3f}]')
                self.get_logger().info(
                    f'  Y: [{self.greenhouse_min_y:.3f}, {self.greenhouse_max_y:.3f}]')
#
                # 驗證尺寸
                detected_width = self.greenhouse_max_x - self.greenhouse_min_x
                detected_height = self.greenhouse_max_y - self.greenhouse_min_y
                self.get_logger().info(
                    f'檢測到的邊界框尺寸: {detected_width:.1f}m x {detected_height:.1f}m')

                if abs(detected_width - self.field_length) > 10.0 or abs(detected_height - self.field_width) > 10.0:
                    self.get_logger().warn(f'檢測到的尺寸與預期尺寸差異較大，可能不準確')
                    self.get_logger().warn(
                        f'預期尺寸: {self.field_length:.1f}m x {self.field_width:.1f}m')
            else:
                self.get_logger().warn('沒有找到面積足夠大的輪廓，使用預設值')
                self._use_default_boundaries()
        else:
            self.get_logger().warn('無法在地圖中找到邊界輪廓，使用預設值')
            self._use_default_boundaries()

        # 添加視覺化標記
        self.publish_boundary_markers()

    def generate_coverage_path(self):
        """基於檢測到的溫室角點生成S形覆蓋路徑，參考Farmland-Path-Planning專案"""
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        # 確認已經檢測到溫室角點
        if not hasattr(self, 'greenhouse_corners_world') or len(self.greenhouse_corners_world) < 4:
            self.get_logger().warn('無溫室角點資訊，無法生成路徑')
            return

        corners = self.greenhouse_corners_world

        # 計算邊長以確定長邊和短邊
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        # 計算四條邊的長度
        sides = [
            distance(corners[0], corners[1]),
            distance(corners[1], corners[2]),
            distance(corners[2], corners[3]),
            distance(corners[3], corners[0])
        ]

        # 找出相鄰的邊（兩條短邊和兩條長邊）
        side_pairs = [(0, 2), (1, 3)]  # 假設矩形中對邊平行且等長

        # 確定哪一對是長邊
        if sides[side_pairs[0][0]] + sides[side_pairs[0][1]] > sides[side_pairs[1][0]] + sides[side_pairs[1][1]]:
            long_sides_indices = side_pairs[0]
            short_sides_indices = side_pairs[1]
        else:
            long_sides_indices = side_pairs[1]
            short_sides_indices = side_pairs[0]

        # 獲取長邊的矢量方向
        long_side_index = long_sides_indices[0]
        long_edge_start = corners[long_side_index]
        long_edge_end = corners[(long_side_index + 1) % 4]
        long_edge_vec = np.array([long_edge_end[0] - long_edge_start[0],
                                  long_edge_end[1] - long_edge_start[1]])
        long_edge_length = np.linalg.norm(long_edge_vec)
        long_edge_dir = long_edge_vec / long_edge_length

        # 獲取短邊的矢量方向
        short_side_index = short_sides_indices[0]
        short_edge_start = corners[short_side_index]
        short_edge_end = corners[(short_side_index + 1) % 4]
        short_edge_vec = np.array([short_edge_end[0] - short_edge_start[0],
                                   short_edge_end[1] - short_edge_start[1]])
        short_edge_length = np.linalg.norm(short_edge_vec)
        short_edge_dir = short_edge_vec / short_edge_length

        self.get_logger().info(
            f'長邊長度: {long_edge_length:.2f}m, 短邊長度: {short_edge_length:.2f}m')

        # 計算實際工作區域（考慮不可通行區域和安全邊距）
        no_go_zone_vec = self.no_go_zone * long_edge_dir
        safety_margin_vec_long = self.safety_margin * long_edge_dir
        safety_margin_vec_short = self.safety_margin * short_edge_dir

        # 計算矩形的四個頂點（按順時針或逆時針順序）
        field_corners = [
            corners[0],
            corners[1],
            corners[2],
            corners[3]
        ]

        # 確定工作區域的角點（考慮不可通行區域和安全邊距）
        work_area_corners = [
            np.array(field_corners[0]) + no_go_zone_vec
            + safety_margin_vec_short,
            np.array(field_corners[1]) + no_go_zone_vec
            - safety_margin_vec_short,
            np.array(field_corners[2])
            - safety_margin_vec_long - safety_margin_vec_short,
            np.array(field_corners[3])
            - safety_margin_vec_long + safety_margin_vec_short,
        ]

        length_1 = distance(work_area_corners[0], work_area_corners[3])
        length_2 = distance(work_area_corners[0], work_area_corners[1])

        # 計算工作區域的尺寸
        if length_1 > length_2:
            work_area_length = length_1
            work_area_width = length_2
        else:
            work_area_length = length_2
            work_area_width = length_1

        self.get_logger().info(
            f'工作區域尺寸: 長={work_area_length:.2f}m, 寬={work_area_width:.2f}m')

        # 檢查工作區域是否有效
        if work_area_length <= 0 or work_area_width <= 0:
            self.get_logger().error(f'工作區域無效，請檢查參數設置')
            return

        # 使用固定的土畦數量
        num_lanes = self.num_ridges
        self.get_logger().info(f'使用固定的土畦數量: {num_lanes}')

        # 調整車道間距以均勻分布
        lane_width = work_area_width / num_lanes
        self.get_logger().info(f'計算得到的車道寬度: {lane_width:.2f}m')

        # S形路徑生成
        waypoints = []
        path_points = []

        # 選擇起始點
        start_corner = work_area_corners[0]

        # 計算車道方向向量 (從下到上)
        lane_dir = (work_area_corners[3] - work_area_corners[0]) / \
            np.linalg.norm(work_area_corners[3] - work_area_corners[0])

        # 計算橫向移動向量 (從左到右)
        side_dir = (work_area_corners[1] - work_area_corners[0]) / \
            np.linalg.norm(work_area_corners[1] - work_area_corners[0])

        for lane in range(num_lanes):
            # 計算當前車道的中心線位置
            lane_offset = (lane + 0.5) * lane_width

            # 計算當前車道的起點和終點
            if lane % 2 == 0:  # 偶數車道 (從下往上)
                lane_start = start_corner + side_dir * lane_offset
                lane_end = lane_start + lane_dir * work_area_length
                lane_direction = lane_dir
            else:  # 奇數車道 (從上往下)
                lane_start = start_corner + side_dir * lane_offset + lane_dir * work_area_length
                lane_end = lane_start - lane_dir * work_area_length
                lane_direction = -lane_dir

            # 添加車道起點
            pose_start = PoseStamped()
            pose_start.header = path.header
            pose_start.pose.position.x = lane_start[0]
            pose_start.pose.position.y = lane_start[1]
            pose_start.pose.position.z = 0.0
            quat = self._get_quaternion_from_direction(lane_direction)
            pose_start.pose.orientation.x = quat[0]
            pose_start.pose.orientation.y = quat[1]
            pose_start.pose.orientation.z = quat[2]
            pose_start.pose.orientation.w = quat[3]

            path.poses.append(pose_start)
            waypoints.append(pose_start)
            path_points.append((lane_start[0], lane_start[1]))

            # 添加車道路徑點
            num_points = max(2, int(work_area_length / 0.5))  # 每0.5米添加一個點
            for i in range(1, num_points - 1):
                t = i / (num_points - 1)
                point = lane_start * (1 - t) + lane_end * t

                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = point[0]
                pose.pose.position.y = point[1]
                pose.pose.position.z = 0.0
                quat = self._get_quaternion_from_direction(lane_direction)
                pose.pose.orientation.x = quat[0]
                pose.pose.orientation.y = quat[1]
                pose.pose.orientation.z = quat[2]
                pose.pose.orientation.w = quat[3]

                path.poses.append(pose)
                path_points.append((point[0], point[1]))

            # 添加車道終點
            pose_end = PoseStamped()
            pose_end.header = path.header
            pose_end.pose.position.x = lane_end[0]
            pose_end.pose.position.y = lane_end[1]
            pose_end.pose.position.z = 0.0
            quat = self._get_quaternion_from_direction(lane_direction)
            pose_end.pose.orientation.x = quat[0]
            pose_end.pose.orientation.y = quat[1]
            pose_end.pose.orientation.z = quat[2]
            pose_end.pose.orientation.w = quat[3]

            path.poses.append(pose_end)
            waypoints.append(pose_end)
            path_points.append((lane_end[0], lane_end[1]))

            # 如果不是最後一條車道，添加轉彎路徑
            if lane < num_lanes - 1:
                next_lane_offset = (lane + 1 + 0.5) * lane_width

                if lane % 2 == 0:  # 從下往上的車道，到上面後向右轉
                    turn_start = lane_end
                    turn_end = start_corner + side_dir * \
                        next_lane_offset + lane_dir * work_area_length
                else:  # 從上往下的車道，到下面後向右轉
                    turn_start = lane_end
                    turn_end = start_corner + side_dir * next_lane_offset

                # 添加轉彎路徑點
                turn_points = 10
                for i in range(1, turn_points):
                    t = i / turn_points
                    turn_point = turn_start * (1 - t) + turn_end * t

                    # 計算轉彎過程中的方向
                    if lane % 2 == 0:  # 向右上轉彎
                        turn_dir = lane_dir * (1 - t) + side_dir * t
                    else:  # 向右下轉彎
                        turn_dir = -lane_dir * (1 - t) + side_dir * t

                    turn_dir = turn_dir / np.linalg.norm(turn_dir)

                    pose = PoseStamped()
                    pose.header = path.header
                    pose.pose.position.x = turn_point[0]
                    pose.pose.position.y = turn_point[1]
                    pose.pose.position.z = 0.0
                    quat = self._get_quaternion_from_direction(turn_dir)
                    pose.pose.orientation.x = quat[0]
                    pose.pose.orientation.y = quat[1]
                    pose.pose.orientation.z = quat[2]
                    pose.pose.orientation.w = quat[3]

                    path.poses.append(pose)
                    path_points.append((turn_point[0], turn_point[1]))

        self.get_logger().info(
            f'生成了 {len(waypoints)} 個關鍵航點，總共 {len(path.poses)} 個路徑點')

        self.work_area_corners = [
            (work_area_corners[0][0], work_area_corners[0][1]),
            (work_area_corners[1][0], work_area_corners[1][1]),
            (work_area_corners[2][0], work_area_corners[2][1]),
            (work_area_corners[3][0], work_area_corners[3][1])
        ]

        # 發布路徑
        self.path_publisher.publish(path)

        # 保存路徑到CSV文件
        self.save_path_to_csv(path)

        self.plot_map_and_path(path)

        # 創建可視化標記
        self.publish_markers(waypoints)

        # 定期重新發布路徑
        if not hasattr(self, 'path_timer'):
            self.path_timer = self.create_timer(
                1.0, lambda: self.path_publisher.publish(path))

    def _get_quaternion_from_direction(self, direction):
        """從方向向量計算四元數"""
        # 計算yaw角（繞Z軸的旋轉）
        yaw = np.arctan2(direction[1], direction[0])

        # 從yaw角計算四元數
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        # 假設繞X和Y軸的旋轉為0
        return [0.0, 0.0, sy, cy]

    def _use_default_boundaries(self):
        """使用預設邊界值"""
        self.greenhouse_min_x = self.map_origin_x
        self.greenhouse_max_x = self.map_origin_x + self.field_length
        self.greenhouse_min_y = self.map_origin_y
        self.greenhouse_max_y = self.map_origin_y + self.field_width

        # 創建預設的角點
        self.greenhouse_corners_world = [
            (self.greenhouse_min_x, self.greenhouse_min_y),
            (self.greenhouse_max_x, self.greenhouse_min_y),
            (self.greenhouse_max_x, self.greenhouse_max_y),
            (self.greenhouse_min_x, self.greenhouse_max_y)
        ]

        self.get_logger().info(
            f'使用預設溫室邊界: ({self.greenhouse_min_x}, {self.greenhouse_min_y}) - ({self.greenhouse_max_x}, {self.greenhouse_max_y})')

    def publish_markers(self, waypoints):
        """發布可視化標記"""
        marker_array = MarkerArray()

        # 路徑線條標記
        line_marker = Marker()
        line_marker.header.frame_id = "map"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = "coverage_path"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.1
        line_marker.color.a = 1.0
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0

        for waypoint in waypoints:
            line_marker.points.append(waypoint.pose.position)

        marker_array.markers.append(line_marker)

        # 添加工作區域邊界
        if hasattr(self, 'work_area_corners') and len(self.work_area_corners) >= 4:
            # 顯示工作區域邊界
            work_area_marker = Marker()
            work_area_marker.header.frame_id = "map"
            work_area_marker.header.stamp = self.get_clock().now().to_msg()
            work_area_marker.ns = "work_area"
            work_area_marker.id = 1
            work_area_marker.type = Marker.LINE_STRIP
            work_area_marker.action = Marker.ADD
            work_area_marker.scale.x = 0.12
            work_area_marker.color.a = 1.0
            work_area_marker.color.r = 1.0
            work_area_marker.color.g = 0.5
            work_area_marker.color.b = 0.0

            # 工作區域的四個角加上閉合
            corners = self.work_area_corners.copy()
            corners.append(corners[0])  # 閉合

            for corner in corners:
                p = PoseStamped().pose.position
                p.x = corner[0]
                p.y = corner[1]
                p.z = 0.05
                work_area_marker.points.append(p)

            marker_array.markers.append(work_area_marker)

        # 添加溫室邊界
        if hasattr(self, 'greenhouse_corners_world') and len(self.greenhouse_corners_world) >= 4:
            # 顯示檢測到的溫室邊界
            greenhouse_marker = Marker()
            greenhouse_marker.header.frame_id = "map"
            greenhouse_marker.header.stamp = self.get_clock().now().to_msg()
            greenhouse_marker.ns = "greenhouse"
            greenhouse_marker.id = 2
            greenhouse_marker.type = Marker.LINE_STRIP
            greenhouse_marker.action = Marker.ADD
            greenhouse_marker.scale.x = 0.15
            greenhouse_marker.color.a = 1.0
            greenhouse_marker.color.r = 0.0
            greenhouse_marker.color.g = 0.0
            greenhouse_marker.color.b = 1.0

            # 溫室的四個角加上閉合
            corners = self.greenhouse_corners_world.copy()
            corners.append(corners[0])  # 閉合

            for corner in corners:
                p = PoseStamped().pose.position
                p.x = corner[0]
                p.y = corner[1]
                p.z = 0.05
                greenhouse_marker.points.append(p)

            marker_array.markers.append(greenhouse_marker)

        # 發布標記
        self.marker_publisher.publish(marker_array)

    def publish_boundary_markers(self):
        """發布溫室邊界角點和輪廓的視覺化標記"""
        marker_array = MarkerArray()

        # 如果檢測到了溫室角點
        if hasattr(self, 'greenhouse_corners_world'):
            # 1. 角點標記
            corners_marker = Marker()
            corners_marker.header.frame_id = "map"
            corners_marker.header.stamp = self.get_clock().now().to_msg()
            corners_marker.ns = "greenhouse_corners"
            corners_marker.id = 10
            corners_marker.type = Marker.SPHERE_LIST
            corners_marker.action = Marker.ADD
            corners_marker.scale.x = 0.3
            corners_marker.scale.y = 0.3
            corners_marker.scale.z = 0.3
            corners_marker.color.a = 1.0
            corners_marker.color.r = 1.0
            corners_marker.color.g = 0.0
            corners_marker.color.b = 0.0

            for corner in self.greenhouse_corners_world:
                point = PoseStamped().pose.position
                point.x = corner[0]
                point.y = corner[1]
                point.z = 0.1
                corners_marker.points.append(point)

            marker_array.markers.append(corners_marker)

            # 2. 輪廓線標記
            contour_marker = Marker()
            contour_marker.header.frame_id = "map"
            contour_marker.header.stamp = self.get_clock().now().to_msg()
            contour_marker.ns = "greenhouse_contour"
            contour_marker.id = 11
            contour_marker.type = Marker.LINE_STRIP
            contour_marker.action = Marker.ADD
            contour_marker.scale.x = 0.1
            contour_marker.color.a = 1.0
            contour_marker.color.r = 0.0
            contour_marker.color.g = 1.0
            contour_marker.color.b = 1.0

            # 添加所有角點並閉合輪廓
            corners = self.greenhouse_corners_world.copy()
            corners.append(corners[0])  # 閉合輪廓

            for corner in corners:
                point = PoseStamped().pose.position
                point.x = corner[0]
                point.y = corner[1]
                point.z = 0.05
                contour_marker.points.append(point)

            marker_array.markers.append(contour_marker)

        # 發布標記
        self.marker_publisher.publish(marker_array)

        # 設置定時器定期發布標記
        if not hasattr(self, 'boundary_marker_timer'):
            self.boundary_marker_timer = self.create_timer(
                1.0, lambda: self.marker_publisher.publish(marker_array))

    def plot_map_and_path(self, path):
        """使用matplotlib繪製地圖邊界和CCPP路徑"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # 創建一個新的圖表
            plt.figure(figsize=(10, 8))

            # 繪製溫室邊界
            if hasattr(self, 'greenhouse_corners_world') and len(self.greenhouse_corners_world) >= 3:
                # 複製角點並加上第一個點以閉合多邊形
                corners = self.greenhouse_corners_world.copy()
                corners.append(corners[0])

                # 轉換為x和y座標列表
                x_corners = [p[0] for p in corners]
                y_corners = [p[1] for p in corners]

                # 繪製溫室邊界（藍色線）
                plt.plot(x_corners, y_corners, 'b-', linewidth=2,
                         label='Greenhouse boundary')

                # 填充溫室區域（半透明藍色）
                plt.fill(x_corners, y_corners, 'b', alpha=0.1)

                # 標記角點
                for i, corner in enumerate(self.greenhouse_corners_world):
                    plt.plot(corner[0], corner[1], 'ro', markersize=6)
                    plt.text(corner[0], corner[1], f'C{i}', fontsize=10)

            # 檢查是否有路徑數據可用
            if len(path.poses) > 0:
                # 使用保存的路徑數據
                x_path = [pose.pose.position.x for pose in path.poses]
                y_path = [pose.pose.position.y for pose in path.poses]

                # 繪製路徑（綠色線）
                plt.plot(x_path, y_path, 'g-', linewidth=1.5,
                         label='coverage path')

                # 標記起點和終點
                plt.plot(x_path[0], y_path[0], 'go',
                         markersize=8, label='start')
                plt.plot(x_path[-1], y_path[-1], 'mo',
                         markersize=8, label='end')
            else:
                self.get_logger().warn('沒有可用的路徑數據用於繪圖')

            # 設定圖表標題和軸標籤
            plt.title('Greenhouse boundary & CCPP')
            plt.xlabel('X coordinate (m)')
            plt.ylabel('Y coordinate (m)')
            plt.axis('equal')  # 保持X和Y軸比例相同
            plt.grid(True)
            plt.legend()

            # 保存圖表到文件
            timestamp = self.get_clock().now().to_msg().sec
            filename = f'greenhouse_ccpp_{timestamp}.png'
            plt.savefig(filename)
            self.get_logger().info(f'已保存圖表到: {filename}')

            # 顯示圖表
            plt.show()

        except ImportError:
            self.get_logger().warn('無法繪製圖表 - matplotlib可能未安裝')
        except Exception as e:
            self.get_logger().error(f'繪製圖表時發生錯誤: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def save_path_to_csv(self, path, filename='coverage_path.csv'):
        """將路徑保存為CSV文件"""
        import csv

        self.get_logger().info(f'正在將路徑保存為CSV文件: {filename}')

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 寫入標題
            writer.writerow(['index', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

            # 寫入路徑點
            for i, pose in enumerate(path.poses):
                writer.writerow([
                    i,
                    pose.pose.position.x,
                    pose.pose.position.y,
                    pose.pose.position.z,
                    pose.pose.orientation.x,
                    pose.pose.orientation.y,
                    pose.pose.orientation.z,
                    pose.pose.orientation.w
                ])

        self.get_logger().info(f'路徑已保存到: {filename}')


def main(args=None):
    rclpy.init(args=args)

    # 等待一下讓其他節點啟動
    time.sleep(2.0)

    node = CCPPGenerator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
