import cv2
import rclpy
import numpy as np
from rclpy.node import Node
import tf2_ros
from std_msgs.msg import UInt16MultiArray, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from cv_bridge import CvBridge
import time


class TrajectoryPlanNode(Node):
    def __init__(self):
        super().__init__("trajectory_plan_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.current_depth = None

        self._declare_parameters()
        self._init_variables()

        # Create a subscriber to the plant_cord topic
        self.cord_subscriber = self.create_subscription(
            UInt16MultiArray, 'removed_cords', self.cords_callback, 10)

        self.depth_subscriber = self.create_subscription(
            Image, '/agri_bot/D455f/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        # Create a publisher to the delta_cord topic
        self.cord_publisher = self.create_publisher(
            Float32MultiArray, 'delta_x/target_array', 10)

    def _declare_parameters(self):
        """宣告ROS2參數"""
        self.declare_parameter('delta_working_level', [-550.0, -625.0, -550.0])
        self.declare_parameter('K', [0.0])
        self.declare_parameter('D', [0.0])
        self.declare_parameter('rvec', [0.0])

    def _init_variables(self):
        """取得ROS2參數"""
        delta_working_level = self.get_parameter(
            'delta_working_level').get_parameter_value().double_array_value
        D = self.get_parameter(
            'D').get_parameter_value().double_array_value
        K = self.get_parameter(
            'K').get_parameter_value().double_array_value
        rvec = self.get_parameter(
            'rvec').get_parameter_value().double_array_value
        # self.tvec = self._get_camera2delta_tf()

        self.delta_working_level = np.array(delta_working_level)
        self.D = np.array(D)
        self.K = np.array(K).reshape(3, 3)  # 相機內參矩陣通常是 3x3
        self.rvec = np.array(rvec)

        # 印出參數值確認
        self.get_logger().info(f'Delta工作高度: {self.delta_working_level}')
        self.get_logger().info(f'畸變矩陣: {self.D}')
        self.get_logger().info(f'相機矩陣: {self.K}')
        self.get_logger().info(f'剛體旋轉矩陣: {self.rvec}')

    def _get_camera2delta_tf(self):
        """取得D455f與Delta的平移向量"""
        delta1_tf = self.tf_buffer.lookup_transform(
            'base_link', 'upper_Frameconnector1', rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=5.0))
        delta2_tf = self.tf_buffer.lookup_transform(
            'base_link', 'upper_Frameconnector2', rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=5.0))
        delta3_tf = self.tf_buffer.lookup_transform(
            'base_link', 'upper_Frameconnector3', rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=5.0))

        d455_tf = self.tf_buffer.lookup_transform(
            'base_link', 'D455f_color_optical_frame', rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=5.0))

        # 計算三個馬達的幾何中心
        delta_center_x = (delta1_tf.transform.translation.x +
                          delta2_tf.transform.translation.x +
                          delta3_tf.transform.translation.x) / 3
        delta_center_y = (delta1_tf.transform.translation.y +
                          delta2_tf.transform.translation.y +
                          delta3_tf.transform.translation.y) / 3
        delta_center_z = np.mean([
            delta1_tf.transform.translation.z,
            delta2_tf.transform.translation.z,
            delta3_tf.transform.translation.z
        ])

        # 計算D455相對於Delta中心的位置
        d455_relative_x = d455_tf.transform.translation.x - delta_center_x
        d455_relative_y = d455_tf.transform.translation.y - delta_center_y
        d455_relative_z = d455_tf.transform.translation.z - delta_center_z

        tvec = np.array([d455_relative_x * 1000,
                         d455_relative_y * 1000,
                         d455_relative_z * 1000], dtype=np.float32)
        return tvec

    def cords_callback(self, msg):
        self.tvec = self._get_camera2delta_tf()

        self.get_logger().info("接收去除目標點")
        groups_size = msg.layout.dim[0].size
        cord_size = msg.layout.dim[1].size

        target_crods = []
        for group in range(groups_size):
            pixel_cord = list(msg.data[group*cord_size:(group+1)*cord_size])
            target_crod = self.__transform_pixel2delta(pixel_cord)

            # 範圍檢測：800mm*600mm矩形範圍
            if self.__is_within_working_range(target_crod):
                target_crods.append(target_crod)
            else:
                self.get_logger().info(
                    f"目標點 [{target_crod[0]:.2f}, {target_crod[1]:.2f}] 超出工作範圍，已過濾")

        delta_cords = self.__trajectory_plan(target_crods)

        # self.get_logger().info(f"發佈轉換後去除目標點{delta                                              _cords}")
        self.__publish_target_cords(delta_cords)

    def __transform_pixel2delta(self, pixel_cord):
        """將像素座標系轉換到手臂座標系"""

        if self.current_depth is None:
            depth_mm = 650.0  # 預設深度
        else:
            depth_mm = self.current_depth[int(
                pixel_cord[1]), int(pixel_cord[0])]

        # 將像素座標轉換為相機座標系
        pixel_points = np.array(
            [[pixel_cord]], dtype=np.float32)  # 注意格式：[[x, y]]

        # 使用 undistortPoints 獲得正規化座標
        normalized_points = cv2.undistortPoints(pixel_points, self.K, self.D)

        # 將正規化座標轉換回相機座標系（以毫米為單位）
        x_normalized = normalized_points[0][0][0]
        y_normalized = normalized_points[0][0][1]

        # 轉換為相機座標系的 3D 點
        cam_cord = np.array([
            x_normalized * depth_mm,
            y_normalized * depth_mm,
            depth_mm  # 注意：這裡應該是正值
        ], dtype=np.float32)

        # 除錯：檢查中間結果
        self.get_logger().debug(
            f"Pixel: {pixel_cord}, Normalized: [{x_normalized:.6f}, {y_normalized:.6f}]")
        self.get_logger().debug(
            f"Camera coord: [{cam_cord[0]:.2f}, {cam_cord[1]:.2f}, {cam_cord[2]:.2f}]")

        # 剛體轉換
        R, _ = cv2.Rodrigues(self.rvec)
        # 注意：這裡是 R * cam + t，不是 R.T * (cam - t)
        # delta_cord = np.dot(R, cam_cord) + self.tvec
        delta_cord = np.dot(R.T, cam_cord - self.tvec)

        delta_cord[0] = delta_cord[0] + 15
        delta_cord[1] = delta_cord[1] - 15
        delta_cord[2] = -cam_cord[2]-30

        # self.get_logger().info(
        # f"Delta coord: [{delta_cord[0]:.2f}, {delta_cord[1]:.2f}, {delta_cord[2]:.2f}]")

        return delta_cord

    def depth_callback(self, msg):
        """深度影像回調函式"""
        self.current_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def __is_within_working_range(self, delta_coord):
        """
        檢查目標點是否在Delta機械手臂的工作範圍內
        工作範圍：以Delta中心為基準的800mm*600mm矩形
        """
        x, y = delta_coord[0], delta_coord[1]

        # 矩形範圍：x方向±400mm，y方向±300mm
        x_range = 400.0  # 800mm / 2
        y_range = 300.0  # 600mm / 2

        if abs(x) <= x_range and abs(y) <= y_range:
            return True
        else:
            return False

    def __trajectory_plan(self, target_crods):
        # 取得參數值
        self.get_logger().info(
            f"self.delta_working_level: {getattr(self, 'delta_working_level', '未設定')}")
        delta_working_level = self.delta_working_level
        planed_cord = []

        for target_crod in target_crods:
            # 確保 target_crod 是列表格式並且只取前兩個座標
            x, y = float(target_crod[0]), float(target_crod[1])
            if target_crod[2] < self.delta_working_level[2]:
                delta_working_level[1] = target_crod[2]
            else:
                delta_working_level[1] = self.delta_working_level[1]
            for z in delta_working_level:
                planing_cord = [x, y, float(z)]  # 明確轉換為 float
                self.get_logger().info(f"{planing_cord}")
                planed_cord.append(planing_cord)

        # 除錯：檢查結果
        self.get_logger().debug(f"planed_cord length: {len(planed_cord)}")
        if planed_cord:
            self.get_logger().debug(
                f"First few planned coordinates: {planed_cord[:3]}")

        return planed_cord

    def __publish_target_cords(self, targets: np.ndarray):
        """
        Publish the target coordinates to the plant_cord topic.
        """
        # 除錯：檢查輸入資料
        self.get_logger().debug(f"targets type: {type(targets)}")
        self.get_logger().debug(
            f"targets shape/length: {len(targets) if hasattr(targets, '__len__') else 'No length'}")
        if len(targets) > 0:
            self.get_logger().debug(
                f"First target: {targets[0]}, type: {type(targets[0])}")

        cord_array = Float32MultiArray()
        num_groups = len(targets)
        num_coords = len(targets[0]) if num_groups > 0 else 0

        # 設定 Layout 的維度
        cord_array.layout.dim = [
            MultiArrayDimension(label="group", size=num_groups,
                                stride=num_groups * num_coords),
            MultiArrayDimension(label="coordinate",
                                size=num_coords, stride=num_coords)
        ]
        cord_array.layout.data_offset = 0

        # 設定數據並發佈
        if isinstance(targets, np.ndarray):
            targets_list = targets.tolist()
        else:
            targets_list = targets
        cord_array.data = self.__flatten_2d_array(targets_list)
        self.cord_publisher.publish(cord_array)

    def __flatten_2d_array(self, array: list[list[float]]) -> list[float]:
        """
        Helper function to flatten a 2D list into a 1D list.

        :param array: A 2D list of floats
        :return: A flattened 1D list
        """
        return [float(item) for sublist in array for item in sublist]


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
