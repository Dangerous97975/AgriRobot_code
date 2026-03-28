#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32, Bool
from nav_msgs.msg import Odometry


class RotationTracker(Node):
    def __init__(self):
        super().__init__('rotation_tracker')

        # 狀態變數
        self.target_angle = 0.0
        self.start_angle = 0.0
        self.current_angle = 0.0
        self.rotation_active = False
        self.angle_tolerance = 5.0  # 度

        self._declare_parameters()

        # 訂閱器
        self.target_angle_sub = self.create_subscription(
            Float32, '/target_rotation_angle', self.target_angle_callback, 10)

        self.rotation_start_sub = self.create_subscription(
            Bool, '/start_rotation', self.rotation_start_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/icp_odom', self.odom_callback, 10)

        # 發布器
        self.rotation_completed_pub = self.create_publisher(
            Bool, '/rotation_completed', 10)

        # 定時檢查 (20Hz)
        self.check_timer = self.create_timer(
            0.05, self.check_rotation_progress)

        self.get_logger().info('Rotation Tracker initialized')

    def _declare_parameters(self):
        """宣告參數"""
        self.declare_parameter('angle_tolerance', 5.0)
        self.angle_tolerance = self.get_parameter('angle_tolerance').value

    def target_angle_callback(self, msg):
        """接收目標角度"""
        self.target_angle = msg.data

    def rotation_start_callback(self, msg):
        """開始角度追蹤"""
        if msg.data:
            self.rotation_active = True
            self.start_angle = self.current_angle
            self.get_logger().info(
                f'Starting rotation tracking: target={self.target_angle}°, start={np.degrees(self.start_angle):.1f}°')

    def odom_callback(self, msg):
        """更新當前角度"""
        orientation = msg.pose.pose.orientation
        self.current_angle = self._quaternion_to_yaw(orientation)

    def check_rotation_progress(self):
        """檢查旋轉進度"""
        if not self.rotation_active:
            return

        # 計算已旋轉角度
        angle_diff = self._normalize_angle(
            self.current_angle - self.start_angle)
        rotated_degrees = np.degrees(angle_diff)

        # 檢查是否達到目標
        error = abs(rotated_degrees - self.target_angle)

        if error <= self.angle_tolerance:
            # 旋轉完成
            self.rotation_active = False
            completed_msg = Bool()
            completed_msg.data = True
            self.rotation_completed_pub.publish(completed_msg)

            self.get_logger().info(
                f'Rotation completed! Target: {self.target_angle}°, Actual: {rotated_degrees:.1f}°')

        # 發布進度資訊
        self.get_logger().debug(
            f'Rotation progress: {rotated_degrees:.1f}°/{self.target_angle}°')

    def _quaternion_to_yaw(self, quaternion):
        """四元數轉偏航角"""
        import math
        siny_cosp = 2 * (quaternion.w * quaternion.z +
                         quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y +
                             quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle):
        """正規化角度到 [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = RotationTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
