#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from customize_interface.msg import JoyMotionCommand
from std_msgs.msg import Bool
import time


class MotionArbitrator(Node):
    def __init__(self):
        super().__init__("motion_arbitrator")

        # 狀態變數
        self.emergency_stop = False
        self.joy_command = JoyMotionCommand()
        self.nav_command = JoyMotionCommand()
        self.last_joy_time = 0.0
        self.last_nav_time = 0.0

        # 參數設定
        self.declare_parameter("timeout_duration", 0.5)  # 超時時間（秒）
        self.timeout_duration = self.get_parameter("timeout_duration").value

        # 訂閱者
        self.joy_raw_subscriber = self.create_subscription(
            JoyMotionCommand,
            'joy_raw_command',
            self.joy_raw_callback,
            10)

        self.nav_subscriber = self.create_subscription(
            JoyMotionCommand,
            'nav_command',
            self.nav_callback,
            10)

        self.emergency_subscriber = self.create_subscription(
            Bool,
            'emergency_stop',
            self.emergency_callback,
            10)

        # 發布者
        self.final_command_publisher = self.create_publisher(
            JoyMotionCommand, "joy_command", 10)

        # 定時器 - 定期檢查並發布命令
        self.timer = self.create_timer(
            0.05, self.publish_final_command)  # 100Hz

        self.get_logger().info("運動仲裁節點已啟動")

    def joy_raw_callback(self, msg):
        """接收手把原始命令"""
        self.joy_command = msg
        self.last_joy_time = time.time()
        # self.get_logger().debug(f"收到手把命令: linear_x={msg.linear_x}, angle={msg.center_rotate_angle}")

    def nav_callback(self, msg):
        """接收導航命令"""
        self.nav_command = msg
        self.last_nav_time = time.time()
        # self.get_logger().debug(f"收到導航命令: linear_x={msg.linear_x}, angle={msg.center_rotate_angle}")

    def emergency_callback(self, msg):
        """接收緊急停止狀態"""
        if self.emergency_stop != msg.data:
            self.emergency_stop = msg.data
            if self.emergency_stop:
                self.get_logger().warn("緊急停止啟動 - 所有命令將被強制歸零")
            else:
                # self.emergency_stop = False
                self.get_logger().info("緊急停止解除")

    def is_joy_command_active(self):
        """檢查手把命令是否為非零且未超時"""
        current_time = time.time()

        # 檢查超時
        if current_time - self.last_joy_time > self.timeout_duration:
            return False

        # 檢查是否為非零命令或原地自轉模式
        if abs(self.joy_command.linear_x) > 0.01 or abs(self.joy_command.center_rotate_angle) > 0.1:
            return True

        # 檢查原地自轉模式 (turning_mode == 1)
        if self.joy_command.turning_mode == 1:
            return True

        return False

    def is_nav_command_valid(self):
        """檢查導航命令是否有效（未超時）"""
        current_time = time.time()
        return current_time - self.last_nav_time <= self.timeout_duration

    def publish_final_command(self):
        """發布最終運動命令"""
        final_command = JoyMotionCommand()

        # 如果緊急停止啟動，強制所有命令為零
        if self.emergency_stop:
            final_command.linear_x = 0.0
            final_command.center_rotate_angle = 0.0
            final_command.turning_mode = 0

        # 優先級邏輯：手把有非零命令時覆蓋導航命令
        elif self.is_joy_command_active():
            final_command.linear_x = self.joy_command.linear_x
            final_command.center_rotate_angle = -self.joy_command.center_rotate_angle
            final_command.turning_mode = self.joy_command.turning_mode
            # self.get_logger().debug("使用手把命令")

        # 否則使用導航命令（如果有效）
        elif self.is_nav_command_valid():
            final_command.linear_x = self.nav_command.linear_x
            final_command.center_rotate_angle = self.nav_command.center_rotate_angle
            final_command.turning_mode = self.nav_command.turning_mode
            # self.get_logger().debug("使用導航命令")

        # 都無效時發送零命令
        else:
            final_command.linear_x = 0.0
            final_command.center_rotate_angle = 0.0
            final_command.turning_mode = 0
            # self.get_logger().debug("發送零命令")

        self.final_command_publisher.publish(final_command)

    def destroy_node(self):
        """節點銷毀時發送停止命令"""
        stop_command = JoyMotionCommand()
        stop_command.linear_x = 0.0
        stop_command.center_rotate_angle = 0.0
        stop_command.turning_mode = 0
        self.final_command_publisher.publish(stop_command)
        self.get_logger().info("仲裁節點關閉，已發送停止命令")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotionArbitrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
