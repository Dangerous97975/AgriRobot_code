#!/usr/bin/python3
import rclpy
import threading
from rclpy.node import Node
from customize_interface.msg import JoyMotionCommand
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

joyMotionCommand = JoyMotionCommand()


def linear_mapping(value, in_min, in_max, out_min, out_max):
    return round((value-in_min) * (out_max-out_min) / (in_max-in_min) + out_min, 2)


class joyController(Node):
    def __init__(self):
        super().__init__("joy_controller")

        self.turning_mode = 0
        self.button1_flag = True
        self.vel_direction = True
        self.emergency_stop_active = False
        self.button_X_flag = True  # 用於按鍵防彈跳 (假設用 buttons[1] 作為緊急停止/解鎖)

        self.declare_parameter("Max_vel", 1.5)

        self.Max_vel = self.get_parameter("Max_vel").value

        self.joy_subscriber = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10)

        self.joy_publisher = self.create_publisher(
            JoyMotionCommand, "/joy_raw_command", 10)

        self.emergency_stop_publisher = self.create_publisher(
            Bool, "emergency_stop", 10)

    def joy_callback(self, joy_data):
        # * 轉彎模態切換 ； Xbox X鍵/PS5 方塊鍵
        if joy_data.buttons[0] == 1 and self.button1_flag == True:
            self.turning_mode = not (self.turning_mode)
            self.button1_flag = False
        elif joy_data.buttons[0] == 0:
            self.button1_flag = True

        joyMotionCommand.turning_mode = self.turning_mode
        linear_x = linear_mapping(joy_data.axes[4], 1, -1, 0, self.Max_vel)

        # * Ackerman
        if self.turning_mode == 0:

            # * 轉速與正反轉處理
            if joy_data.buttons[4] == 1 and linear_x == 0:
                self.vel_direction = False
            elif joy_data.buttons[5] == 1 and linear_x == 0:
                self.vel_direction = True

            if self.vel_direction == True:
                joyMotionCommand.linear_x = linear_x
            elif self.vel_direction == False:
                joyMotionCommand.linear_x = -linear_x

            # * 轉彎功能
            joyMotionCommand.center_rotate_angle = linear_mapping(
                joy_data.axes[0], -1, 1, -40, 40)

        elif self.turning_mode == 1:
            joyMotionCommand.linear_x = 0.0
            joyMotionCommand.center_rotate_angle = -linear_mapping(
                joy_data.axes[0], -1, 1, -40, 40)

        # * 緊急停止/解鎖切換 (Xbox B鍵/PS5 X鍵)
        if joy_data.buttons[1] == 1 and self.button_X_flag == True:
            self.emergency_stop_active = not self.emergency_stop_active
            self.button_X_flag = False
            if self.emergency_stop_active:
                self.get_logger().info("緊急停止啟動!")
            else:
                self.get_logger().info("緊急停止解除!")
        elif joy_data.buttons[1] == 0:
            self.button_X_flag = True

        # 如果緊急停止啟動，強制命令為0
        if self.emergency_stop_active:
            joyMotionCommand.linear_x = 0.0
            joyMotionCommand.center_rotate_angle = 0.0

        # 發布緊急停止狀態
        emergency_msg = Bool()
        emergency_msg.data = self.emergency_stop_active
        self.emergency_stop_publisher.publish(emergency_msg)

        self.joy_publisher.publish(joyMotionCommand)

    def destroy_node(self):
        joyMotionCommand.linear_x = 0.0
        joyMotionCommand.center_rotate_angle = 0.0
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = joyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
