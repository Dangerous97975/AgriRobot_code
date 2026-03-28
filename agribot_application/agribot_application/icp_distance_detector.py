#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import math


class ICPDistanceDetector(Node):
    def __init__(self):
        super().__init__('icp_distance_detector')

        # ICP odom
        self.odom_sub = self.create_subscription(
            Odometry,
            '/icp_odom',
            self.odom_callback,
            10)

        self.emergency_publisher = self.create_publisher(
            Bool,
            'emergency_stop',
            10)

        self.start_position = None
        self.current_distance = 0.0
        self.target_distance = 1.0
        self.distance_reached = False

        self.get_logger().info('ICP Distance Detector started')
        self.get_logger().info(
            f'Target distance: {self.target_distance} meters')
        self.get_logger().info('µ¥«Ý²Ä¤@­Ó¦ì¸m§@¬°°_©lÂI...')

    def odom_callback(self, msg):
        current_position = msg.pose.pose.position

        if self.start_position is None:
            self.start_position = current_position
            self.get_logger().info(
                f'Start point: x={self.start_position.x:.3f}, y={self.start_position.y:.3f}, z={self.start_position.z:.3f}')
            return

        dx = current_position.x - self.start_position.x
        dy = current_position.y - self.start_position.y

        self.current_distance = math.sqrt(dx*dx + dy*dy)

        # ÀË¬d¬O§_¹F¨ì¥Ø¼Ð¶ZÂ÷
        if not self.distance_reached and self.current_distance >= self.target_distance:
            self.distance_reached = True
            self.get_logger().info(
                f'Current distance{self.current_distance:.3f}, Goal!')

            # µo¥¬ºò«æ°±¤î«H¸¹
            emergency_msg = Bool()
            emergency_msg.data = True
            self.emergency_publisher.publish(emergency_msg)

        # ¨C¤½¤Ø¿é¥X¤@¦¸¶i«×
        if int(self.current_distance) > int(self.current_distance - 0.1):
            self.get_logger().info(
                f'Current distance: {self.current_distance:.3f} m')


def main(args=None):
    rclpy.init(args=args)

    detector = ICPDistanceDetector()

    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass

    detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
