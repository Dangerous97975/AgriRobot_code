#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class DeltaCenterFinder(Node):
    def __init__(self):
        super().__init__('delta_center_finder')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 等待TF樹建立
        self.create_timer(2.0, self.find_transforms)

    def find_transforms(self):
        try:
            # 獲取三個馬達的位置
            motor1_tf = self.tf_buffer.lookup_transform(
                'base_link', 'stepper_motor1', rclpy.time.Time())
            motor2_tf = self.tf_buffer.lookup_transform(
                'base_link', 'stepper_motor2', rclpy.time.Time())
            motor3_tf = self.tf_buffer.lookup_transform(
                'base_link', 'stepper_motor3', rclpy.time.Time())

            # 獲取下平台位置
            plate_tf = self.tf_buffer.lookup_transform(
                'base_link', 'plate_down', rclpy.time.Time())

            # 獲取D455位置
            d455_tf = self.tf_buffer.lookup_transform(
                'base_link', 'D455f_color_optical_frame', rclpy.time.Time())

            print("相對於base_link的位置:")
            print(
                f"馬達1: x={motor1_tf.transform.translation.x:.3f}, y={motor1_tf.transform.translation.y:.3f}, z={motor1_tf.transform.translation.z:.3f}")
            print(
                f"馬達2: x={motor2_tf.transform.translation.x:.3f}, y={motor2_tf.transform.translation.y:.3f}, z={motor2_tf.transform.translation.z:.3f}")
            print(
                f"馬達3: x={motor3_tf.transform.translation.x:.3f}, y={motor3_tf.transform.translation.y:.3f}, z={motor3_tf.transform.translation.z:.3f}")
            print(
                f"下平台: x={plate_tf.transform.translation.x:.3f}, y={plate_tf.transform.translation.y:.3f}, z={plate_tf.transform.translation.z:.3f}")
            print(
                f"D455: x={d455_tf.transform.translation.x:.3f}, y={d455_tf.transform.translation.y:.3f}, z={d455_tf.transform.translation.z:.3f}")

            # 計算三個馬達的幾何中心
            motor_center_x = (motor1_tf.transform.translation.x +
                              motor2_tf.transform.translation.x + motor3_tf.transform.translation.x) / 3
            motor_center_y = (motor1_tf.transform.translation.y +
                              motor2_tf.transform.translation.y + motor3_tf.transform.translation.y) / 3
            motor_center_z = (motor1_tf.transform.translation.z +
                              motor2_tf.transform.translation.z + motor3_tf.transform.translation.z) / 3

            print(
                f"\n三馬達幾何中心: x={motor_center_x:.3f}, y={motor_center_y:.3f}, z={motor_center_z:.3f}")

            # 計算D455相對於Delta中心的位置
            d455_relative_x = d455_tf.transform.translation.x - motor_center_x
            d455_relative_y = d455_tf.transform.translation.y - motor_center_y
            d455_relative_z = d455_tf.transform.translation.z - motor_center_z

            print(
                f"D455相對於Delta中心: x={d455_relative_x:.3f}, y={d455_relative_y:.3f}, z={d455_relative_z:.3f}")

            # 執行完畢後關閉節點
            self.destroy_node()
            rclpy.shutdown()

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f'錯誤: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = DeltaCenterFinder()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
