#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy.time
import rclpy.duration


class DeltaCenterTfPublisher(Node):
    def __init__(self):
        super().__init__('delta_center_tf_publisher')

        # 初始化 TF 相關組件
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # 創建定時器，定期發佈 Delta 中心座標系
        self.timer = self.create_timer(
            0.1, self.publish_delta_center_tf)  # 10Hz

        self.get_logger().info('Delta center TF publisher started')

    def publish_delta_center_tf(self):
        try:
            # 取得三個 Delta 連接器的 TF
            delta1_tf = self.tf_buffer.lookup_transform(
                'base_link', 'upper_Frameconnector1', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            delta2_tf = self.tf_buffer.lookup_transform(
                'base_link', 'upper_Frameconnector2', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            delta3_tf = self.tf_buffer.lookup_transform(
                'base_link', 'upper_Frameconnector3', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))

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
            ])-0.06  # 假設 Z 軸偏移 5cm

            # 創建並發佈 Delta 中心座標系
            delta_center_tf = TransformStamped()
            delta_center_tf.header.stamp = self.get_clock().now().to_msg()
            delta_center_tf.header.frame_id = 'base_link'
            delta_center_tf.child_frame_id = 'delta_center'

            delta_center_tf.transform.translation.x = delta_center_x
            delta_center_tf.transform.translation.y = delta_center_y
            delta_center_tf.transform.translation.z = delta_center_z

            # 以Z軸中心向右旋轉90度（-90度）
            delta_center_tf.transform.rotation.x = 0.0
            delta_center_tf.transform.rotation.y = 0.0
            delta_center_tf.transform.rotation.z = - \
                0.7071067811865476  # sin(-π/4)
            # cos(-π/4)
            delta_center_tf.transform.rotation.w = 0.7071067811865476

            self.tf_broadcaster.sendTransform(delta_center_tf)

        except Exception as e:
            self.get_logger().debug(f'無法取得 Delta 連接器的 TF: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = DeltaCenterTfPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
