#!/usr/bin/env python3
"""
視覺導航性能監控工具
監控 vis_nav_line_detector_ridge 節點的處理時間性能
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import time
import threading


class PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('performance_monitor')

        self.vision_nav_status = False
        self.last_status_time = time.time()

        # 訂閱視覺導航狀態
        self.status_sub = self.create_subscription(
            Bool,
            '/vision_navigation_status',
            self.status_callback,
            10
        )

        # 創建定時器來定期輸出性能統計
        self.timer = self.create_timer(5.0, self.print_performance_stats)

        self.get_logger().info('性能監控器已啟動')
        self.get_logger().info('請確保 vis_nav_line_detector_ridge 節點正在運行')

    def status_callback(self, msg):
        """視覺導航狀態回調"""
        self.vision_nav_status = msg.data
        self.last_status_time = time.time()

        if msg.data:
            self.get_logger().info('視覺導航已啟用 - 開始性能監控')
        else:
            self.get_logger().info('視覺導航已停用')

    def print_performance_stats(self):
        """定期輸出性能統計"""
        current_time = time.time()

        # 檢查是否有狀態更新
        if current_time - self.last_status_time > 10.0:
            self.get_logger().warn(
                '警告: 超過10秒未收到視覺導航狀態更新，'
                '請檢查 vis_nav_line_detector_ridge 節點是否正在運行'
            )
            return

        if self.vision_nav_status:
            self.get_logger().info(
                '視覺導航運行中 - 請查看 vis_nav_line_detector_ridge 節點的日誌獲取詳細性能數據'
            )
        else:
            self.get_logger().info('視覺導航已停用 - 無性能數據')


def main(args=None):
    rclpy.init(args=args)

    monitor = PerformanceMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
