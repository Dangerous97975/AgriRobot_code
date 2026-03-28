# farm_navigation/farm_navigation/pointcloud_capture_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
import numpy as np
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2
import os
from datetime import datetime


class PointCloudCaptureNode(Node):
    def __init__(self):
        super().__init__('pointcloud_capture')

        # 參數設定
        self.declare_parameter(
            'pointcloud_topic', '/agri_bot/D455f/depth/color/points')
        self.declare_parameter('trigger_topic', '/capture_trigger')
        self.declare_parameter('save_directory', './captured_pointclouds/')
        self.declare_parameter('capture_mode', 'topic')  # 'topic' 或 'service'

        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.save_directory = self.get_parameter('save_directory').value
        self.capture_mode = self.get_parameter('capture_mode').value

        # 建立儲存目錄
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            self.get_logger().info(f'建立儲存目錄: {self.save_directory}')

        # 初始化變數
        self.current_pointcloud = None
        self.capture_count = 0

        # 訂閱點雲話題
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            10
        )

        # 訂閱觸發話題
        self.trigger_sub = self.create_subscription(
            Bool,
            self.trigger_topic,
            self.trigger_callback,
            10
        )

        self.get_logger().info('點雲捕捉節點已啟動')
        self.get_logger().info(f'監聽點雲話題: {self.pointcloud_topic}')
        self.get_logger().info(f'監聽觸發話題: {self.trigger_topic}')

    def pointcloud_callback(self, msg):
        """更新當前點雲數據"""
        self.current_pointcloud = msg

    def trigger_callback(self, msg):
        """處理觸發訊息（備用）"""
        if msg.data:
            self.capture_pointcloud()

    def capture_pointcloud(self):
        """捕捉並儲存當前點雲"""
        if self.current_pointcloud is None:
            self.get_logger().warn('尚未接收到點雲數據')
            return

        try:
            # 先檢查點雲中有哪些欄位
            field_names = [
                field.name for field in self.current_pointcloud.fields]
            self.get_logger().info(f'點雲欄位: {field_names}')

            # 根據可用欄位讀取數據
            if all(field in field_names for field in ['r', 'g', 'b']):
                # 有顏色資訊
                pc_data = list(pc2.read_points(
                    self.current_pointcloud,
                    skip_nans=True,
                    field_names=("x", "y", "z", "r", "g", "b")
                ))
                has_colors = True
            else:
                # 只有 XYZ 座標
                pc_data = list(pc2.read_points(
                    self.current_pointcloud,
                    skip_nans=True,
                    field_names=("x", "y", "z")
                ))
                has_colors = False

            if len(pc_data) == 0:
                self.get_logger().warn('點雲數據為空')
                return

            # 準備點雲數據
            points = []
            colors = []

            for point in pc_data:
                points.append([point[0], point[1], point[2]])
                if has_colors and len(point) >= 6:
                    colors.append(
                        [point[3]/255.0, point[4]/255.0, point[5]/255.0])

            # 轉換為 Open3D 點雲
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))

            if has_colors and len(colors) == len(points):
                pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

            # 生成檔案名稱
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_pointcloud_{self.capture_count:04d}_{timestamp}.ply"
            filepath = os.path.join(self.save_directory, filename)

            # 儲存點雲
            o3d.io.write_point_cloud(filepath, pcd)
            self.capture_count += 1

            self.get_logger().info(f'成功捕捉點雲！儲存至: {filepath}')
            self.get_logger().info(f'點雲包含 {len(points)} 個點')

        except Exception as e:
            self.get_logger().error(f'捕捉點雲時發生錯誤: {e}')


def keyboard_listener(node):
    """獨立的鍵盤監聽執行緒"""
    import sys
    import termios
    import tty

    def get_key():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    print("\n========================================")
    print("點雲捕捉程式已啟動！")
    print("按鍵說明：")
    print("  [空白鍵] - 捕捉點雲")
    print("  [q] - 結束程式")
    print("========================================\n")

    while rclpy.ok():
        key = get_key()

        if key == ' ':  # 空白鍵
            node.capture_pointcloud()
        elif key == 'q' or key == 'Q':  # Q 鍵
            print("\n結束程式...")
            rclpy.shutdown()
            break
        elif key == '\x03':  # Ctrl+C
            rclpy.shutdown()
            break


def main(args=None):
    import threading

    rclpy.init(args=args)
    node = PointCloudCaptureNode()

    # 建立鍵盤監聽執行緒
    keyboard_thread = threading.Thread(target=keyboard_listener, args=(node,))
    keyboard_thread.daemon = True
    keyboard_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
