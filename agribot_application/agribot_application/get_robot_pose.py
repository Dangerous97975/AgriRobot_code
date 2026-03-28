import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from tf_transformations import euler_from_quaternion
import csv
import numpy as np
import matplotlib.pyplot as plt
import time
import signal
import sys


class TFListener(Node):
    def __init__(self):
        super().__init__('tf2_listener')
        self.tf_buffer = Buffer()
        self.listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1, self.get_transform)

        # 用於存儲位姿資料的清單
        self.timestamps = []
        self.x_positions = []
        self.y_positions = []
        self.z_positions = []
        self.roll_angles = []
        self.pitch_angles = []
        self.yaw_angles = []

        # 記錄起始時間
        self.start_time = time.time()

        # 設定信號處理器以便在Ctrl+C時保存數據
        signal.signal(signal.SIGINT, self.signal_handler)

    def get_transform(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                'icp_odom',
                'base_footprint',
                rclpy.time.Time(seconds=0),
                rclpy.time.Duration(seconds=1))
            transform = tf.transform
            rotation_euler = euler_from_quaternion([
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w
            ])

            # 記錄時間戳 (相對於開始時間)
            current_time = time.time() - self.start_time
            self.timestamps.append(current_time)

            # 記錄位置
            self.x_positions.append(transform.translation.x)
            self.y_positions.append(transform.translation.y)
            self.z_positions.append(transform.translation.z)

            # 記錄方向 (歐拉角)
            self.roll_angles.append(rotation_euler[0])
            self.pitch_angles.append(rotation_euler[1])
            self.yaw_angles.append(rotation_euler[2])

            self.get_logger().info(
                f"\n平移: {transform.translation}\n, 四元數: {transform.rotation}\n, 歐拉角: {rotation_euler}")
        except Exception as e:
            self.get_logger().warn(f"不能獲取座標轉換: {str(e)}")

    def save_to_csv(self, filename='pose_data.csv'):
        """將位姿資料儲存到CSV檔案"""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 寫入標題
            writer.writerow(['Time', 'X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw'])
            # 寫入數據
            for i in range(len(self.timestamps)):
                writer.writerow([
                    self.timestamps[i],
                    self.x_positions[i],
                    self.y_positions[i],
                    self.z_positions[i],
                    self.roll_angles[i],
                    self.pitch_angles[i],
                    self.yaw_angles[i]
                ])
        self.get_logger().info(f"位姿資料已儲存到 {filename}")

    def plot_data(self):
        """產生位姿資料的圖表"""
        # 建立一個2x3的子圖佈局
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        # 位置圖表
        axs[0, 0].plot(self.timestamps, self.x_positions, 'r-', label='X')
        axs[0, 0].set_title('X Position vs Time')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('X Position (m)')
        axs[0, 0].grid(True)

        axs[0, 1].plot(self.timestamps, self.y_positions, 'g-', label='Y')
        axs[0, 1].set_title('Y Position vs Time')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Y Position (m)')
        axs[0, 1].grid(True)

        axs[0, 2].plot(self.timestamps, self.z_positions, 'b-', label='Z')
        axs[0, 2].set_title('Z Position vs Time')
        axs[0, 2].set_xlabel('Time (s)')
        axs[0, 2].set_ylabel('Z Position (m)')
        axs[0, 2].grid(True)

        # 歐拉角圖表
        axs[1, 0].plot(self.timestamps, self.roll_angles, 'r-', label='Roll')
        axs[1, 0].set_title('Roll Angle vs Time')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Roll (rad)')
        axs[1, 0].grid(True)

        axs[1, 1].plot(self.timestamps, self.pitch_angles, 'g-', label='Pitch')
        axs[1, 1].set_title('Pitch Angle vs Time')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Pitch (rad)')
        axs[1, 1].grid(True)

        axs[1, 2].plot(self.timestamps, self.yaw_angles, 'b-', label='Yaw')
        axs[1, 2].set_title('Yaw Angle vs Time')
        axs[1, 2].set_xlabel('Time (s)')
        axs[1, 2].set_ylabel('Yaw (rad)')
        axs[1, 2].grid(True)

        # 2D軌跡圖（X-Y平面）
        plt.figure(figsize=(10, 8))
        plt.plot(self.x_positions, self.y_positions, 'k-', label='Trajectory')
        plt.title('Robot Trajectory (X-Y Plane)')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.grid(True)
        plt.axis('equal')  # 使X軸和Y軸比例相同

        plt.tight_layout()
        plt.show()

    def signal_handler(self, sig, frame):
        """處理Ctrl+C信號，保存數據並繪製圖表"""
        self.get_logger().info("接收到終止信號，正在保存數據並繪製圖表...")
        self.save_to_csv()
        self.plot_data()
        sys.exit(0)


def main():
    rclpy.init()
    node = TFListener()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 保存數據並繪製圖表
        node.save_to_csv()
        node.plot_data()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
