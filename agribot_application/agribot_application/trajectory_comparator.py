#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import math
from datetime import datetime
import os
import signal
import sys


class TrajectoryComparator(Node):
    def __init__(self):
        super().__init__('trajectory_comparator')

        # 儲存軌跡的佇列
        self.max_points = 50000
        self.amcl_trajectory = deque(maxlen=self.max_points)
        self.icp_trajectory = deque(maxlen=self.max_points)
        self.ekf_trajectory = deque(maxlen=self.max_points)

        # 儲存時間戳記
        self.amcl_timestamps = deque(maxlen=self.max_points)
        self.icp_timestamps = deque(maxlen=self.max_points)
        self.ekf_timestamps = deque(maxlen=self.max_points)

        # 儲存協方差
        self.amcl_covariances = deque(maxlen=self.max_points)
        self.icp_covariances = deque(maxlen=self.max_points)
        self.ekf_covariances = deque(maxlen=self.max_points)

        # 訂閱者
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            10
        )

        self.icp_sub = self.create_subscription(
            Odometry,
            '/icp_odom',
            self.icp_callback,
            10
        )

        # self.ekf_sub = self.create_subscription(
        #     Odometry,
        #     '/odom_filtered',
        #     self.ekf_callback,
        #     10
        # )

        # 統計資訊
        self.stats = {
            'amcl': {'count': 0, 'last_update': None},
            'icp': {'count': 0, 'last_update': None},
            'ekf': {'count': 0, 'last_update': None}
        }

        # 定期顯示狀態
        self.status_timer = self.create_timer(5.0, self.print_status)

        self.get_logger().info('Trajectory Comparator Node started')
        self.get_logger().info('Recording data... Press Ctrl+C to stop and generate plots')

        # 建立輸出目錄
        self.output_dir = f'trajectory_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.output_dir, exist_ok=True)

        # 註冊信號處理器
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """處理Ctrl+C信號"""
        self.get_logger().info('Stopping data collection and generating plots...')
        self.generate_final_plots()
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    def quaternion_to_yaw(self, q):
        """將四元數轉換為yaw角"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def extract_covariance_metrics(self, covariance):
        """提取協方差矩陣的關鍵指標"""
        x_var = covariance[0]
        y_var = covariance[7]
        yaw_var = covariance[35] if len(covariance) > 35 else covariance[-1]

        return {
            'x_std': math.sqrt(x_var),
            'y_std': math.sqrt(y_var),
            'yaw_std': math.sqrt(yaw_var),
            'total_uncertainty': math.sqrt(x_var + y_var)
        }

    def rotate_180(self, pose):
        """將位姿旋轉180度"""
        import math
        from tf_transformations import euler_from_quaternion, quaternion_from_euler

        # 取得當前姿態
        q = [pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w]
        roll, pitch, yaw = euler_from_quaternion(q)

        # 旋轉180度
        yaw += math.pi

        # 轉回四元數
        q_new = quaternion_from_euler(roll, pitch, yaw)
        pose.orientation.x = q_new[0]
        pose.orientation.y = q_new[1]
        pose.orientation.z = q_new[2]
        pose.orientation.w = q_new[3]

        # 位置也需要反向
        pose.position.x = -pose.position.x
        pose.position.y = -pose.position.y

        return pose

    def amcl_callback(self, msg):
        """處理AMCL資料"""
        pose = msg.pose.pose
        x = pose.position.x
        y = pose.position.y
        yaw = self.quaternion_to_yaw(pose.orientation)

        self.amcl_trajectory.append((x, y, yaw))
        self.amcl_timestamps.append(
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.amcl_covariances.append(
            self.extract_covariance_metrics(msg.pose.covariance))

        self.stats['amcl']['count'] += 1
        self.stats['amcl']['last_update'] = self.get_clock().now()

    def icp_callback(self, msg):
        """處理ICP odometry資料"""
        pose = msg.pose.pose

        # 旋轉180度
        # pose = self.rotate_180(pose)

        x = pose.position.x
        y = pose.position.y
        yaw = self.quaternion_to_yaw(pose.orientation)

        self.icp_trajectory.append((x, y, yaw))
        self.icp_timestamps.append(
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.icp_covariances.append(
            self.extract_covariance_metrics(msg.pose.covariance))

        self.stats['icp']['count'] += 1
        self.stats['icp']['last_update'] = self.get_clock().now()

    # def ekf_callback(self, msg):
    #     """處理EKF融合後的資料"""
    #     pose = msg.pose.pose
    #     x = pose.position.x
    #     y = pose.position.y
    #     yaw = self.quaternion_to_yaw(pose.orientation)

    #     self.ekf_trajectory.append((x, y, yaw))
    #     self.ekf_timestamps.append(
    #         msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
    #     self.ekf_covariances.append(
    #         self.extract_covariance_metrics(msg.pose.covariance))

    #     self.stats['ekf']['count'] += 1
    #     self.stats['ekf']['last_update'] = self.get_clock().now()

    def print_status(self):
        """定期印出狀態"""
        self.get_logger().info(f"Data collected - AMCL: {self.stats['amcl']['count']}, "
                               f"ICP: {self.stats['icp']['count']}, "
                               f"EKF: {self.stats['ekf']['count']}")

    def calculate_path_smoothness(self, trajectory):
        """計算路徑平滑度（基於加速度）"""
        if len(trajectory) < 3:
            return 0

        accelerations = []
        for i in range(2, len(trajectory)):
            x0, y0, _ = trajectory[i-2]
            x1, y1, _ = trajectory[i-1]
            x2, y2, _ = trajectory[i]

            # 計算二階差分（加速度的近似）
            ax = x2 - 2*x1 + x0
            ay = y2 - 2*y1 + y0

            acc = math.sqrt(ax**2 + ay**2)
            accelerations.append(acc)

        return np.mean(accelerations) if accelerations else 0

    def calculate_trajectory_length(self, trajectory):
        """計算軌跡總長度"""
        if len(trajectory) < 2:
            return 0

        total_length = 0
        for i in range(1, len(trajectory)):
            x0, y0, _ = trajectory[i-1]
            x1, y1, _ = trajectory[i]
            dist = math.sqrt((x1-x0)**2 + (y1-y0)**2)
            total_length += dist

        return total_length

    def generate_final_plots(self):
        """生成最終的分析圖表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Trajectory Comparison: AMCL vs ICP vs EKF', fontsize=16)

        # 1. 軌跡比較圖
        ax1 = axes[0, 0]
        if len(self.amcl_trajectory) > 0:
            amcl_data = np.array(self.amcl_trajectory)
            ax1.plot(amcl_data[:, 0], amcl_data[:, 1], 'r-',
                     label='AMCL', alpha=0.7, linewidth=2)
            ax1.scatter(amcl_data[0, 0], amcl_data[0, 1],
                        c='red', s=100, marker='o', label='AMCL Start')
            ax1.scatter(amcl_data[-1, 0], amcl_data[-1, 1],
                        c='red', s=100, marker='x', label='AMCL End')

        if len(self.icp_trajectory) > 0:
            icp_data = np.array(self.icp_trajectory)
            ax1.plot(icp_data[:, 0], icp_data[:, 1], 'g-',
                     label='ICP', alpha=0.7, linewidth=2)

        if len(self.ekf_trajectory) > 0:
            ekf_data = np.array(self.ekf_trajectory)
            ax1.plot(ekf_data[:, 0], ekf_data[:, 1], 'b-',
                     label='EKF', alpha=0.7, linewidth=2)

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Trajectory Comparison')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')

        # 2. 不確定性比較
        ax2 = axes[0, 1]
        if len(self.amcl_covariances) > 0:
            amcl_unc = [cov['total_uncertainty']
                        for cov in self.amcl_covariances]
            amcl_time = np.array(self.amcl_timestamps) - \
                self.amcl_timestamps[0]
            ax2.plot(amcl_time, amcl_unc, 'r-', label='AMCL', linewidth=2)

        if len(self.icp_covariances) > 0:
            icp_unc = [cov['total_uncertainty']
                       for cov in self.icp_covariances]
            icp_time = np.array(self.icp_timestamps) - self.icp_timestamps[0]
            ax2.plot(icp_time, icp_unc, 'g-', label='ICP', linewidth=2)

        if len(self.ekf_covariances) > 0:
            ekf_unc = [cov['total_uncertainty']
                       for cov in self.ekf_covariances]
            ekf_time = np.array(self.ekf_timestamps) - self.ekf_timestamps[0]
            ax2.plot(ekf_time, ekf_unc, 'b-', label='EKF', linewidth=2)

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Total Uncertainty (m)')
        ax2.set_title('Localization Uncertainty Over Time')
        ax2.legend()
        ax2.grid(True)

        # 3. 更新頻率比較
        ax3 = axes[0, 2]
        methods = ['AMCL', 'ICP', 'EKF']
        counts = [self.stats['amcl']['count'],
                  self.stats['icp']['count'],
                  self.stats['ekf']['count']]

        bars = ax3.bar(methods, counts, color=['red', 'green', 'blue'])
        ax3.set_ylabel('Update Count')
        ax3.set_title('Update Frequency Comparison')

        # 在條形圖上添加數值
        for i, (method, count) in enumerate(zip(methods, counts)):
            ax3.text(i, count + max(counts)*0.01, str(count),
                     ha='center', va='bottom', fontweight='bold')

        # 4. 路徑平滑度比較
        ax4 = axes[1, 0]
        smoothness_data = []
        smoothness_labels = []

        if len(self.amcl_trajectory) > 2:
            smoothness_data.append(
                self.calculate_path_smoothness(self.amcl_trajectory))
            smoothness_labels.append('AMCL')

        if len(self.icp_trajectory) > 2:
            smoothness_data.append(
                self.calculate_path_smoothness(self.icp_trajectory))
            smoothness_labels.append('ICP')

        if len(self.ekf_trajectory) > 2:
            smoothness_data.append(
                self.calculate_path_smoothness(self.ekf_trajectory))
            smoothness_labels.append('EKF')

        if smoothness_data:
            bars = ax4.bar(smoothness_labels, smoothness_data,
                           color=['red', 'green', 'blue'])
            ax4.set_ylabel('Path Smoothness (lower is better)')
            ax4.set_title('Trajectory Smoothness Comparison')

            # 添加數值標籤
            for i, (label, value) in enumerate(zip(smoothness_labels, smoothness_data)):
                ax4.text(i, value + max(smoothness_data)*0.01, f'{value:.6f}',
                         ha='center', va='bottom', fontweight='bold')

        # 5. 軌跡長度比較
        ax5 = axes[1, 1]
        length_data = []
        length_labels = []

        if len(self.amcl_trajectory) > 1:
            length_data.append(
                self.calculate_trajectory_length(self.amcl_trajectory))
            length_labels.append('AMCL')

        if len(self.icp_trajectory) > 1:
            length_data.append(
                self.calculate_trajectory_length(self.icp_trajectory))
            length_labels.append('ICP')

        if len(self.ekf_trajectory) > 1:
            length_data.append(
                self.calculate_trajectory_length(self.ekf_trajectory))
            length_labels.append('EKF')

        if length_data:
            bars = ax5.bar(length_labels, length_data,
                           color=['red', 'green', 'blue'])
            ax5.set_ylabel('Total Path Length (m)')
            ax5.set_title('Trajectory Length Comparison')

            # 添加數值標籤
            for i, (label, value) in enumerate(zip(length_labels, length_data)):
                ax5.text(i, value + max(length_data)*0.01, f'{value:.2f}m',
                         ha='center', va='bottom', fontweight='bold')

        # 6. X-Y位置散佈圖（顯示分散程度）
        ax6 = axes[1, 2]
        if len(self.amcl_trajectory) > 0:
            amcl_data = np.array(self.amcl_trajectory)
            ax6.scatter(amcl_data[:, 0], amcl_data[:, 1],
                        c='red', alpha=0.3, s=20, label='AMCL')

        if len(self.icp_trajectory) > 0:
            icp_data = np.array(self.icp_trajectory)
            ax6.scatter(icp_data[:, 0], icp_data[:, 1],
                        c='green', alpha=0.3, s=20, label='ICP')

        if len(self.ekf_trajectory) > 0:
            ekf_data = np.array(self.ekf_trajectory)
            ax6.scatter(ekf_data[:, 0], ekf_data[:, 1],
                        c='blue', alpha=0.3, s=20, label='EKF')

        ax6.set_xlabel('X (m)')
        ax6.set_ylabel('Y (m)')
        ax6.set_title('Position Scatter Plot')
        ax6.legend()
        ax6.grid(True)
        ax6.axis('equal')

        plt.tight_layout()

        # 儲存圖表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.output_dir}/final_comparison_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        self.get_logger().info(
            f'Plots saved to {self.output_dir}/final_comparison_{timestamp}.png')

        # 儲存原始資料
        np.save(f'{self.output_dir}/amcl_trajectory.npy',
                np.array(self.amcl_trajectory))
        np.save(f'{self.output_dir}/icp_trajectory.npy',
                np.array(self.icp_trajectory))
        np.save(f'{self.output_dir}/ekf_trajectory.npy',
                np.array(self.ekf_trajectory))

        # 儲存詳細統計資訊
        self.save_statistics()

        # 顯示圖表
        plt.show()

    def save_statistics(self):
        """儲存詳細統計資訊"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(f'{self.output_dir}/detailed_statistics_{timestamp}.txt', 'w') as f:
            f.write("Trajectory Comparison Detailed Statistics\n")
            f.write("========================================\n\n")

            # 基本統計
            f.write("1. Update Statistics:\n")
            f.write(f"   AMCL Updates: {self.stats['amcl']['count']}\n")
            f.write(f"   ICP Updates: {self.stats['icp']['count']}\n")
            f.write(f"   EKF Updates: {self.stats['ekf']['count']}\n\n")

            # 時間統計
            f.write("2. Time Statistics:\n")
            if len(self.amcl_timestamps) > 1:
                amcl_duration = self.amcl_timestamps[-1] - \
                    self.amcl_timestamps[0]
                amcl_rate = self.stats['amcl']['count'] / amcl_duration
                f.write(
                    f"   AMCL Duration: {amcl_duration:.2f}s, Rate: {amcl_rate:.2f}Hz\n")

            if len(self.icp_timestamps) > 1:
                icp_duration = self.icp_timestamps[-1] - self.icp_timestamps[0]
                icp_rate = self.stats['icp']['count'] / icp_duration
                f.write(
                    f"   ICP Duration: {icp_duration:.2f}s, Rate: {icp_rate:.2f}Hz\n")

            if len(self.ekf_timestamps) > 1:
                ekf_duration = self.ekf_timestamps[-1] - self.ekf_timestamps[0]
                ekf_rate = self.stats['ekf']['count'] / ekf_duration
                f.write(
                    f"   EKF Duration: {ekf_duration:.2f}s, Rate: {ekf_rate:.2f}Hz\n\n")

            # 路徑統計
            f.write("3. Path Statistics:\n")
            if len(self.amcl_trajectory) > 1:
                f.write(
                    f"   AMCL Path Length: {self.calculate_trajectory_length(self.amcl_trajectory):.2f}m\n")
                f.write(
                    f"   AMCL Smoothness: {self.calculate_path_smoothness(self.amcl_trajectory):.6f}\n")

            if len(self.icp_trajectory) > 1:
                f.write(
                    f"   ICP Path Length: {self.calculate_trajectory_length(self.icp_trajectory):.2f}m\n")
                f.write(
                    f"   ICP Smoothness: {self.calculate_path_smoothness(self.icp_trajectory):.6f}\n")

            if len(self.ekf_trajectory) > 1:
                f.write(
                    f"   EKF Path Length: {self.calculate_trajectory_length(self.ekf_trajectory):.2f}m\n")
                f.write(
                    f"   EKF Smoothness: {self.calculate_path_smoothness(self.ekf_trajectory):.6f}\n\n")

            # 不確定性統計
            f.write("4. Uncertainty Statistics:\n")
            if len(self.amcl_covariances) > 0:
                amcl_uncertainties = [cov['total_uncertainty']
                                      for cov in self.amcl_covariances]
                f.write(
                    f"   AMCL Mean Uncertainty: {np.mean(amcl_uncertainties):.6f}m\n")
                f.write(
                    f"   AMCL Max Uncertainty: {np.max(amcl_uncertainties):.6f}m\n")
                f.write(
                    f"   AMCL Min Uncertainty: {np.min(amcl_uncertainties):.6f}m\n")

            if len(self.icp_covariances) > 0:
                icp_uncertainties = [cov['total_uncertainty']
                                     for cov in self.icp_covariances]
                f.write(
                    f"   ICP Mean Uncertainty: {np.mean(icp_uncertainties):.6f}m\n")
                f.write(
                    f"   ICP Max Uncertainty: {np.max(icp_uncertainties):.6f}m\n")
                f.write(
                    f"   ICP Min Uncertainty: {np.min(icp_uncertainties):.6f}m\n")

            if len(self.ekf_covariances) > 0:
                ekf_uncertainties = [cov['total_uncertainty']
                                     for cov in self.ekf_covariances]
                f.write(
                    f"   EKF Mean Uncertainty: {np.mean(ekf_uncertainties):.6f}m\n")
                f.write(
                    f"   EKF Max Uncertainty: {np.max(ekf_uncertainties):.6f}m\n")
                f.write(
                    f"   EKF Min Uncertainty: {np.min(ekf_uncertainties):.6f}m\n")

        self.get_logger().info(
            f'Statistics saved to {self.output_dir}/detailed_statistics_{timestamp}.txt')


def main(args=None):
    rclpy.init(args=args)
    comparator = TrajectoryComparator()

    try:
        rclpy.spin(comparator)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')


if __name__ == '__main__':
    main()
