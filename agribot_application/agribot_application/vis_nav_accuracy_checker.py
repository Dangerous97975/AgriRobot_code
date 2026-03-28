#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import tf2_ros
from collections import deque
import threading
import math
import os
import csv
from datetime import datetime


class RealtimeTrajectoryPlotter(Node):
    def __init__(self):
        super().__init__('realtime_trajectory_plotter')

        # 初始化數據儲存
        self.max_points = 1000  # 最大儲存點數

        # ICP 軌跡數據
        self.icp_x_data = deque(maxlen=self.max_points)
        self.icp_y_data = deque(maxlen=self.max_points)
        self.icp_times = deque(maxlen=self.max_points)

        # NAV 軌跡數據 (轉換到 icp_odom)
        self.nav_x_data = deque(maxlen=self.max_points)
        self.nav_y_data = deque(maxlen=self.max_points)
        self.nav_times = deque(maxlen=self.max_points)

        # 偏差數據
        self.angular_errors = deque(maxlen=self.max_points)
        self.lateral_errors = deque(maxlen=self.max_points)
        self.travel_distances = deque(maxlen=self.max_points)  # 累積移動距離

        # 狀態變數
        self.current_icp_pose = None
        self.current_nav_pose = None
        self.navigation_enabled = False
        self.data_lock = threading.Lock()
        self.total_distance = 0.0  # 累積移動距離
        self.last_icp_position = None  # 上一個ICP位置
        self.shutdown_flag = False  # 關閉標誌

        # TF2 設定
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 訂閱者
        self.odom_sub = self.create_subscription(
            Odometry,
            '/icp_odom',
            self.odom_callback,
            10)

        self.navigation_pose_sub = self.create_subscription(
            PoseStamped,
            '/navigation_pose',
            self.navigation_pose_callback,
            10)

        self.navigation_status_sub = self.create_subscription(
            Bool,
            '/vision_navigation_status',
            self.navigation_status_callback,
            10)

        # 初始化繪圖
        self.init_plots()

        # 啟動動畫
        self.animation = FuncAnimation(self.fig, self.update_plots,
                                       interval=100, blit=False)
        self.animation_started = True

        self.get_logger().info('Realtime Trajectory Plotter initialized')

        # 設定圖表儲存路徑
        self.save_dir = os.path.expanduser("~/navigation_plots")
        os.makedirs(self.save_dir, exist_ok=True)

    def navigation_status_callback(self, msg):
        """接收視覺導航狀態"""
        if self.shutdown_flag:
            return

        # with self.data_lock:
        #     self.navigation_enabled = msg.data
        #     if not msg.data:
        #         # 清除 nav 數據
        #         self.nav_x_data.clear()
        #         self.nav_y_data.clear()
        #         self.nav_times.clear()
        #         # 重置移動距離
        #         self.total_distance = 0.0
        #         self.last_icp_position = None
        #         self.angular_errors.clear()
        #         self.lateral_errors.clear()
        #         self.travel_distances.clear()

        self.get_logger().info(
            f'Navigation status: {"Enabled" if msg.data else "Disabled"}')

    def odom_callback(self, msg):
        """接收 ICP 里程計數據"""
        if self.shutdown_flag:
            return

        with self.data_lock:
            current_position = (msg.pose.pose.position.x,
                                msg.pose.pose.position.y)

            # 計算移動距離
            if self.last_icp_position is not None:
                dx = current_position[0] - self.last_icp_position[0]
                dy = current_position[1] - self.last_icp_position[1]
                distance_increment = math.sqrt(dx*dx + dy*dy)
                self.total_distance += distance_increment

            self.last_icp_position = current_position
            self.current_icp_pose = msg.pose.pose

            # 記錄 ICP 軌跡
            current_time = self.get_clock().now().nanoseconds / 1e9
            self.icp_x_data.append(msg.pose.pose.position.x)
            self.icp_y_data.append(msg.pose.pose.position.y)
            self.icp_times.append(current_time)

    def navigation_pose_callback(self, msg):
        """接收導航位姿數據"""
        if self.shutdown_flag:
            return

        with self.data_lock:
            self.current_nav_pose = msg.pose

            # 轉換 nav pose 到 icp_odom 座標系
            nav_in_icp = self.transform_to_icp_odom(msg.pose)

            if nav_in_icp:
                current_time = self.get_clock().now().nanoseconds / 1e9
                self.nav_x_data.append(nav_in_icp.position.x)
                self.nav_y_data.append(nav_in_icp.position.y)
                self.nav_times.append(current_time)

                # 計算偏差
                if self.current_icp_pose:
                    self.calculate_errors()

    def transform_to_icp_odom(self, pose):
        """將位姿轉換到 icp_odom 座標系"""
        try:
            # 從 base_footprint 轉換到 icp_odom
            transform = self.tf_buffer.lookup_transform(
                'icp_odom', 'base_footprint', rclpy.time.Time())

            # 手動進行座標轉換
            trans_x = transform.transform.translation.x
            trans_y = transform.transform.translation.y

            # 建立轉換後的位姿
            from geometry_msgs.msg import Pose
            transformed_pose = Pose()
            transformed_pose.position.x = pose.position.x + trans_x
            transformed_pose.position.y = pose.position.y + trans_y
            transformed_pose.position.z = pose.position.z
            transformed_pose.orientation = pose.orientation

            return transformed_pose

        except Exception as e:
            self.get_logger().debug(f'Transform failed: {str(e)}')
            return pose

    def calculate_errors(self):
        """計算角度和橫向偏差"""
        if not self.current_icp_pose or not self.current_nav_pose:
            return

        # 計算橫向偏差 (lateral error)
        lateral_error = self.current_nav_pose.position.y

        # 計算角度偏差 (angular error)
        nav_yaw = self.quaternion_to_yaw(self.current_nav_pose.orientation)
        angular_error = math.degrees(nav_yaw)

        # 儲存偏差數據與對應的移動距離
        self.angular_errors.append(angular_error)
        self.lateral_errors.append(lateral_error)
        self.travel_distances.append(self.total_distance)

    def quaternion_to_yaw(self, quaternion):
        """四元數轉換為偏航角"""
        siny_cosp = 2 * (quaternion.w * quaternion.z +
                         quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y +
                             quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def init_plots(self):
        """初始化繪圖視窗"""
        # 設定中文字體
        plt.rcParams['font.size'] = 10

        # 建立子圖 (2x2 改為 2x2，但只使用3個)
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)
                   ) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-time Navigation Tracking',
                          fontsize=16, fontweight='bold')

        # 隱藏右下角的子圖
        self.ax4.set_visible(False)

        # 軌跡圖
        self.ax1.set_title('Trajectory (icp_odom frame)')
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')

        # 初始化軌跡線
        self.icp_line, = self.ax1.plot(
            [], [], 'b-', linewidth=2, label='ICP Trajectory', alpha=0.7)
        self.nav_line, = self.ax1.plot(
            [], [], 'r-', linewidth=2, label='NAV Trajectory', alpha=0.7)
        self.icp_point, = self.ax1.plot(
            [], [], 'bo', markersize=8, label='ICP Current')
        self.nav_point, = self.ax1.plot(
            [], [], 'ro', markersize=8, label='NAV Current')
        self.ax1.legend()

        # 角度偏差圖
        self.ax2.set_title('Angular Error vs Travel Distance')
        self.ax2.set_xlabel('Travel Distance (m)')
        self.ax2.set_ylabel('Angular Error (deg)')
        self.ax2.grid(True, alpha=0.3)
        self.angular_line, = self.ax2.plot([], [], 'g-', linewidth=2)
        self.ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # 橫向偏差圖
        self.ax3.set_title('Lateral Error vs Travel Distance')
        self.ax3.set_xlabel('Travel Distance (m)')
        self.ax3.set_ylabel('Lateral Error (m)')
        self.ax3.grid(True, alpha=0.3)
        self.lateral_line, = self.ax3.plot([], [], 'm-', linewidth=2)
        self.ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        plt.tight_layout()

    def update_plots(self, frame):
        """更新繪圖"""
        if self.shutdown_flag:
            return ()

        try:
            with self.data_lock:
                # 更新軌跡圖
                if len(self.icp_x_data) > 0:
                    self.icp_line.set_data(
                        list(self.icp_x_data), list(self.icp_y_data))
                    # 更新當前點
                    self.icp_point.set_data(
                        [self.icp_x_data[-1]], [self.icp_y_data[-1]])

                if len(self.nav_x_data) > 0:
                    self.nav_line.set_data(
                        list(self.nav_x_data), list(self.nav_y_data))
                    # 更新當前點
                    self.nav_point.set_data(
                        [self.nav_x_data[-1]], [self.nav_y_data[-1]])

                # 動態調整軌跡圖範圍
                all_x = list(self.icp_x_data) + list(self.nav_x_data)
                all_y = list(self.icp_y_data) + list(self.nav_y_data)

                if all_x and all_y:
                    margin = 0.5
                    self.ax1.set_xlim(min(all_x) - margin, max(all_x) + margin)
                    self.ax1.set_ylim(min(all_y) - margin, max(all_y) + margin)

                # 更新偏差圖表
                if len(self.travel_distances) > 0 and len(self.angular_errors) > 0:
                    # 角度偏差 vs 移動距離
                    self.angular_line.set_data(
                        list(self.travel_distances), list(self.angular_errors))
                    if self.angular_errors:
                        y_margin = max(abs(min(self.angular_errors)), abs(
                            max(self.angular_errors))) * 0.1
                        self.ax2.set_ylim(min(self.angular_errors) - y_margin,
                                          max(self.angular_errors) + y_margin)

                    # 橫向偏差 vs 移動距離
                    self.lateral_line.set_data(
                        list(self.travel_distances), list(self.lateral_errors))
                    if self.lateral_errors:
                        y_margin = max(abs(min(self.lateral_errors)), abs(
                            max(self.lateral_errors))) * 0.1
                        self.ax3.set_ylim(min(self.lateral_errors) - y_margin,
                                          max(self.lateral_errors) + y_margin)

                    # 更新距離軸
                    if self.travel_distances:
                        distance_margin = max(self.travel_distances) * 0.05
                        for ax in [self.ax2, self.ax3]:
                            ax.set_xlim(
                                0, max(self.travel_distances) + distance_margin)

                # 更新狀態顯示
                status_text = f"Navigation: {'ON' if self.navigation_enabled else 'OFF'}"
                if hasattr(self, 'status_text'):
                    self.status_text.remove()
                self.status_text = self.fig.text(0.02, 0.98, status_text,
                                                 transform=self.fig.transFigure,
                                                 fontsize=12, fontweight='bold',
                                                 color='green' if self.navigation_enabled else 'red')

            return (self.icp_line, self.nav_line, self.icp_point, self.nav_point,
                    self.angular_line, self.lateral_line)

        except Exception as e:
            if not self.shutdown_flag:
                self.get_logger().debug(f'Plot update error: {str(e)}')
            return ()

    def calculate_rmse(self):
        """計算RMSE"""
        results = {}

        if len(self.angular_errors) > 0:
            angular_array = np.array(self.angular_errors)
            results['angular_rmse'] = np.sqrt(np.mean(angular_array**2))
            results['angular_mean'] = np.mean(angular_array)
            results['angular_std'] = np.std(angular_array)
            results['angular_max'] = np.max(np.abs(angular_array))

        if len(self.lateral_errors) > 0:
            lateral_array = np.array(self.lateral_errors)
            results['lateral_rmse'] = np.sqrt(np.mean(lateral_array**2))
            results['lateral_mean'] = np.mean(lateral_array)
            results['lateral_std'] = np.std(lateral_array)
            results['lateral_max'] = np.max(np.abs(lateral_array))

        if len(self.travel_distances) > 0:
            results['total_distance'] = max(self.travel_distances)
            results['data_points'] = len(self.travel_distances)

        return results

    def save_plots_and_data(self):
        """儲存圖表和計算統計數據"""
        try:
            # 生成時間戳記
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 儲存CSV數據
            csv_filename = f"navigation_data_{timestamp}.csv"
            csv_path = os.path.join(self.save_dir, csv_filename)

            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # 寫入表頭
                writer.writerow(['Distance_m', 'Angular_Error_deg', 'Lateral_Error_m',
                                'ICP_X_m', 'ICP_Y_m', 'NAV_X_m', 'NAV_Y_m'])

                # 找出最長的數據序列長度
                max_length = max(len(self.travel_distances), len(
                    self.icp_x_data), len(self.nav_x_data))

                # 逐行寫入數據
                for i in range(max_length):
                    row = []

                    # Distance, Angular Error, Lateral Error
                    if i < len(self.travel_distances):
                        row.append(f"{self.travel_distances[i]:.6f}")
                    else:
                        row.append("")

                    if i < len(self.angular_errors):
                        row.append(f"{self.angular_errors[i]:.6f}")
                    else:
                        row.append("")

                    if i < len(self.lateral_errors):
                        row.append(f"{self.lateral_errors[i]:.6f}")
                    else:
                        row.append("")

                    # ICP trajectory data
                    if i < len(self.icp_x_data):
                        row.append(f"{self.icp_x_data[i]:.6f}")
                    else:
                        row.append("")

                    if i < len(self.icp_y_data):
                        row.append(f"{self.icp_y_data[i]:.6f}")
                    else:
                        row.append("")

                    # NAV trajectory data
                    if i < len(self.nav_x_data):
                        row.append(f"{self.nav_x_data[i]:.6f}")
                    else:
                        row.append("")

                    if i < len(self.nav_y_data):
                        row.append(f"{self.nav_y_data[i]:.6f}")
                    else:
                        row.append("")

                    writer.writerow(row)

            self.get_logger().info(f'CSV data saved: {csv_path}')

            # 儲存組合圖表
            plot_filename = f"navigation_plot_combined_{timestamp}.png"
            plot_path = os.path.join(self.save_dir, plot_filename)
            self.fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.get_logger().info(f'Combined plot saved: {plot_path}')

            # 儲存各個圖表為獨立檔案
            self.save_individual_plots(timestamp)

            # 計算統計數據
            stats = self.calculate_rmse()

            # 儲存統計數據到文字檔
            stats_filename = f"navigation_stats_{timestamp}.txt"
            stats_path = os.path.join(self.save_dir, stats_filename)

            with open(stats_path, 'w') as f:
                f.write("Navigation Performance Statistics\n")
                f.write("=" * 40 + "\n\n")
                f.write(
                    f"Data Collection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                if 'total_distance' in stats:
                    f.write(
                        f"Total Travel Distance: {stats['total_distance']:.3f} m\n")
                    f.write(f"Data Points: {stats['data_points']}\n\n")

                if 'angular_rmse' in stats:
                    f.write("Angular Error Statistics:\n")
                    f.write(f"  RMSE: {stats['angular_rmse']:.3f} degrees\n")
                    f.write(f"  Mean: {stats['angular_mean']:.3f} degrees\n")
                    f.write(f"  Std Dev: {stats['angular_std']:.3f} degrees\n")
                    f.write(
                        f"  Max Absolute: {stats['angular_max']:.3f} degrees\n\n")

                if 'lateral_rmse' in stats:
                    f.write("Lateral Error Statistics:\n")
                    f.write(f"  RMSE: {stats['lateral_rmse']:.3f} m\n")
                    f.write(f"  Mean: {stats['lateral_mean']:.3f} m\n")
                    f.write(f"  Std Dev: {stats['lateral_std']:.3f} m\n")
                    f.write(
                        f"  Max Absolute: {stats['lateral_max']:.3f} m\n\n")

                # 新增原始數據
                f.write("Raw Data:\n")
                f.write("-" * 20 + "\n")
                f.write("Distance(m), Angular_Error(deg), Lateral_Error(m)\n")
                for i in range(len(self.travel_distances)):
                    if i < len(self.angular_errors) and i < len(self.lateral_errors):
                        f.write(f"{self.travel_distances[i]:.3f}, "
                                f"{self.angular_errors[i]:.3f}, "
                                f"{self.lateral_errors[i]:.3f}\n")

            self.get_logger().info(f'Statistics saved: {stats_path}')

            # 在控制台顯示統計摘要
            print("\n" + "="*50)
            print("NAVIGATION PERFORMANCE SUMMARY")
            print("="*50)
            if 'total_distance' in stats:
                print(f"Total Distance: {stats['total_distance']:.3f} m")
                print(f"Data Points: {stats['data_points']}")

            if 'angular_rmse' in stats:
                print(f"\nAngular Error:")
                print(f"  RMSE: {stats['angular_rmse']:.3f}°")
                print(f"  Mean: {stats['angular_mean']:.3f}°")
                print(f"  Max: ±{stats['angular_max']:.3f}°")

            if 'lateral_rmse' in stats:
                print(f"\nLateral Error:")
                print(f"  RMSE: {stats['lateral_rmse']:.4f} m")
                print(f"  Mean: {stats['lateral_mean']:.4f} m")
                print(f"  Max: ±{stats['lateral_max']:.4f} m")

            print("="*50)

        except Exception as e:
            self.get_logger().error(f'Failed to save plots and data: {str(e)}')

    def save_individual_plots(self, timestamp):
        """儲存各個圖表為獨立檔案"""
        try:
            # 1. 軌跡圖
            fig1, ax1 = plt.subplots(figsize=(12, 6))

            if len(self.icp_x_data) > 0:
                ax1.plot(list(self.icp_x_data), list(self.icp_y_data),
                         'b-', linewidth=2, label='ICP Trajectory', alpha=0.7)
                ax1.plot([self.icp_x_data[-1]], [self.icp_y_data[-1]],
                         'bo', markersize=8, label='ICP Current')

            if len(self.nav_x_data) > 0:
                ax1.plot(list(self.nav_x_data), list(self.nav_y_data),
                         'r-', linewidth=2, label='NAV Trajectory', alpha=0.7)
                ax1.plot([self.nav_x_data[-1]], [self.nav_y_data[-1]],
                         'ro', markersize=8, label='NAV Current')

            ax1.set_title('Trajectory (icp_odom frame)',
                          fontsize=14, fontweight='bold')
            ax1.set_xlabel('X (m)', fontsize=12)
            ax1.set_ylabel('Y (m)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            # ax1.set_aspect('equal')
            ax1.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.05, 1))

            # 動態調整軌跡圖範圍
            all_x = list(self.icp_x_data) + list(self.nav_x_data)
            all_y = list(self.icp_y_data) + list(self.nav_y_data)
            if all_x and all_y:
                margin = 0.3
                ax1.set_xlim(min(all_x) - margin, max(all_x) + margin)
                ax1.set_ylim(min(all_y) - margin, max(all_y) + margin)

            trajectory_path = os.path.join(
                self.save_dir, f"trajectory_{timestamp}.png")
            fig1.savefig(trajectory_path, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            self.get_logger().info(f'Trajectory plot saved: {trajectory_path}')

            # 2. 角度偏差圖
            if len(self.travel_distances) > 0 and len(self.angular_errors) > 0:
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(list(self.travel_distances), list(self.angular_errors),
                         'g-', linewidth=2, label='Angular Error')
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax2.set_title('Angular Error vs Travel Distance',
                              fontsize=14, fontweight='bold')
                ax2.set_xlabel('Travel Distance (m)', fontsize=12)
                ax2.set_ylabel('Angular Error (deg)', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.legend(fontsize=10)

                # 設定適當的Y軸範圍
                if self.angular_errors:
                    y_margin = max(abs(min(self.angular_errors)),
                                   abs(max(self.angular_errors))) * 0.1
                    ax2.set_ylim(min(self.angular_errors) - y_margin,
                                 max(self.angular_errors) + y_margin)

                # 設定X軸範圍
                if self.travel_distances:
                    distance_margin = max(self.travel_distances) * 0.05
                    ax2.set_xlim(0, max(self.travel_distances) +
                                 distance_margin)

                angular_path = os.path.join(
                    self.save_dir, f"angular_error_{timestamp}.png")
                fig2.savefig(angular_path, dpi=300, bbox_inches='tight')
                plt.close(fig2)
                self.get_logger().info(
                    f'Angular error plot saved: {angular_path}')

            # 3. 橫向偏差圖
            if len(self.travel_distances) > 0 and len(self.lateral_errors) > 0:
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                ax3.plot(list(self.travel_distances), list(self.lateral_errors),
                         'm-', linewidth=2, label='Lateral Error')
                ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax3.set_title('Lateral Error vs Travel Distance',
                              fontsize=14, fontweight='bold')
                ax3.set_xlabel('Travel Distance (m)', fontsize=12)
                ax3.set_ylabel('Lateral Error (m)', fontsize=12)
                ax3.grid(True, alpha=0.3)
                ax3.legend(fontsize=10)

                # 設定適當的Y軸範圍
                if self.lateral_errors:
                    y_margin = max(abs(min(self.lateral_errors)),
                                   abs(max(self.lateral_errors))) * 0.1
                    ax3.set_ylim(min(self.lateral_errors) - y_margin,
                                 max(self.lateral_errors) + y_margin)

                # 設定X軸範圍
                if self.travel_distances:
                    distance_margin = max(self.travel_distances) * 0.05
                    ax3.set_xlim(0, max(self.travel_distances) +
                                 distance_margin)

                lateral_path = os.path.join(
                    self.save_dir, f"lateral_error_{timestamp}.png")
                fig3.savefig(lateral_path, dpi=300, bbox_inches='tight')
                plt.close(fig3)
                self.get_logger().info(
                    f'Lateral error plot saved: {lateral_path}')

        except Exception as e:
            self.get_logger().error(
                f'Failed to save individual plots: {str(e)}')

    def cleanup(self):
        """清理資源並儲存數據"""
        self.shutdown_flag = True
        self.get_logger().info('Saving plots and calculating statistics...')

        # 停止動畫
        try:
            if hasattr(self, 'animation') and self.animation is not None:
                if hasattr(self.animation, 'event_source') and self.animation.event_source is not None:
                    self.animation.event_source.stop()
                    self.get_logger().info('Animation stopped')
        except Exception as e:
            self.get_logger().debug(f'Animation stop error: {str(e)}')

        # 儲存數據和圖表
        try:
            self.save_plots_and_data()
        except Exception as e:
            self.get_logger().error(f'Save plots error: {str(e)}')

        self.get_logger().info('Cleanup completed')

    def show(self):
        """顯示繪圖視窗"""
        plt.show()


def main(args=None):
    rclpy.init(args=args)

    plotter = None
    ros_thread = None

    try:
        plotter = RealtimeTrajectoryPlotter()

        # 在另一個線程中運行 ROS2 spin
        import threading

        def ros_spin():
            try:
                rclpy.spin(plotter)
            except rclpy.executors.ExternalShutdownException:
                pass  # 正常關閉
            except Exception as e:
                if not plotter.shutdown_flag:
                    print(f"ROS thread error: {e}")

        ros_thread = threading.Thread(target=ros_spin)
        ros_thread.daemon = True
        ros_thread.start()

        # 在主線程中顯示繪圖
        plotter.show()

    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, shutting down...")
    except Exception as e:
        print(f'Unexpected error: {str(e)}')
    finally:
        # 確保在程式結束時儲存數據
        if plotter:
            try:
                plotter.cleanup()
            except Exception as e:
                print(f"Cleanup error: {e}")

            try:
                plotter.destroy_node()
            except Exception as e:
                print(f"Node destroy error: {e}")

        # 關閉 ROS2
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as e:
                print(f"ROS shutdown error: {e}")

        # 等待線程結束
        if ros_thread and ros_thread.is_alive():
            ros_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
