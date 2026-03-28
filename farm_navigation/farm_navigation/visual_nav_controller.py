#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from customize_interface.msg import JoyMotionCommand
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Odometry
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import csv
import os
from std_msgs.msg import Bool
from datetime import datetime


joyMotionCommand = JoyMotionCommand()


class VisualNavigationController(Node):
    def __init__(self):
        super().__init__('visual_navigation_controller')

        self._declare_parameters()
        self._init_variables()
        self._init_fuzzy_controller()
        self._init_csv_recording()

        self.should_publish = True

        # TF2 設定
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.robot_frame = "base_footprint"

        # 訂閱者
        self.navigation_pose_sub = self.create_subscription(
            PoseStamped,
            'navigation_pose',
            self.navigation_pose_callback,
            10)

        self.odom_sub = self.create_subscription(
            Odometry,
            '/icp_odom',
            self.odom_callback,
            10)

        self.navigation_status_sub = self.create_subscription(
            Bool,
            '/vision_navigation_status',
            self.navigation_status_callback,
            10
        )

        # 發布者
        # self.motion_command_pub = self.create_publisher(
        #     JoyMotionCommand,
        #     "/nav_command",
        #     10)

        self.nav_command_suggestion_pub = self.create_publisher(
            JoyMotionCommand,
            "/visual_nav_suggestion",  # 改為建議命令
            10
        )

        self.correction_pose_pub = self.create_publisher(
            PoseStamped,
            '/navigation_correction_pose',
            10
        )

        # 控制迴圈定時器 (20Hz)
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info('Visual Navigation Controller initialized')
        self.get_logger().info(f'Target velocity: {self.target_velocity} m/s')
        self.get_logger().info(
            f'Max steering angle: {self.max_steering_angle} degrees')

    def _declare_parameters(self):
        """宣告ROS2參數"""
        # 車體參數
        self.declare_parameter('target_velocity', 0.5)      # 目標速度 (m/s)
        self.declare_parameter('max_steering_angle', 20.0)  # 最大轉向角度 (度)
        self.declare_parameter('navigation_enabled', True)   # 導航開關
        self.declare_parameter('max_lateral_error', 0.05)     # 最大橫向偏差 (米)

    def _init_variables(self):
        """初始化狀態變數"""
        self.target_velocity = self.get_parameter('target_velocity').value
        self.max_steering_angle = self.get_parameter(
            'max_steering_angle').value
        self.navigation_enabled = self.get_parameter(
            'navigation_enabled').value
        self.max_lateral_error = self.get_parameter(
            'max_lateral_error').value

        # CSV 記錄相關變數
        self.csv_file_path = None
        self.csv_writer = None
        self.csv_file = None
        self.recording_enabled = False

        # 狀態變數
        self.last_navigation_pose = None
        self.robot_pose = None

    def navigation_status_callback(self, msg):
        """接收視覺導航狀態"""
        self.set_navigation_enabled(msg.data)
        self.get_logger().info(f'收到視覺導航狀態變更: {"啟用" if msg.data else "停用"}')

    def _init_csv_recording(self):
        """初始化 CSV 記錄功能"""
        if not self.navigation_enabled:
            return

        try:
            # 建立檔案名稱（包含時間戳記）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"navigation_data_{timestamp}.csv"

            # 建立 logs 資料夾（如果不存在）
            log_dir = os.path.expanduser("~/navigation_logs")
            os.makedirs(log_dir, exist_ok=True)

            self.csv_file_path = os.path.join(log_dir, filename)

            # 開啟 CSV 檔案
            self.csv_file = open(self.csv_file_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            # 寫入標題列
            headers = [
                'timestamp',
                'icp_x', 'icp_y', 'icp_yaw_deg',
                'nav_x', 'nav_y', 'nav_yaw_deg',
                'angular_error_deg', 'lateral_error_m',
                'steering_angle_deg',
                'main_rule_name', 'main_rule_activation'
            ]
            self.csv_writer.writerow(headers)

            self.recording_enabled = True
            self.get_logger().info(
                f'CSV recording started: {self.csv_file_path}')

        except Exception as e:
            self.get_logger().error(
                f'Failed to initialize CSV recording: {str(e)}')
            self.recording_enabled = False

    def _record_to_csv(self, angular_error, lateral_error, steering_angle, fuzzy_debug_info):
        """記錄數據到 CSV 檔案"""
        if not self.recording_enabled or self.csv_writer is None:
            return

        try:
            # 獲取當前時間戳記
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # 座標系轉換 - 將 NAV 轉換到 icp_odom 座標系
            # ICP 已經在 icp_odom 座標系中，不需要轉換
            icp_pose = self.robot_pose if self.robot_pose else None
            nav_pose = self.last_navigation_pose.pose if self.last_navigation_pose else None
            nav_in_icp_odom = self._transform_to_icp_odom(
                nav_pose) if nav_pose else None

            if icp_pose and hasattr(icp_pose, 'position'):
                icp_x = icp_pose.position.x
                icp_y = icp_pose.position.y
                icp_yaw_deg = np.degrees(self._quaternion_to_yaw(
                    icp_pose.orientation))
            else:
                icp_x = icp_y = icp_yaw_deg = 0.0

            if nav_in_icp_odom and hasattr(nav_in_icp_odom, 'position'):
                nav_x = nav_in_icp_odom.position.x
                nav_y = nav_in_icp_odom.position.y
                nav_yaw_deg = np.degrees(self._quaternion_to_yaw(
                    nav_in_icp_odom.orientation))
            else:
                nav_x = nav_y = nav_yaw_deg = 0.0

            # 轉向角度正負顛倒（僅用於記錄）
            steering_angle_deg_inverted = -steering_angle

            # 獲取主要觸發規則資訊
            main_rule = fuzzy_debug_info.get(
                'main_rule') if fuzzy_debug_info else None
            main_rule_name = main_rule.get(
                'description', 'None') if main_rule else 'None'
            main_rule_activation = main_rule.get(
                'activation', 0.0) if main_rule else 0.0

            # 寫入數據
            row = [
                timestamp,
                icp_x, icp_y, icp_yaw_deg,
                nav_x, nav_y, nav_yaw_deg,
                angular_error, lateral_error, steering_angle_deg_inverted,
                main_rule_name, main_rule_activation
            ]
            self.csv_writer.writerow(row)

            # 立即寫入檔案（避免數據丟失）
            self.csv_file.flush()

        except Exception as e:
            self.get_logger().error(f'Failed to record data to CSV: {str(e)}')

    def _transform_to_icp_odom(self, pose):
        """將位姿轉換到 icp_odom 座標系"""
        if pose is None:
            return None

        try:
            # 從 base_footprint 轉換到 icp_odom
            transform = self.tf_buffer.lookup_transform(
                'icp_odom', 'base_footprint', rclpy.time.Time())

            # 手動進行座標轉換
            trans_x = transform.transform.translation.x
            trans_y = transform.transform.translation.y
            trans_z = transform.transform.translation.z

            # 建立轉換後的位姿
            transformed_pose = type(pose)()

            # 進行平移轉換
            transformed_pose.position.x = pose.position.x + trans_x
            transformed_pose.position.y = pose.position.y + trans_y
            transformed_pose.position.z = pose.position.z + trans_z
            transformed_pose.orientation = pose.orientation

            return transformed_pose

        except Exception as e:
            self.get_logger().warn(
                f'座標系轉換失敗 (base_footprint -> icp_odom): {str(e)}')
            return pose  # 轉換失敗時返回原始位姿

    def _close_csv_recording(self):
        """關閉 CSV 記錄"""
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            self.recording_enabled = False
            self.get_logger().info('CSV recording stopped')

    def _init_fuzzy_controller(self):
        """初始化模糊控制器"""
        self.angular_error = ctrl.Antecedent(
            np.arange(-15, 15, 0.1), 'angular_error')

        self.lateral_error = ctrl.Antecedent(
            np.arange(-0.2, 0.2, 0.01), 'lateral_error')

        self.steering_angle = ctrl.Consequent(
            np.arange(-self.max_steering_angle, self.max_steering_angle+1, 1), 'steering_angle')

        # 定義模糊集
        # Angular Error
        self.angular_error['NL'] = fuzz.trapmf(
            self.angular_error.universe, [-15, -15, -10, -5])
        self.angular_error['NS'] = fuzz.trimf(
            self.angular_error.universe, [-10, -5, 0])
        self.angular_error['ZO'] = fuzz.trimf(
            self.angular_error.universe, [-5, 0, 5])
        self.angular_error['PS'] = fuzz.trimf(
            self.angular_error.universe, [0, 5, 10])
        self.angular_error['PL'] = fuzz.trapmf(
            self.angular_error.universe, [5, 10, 15, 15])

        self.lateral_error['NL'] = fuzz.trapmf(
            self.lateral_error.universe, [-0.2, -0.2, -0.1, -0.05])
        self.lateral_error['NS'] = fuzz.trimf(
            self.lateral_error.universe, [-0.1, -0.05, 0])
        self.lateral_error['ZO'] = fuzz.trimf(
            self.lateral_error.universe, [-0.05, 0, 0.05])
        self.lateral_error['PS'] = fuzz.trimf(
            self.lateral_error.universe, [0, 0.05, 0.1])
        self.lateral_error['PL'] = fuzz.trapmf(
            self.lateral_error.universe, [0.05, 0.1, 0.2, 0.2])

        # Left左轉為正，Right右轉為負
        self.steering_angle['LR'] = fuzz.trapmf(
            self.steering_angle.universe, [-10, -10, -8, 5])  # 加入左端點
        self.steering_angle['SR'] = fuzz.trimf(
            self.steering_angle.universe, [-10, -5, -3])
        self.steering_angle['CN'] = fuzz.trimf(
            self.steering_angle.universe, [-5, 0, 5])
        self.steering_angle['SL'] = fuzz.trimf(
            self.steering_angle.universe, [3, 5, 10])
        self.steering_angle['LL'] = fuzz.trapmf(
            self.steering_angle.universe, [5, 8, 10, 10])     # 加入右端點

        # 定義模糊規則（5x5 = 25條規則）
        rules, self.rule_mapping = self._create_rules()

        # 建立控制系統
        self.steering_ctrl = ctrl.ControlSystem(rules)
        self.steering_simulation = ctrl.ControlSystemSimulation(
            self.steering_ctrl)

        self.get_logger().info('Fuzzy controller initialized with 25 rules')

    def _create_rules(self):
        """建立規則表"""
        rules = []
        rule_mapping = {}

        angular_labels = ['NL', 'NS', 'ZO', 'PS', 'PL']
        lateral_labels = ['NL', 'NS', 'ZO', 'PS', 'PL']

        # rule_matrix = [
        #     # lateral_error:  NL    NS    ZO    PS    PL
        #     ['PL', 'PL', 'PL', 'PS', 'NL'],   # angular_error NL
        #     ['PL', 'PS', 'PS', 'PS', 'NS'],   # angular_error NS
        #     ['PL', 'PS', 'ZO', 'NS', 'NL'],   # angular_error ZO
        #     ['PS', 'PS', 'NS', 'ZO', 'NL'],   # angular_error PS
        #     ['PL', 'NS', 'NL', 'NL', 'NL'],   # angular_error PL
        # ]
        rule_matrix = [
            # lateral_error:  NL    NS    ZO    PS    PL
            ['LL', 'LL', 'LL', 'SL', 'CN'],   # angular_error NL
            ['LL', 'SL', 'SL', 'SL', 'SR'],   # angular_error NS
            ['LL', 'SL', 'CN', 'SR', 'LR'],   # angular_error ZO
            ['SL', 'SL', 'SR', 'SR', 'LR'],   # angular_error PS
            ['CN', 'SR', 'LR', 'LR', 'LR'],   # angular_error PL
        ]

        rule_id = 0
        for i, ang_label in enumerate(angular_labels):
            for j, lat_label in enumerate(lateral_labels):
                output_label = rule_matrix[i][j]

                rule = ctrl.Rule(
                    self.angular_error[ang_label] & self.lateral_error[lat_label],
                    self.steering_angle[output_label]
                )
                rules.append(rule)

                # 建立規則對應表
                rule_mapping[rule_id] = {
                    'angular': ang_label,
                    'lateral': lat_label,
                    'output': output_label,
                    'description': f"IF 角度={ang_label} AND 橫向={lat_label} THEN 轉向={output_label}"
                }
                rule_id += 1

        return rules, rule_mapping

    def navigation_pose_callback(self, msg):
        """接收導航線姿態資訊"""
        self.last_navigation_pose = msg
        self.should_publish = True  # 重新啟動發佈
        self.get_logger().debug('Received navigation pose')

    def odom_callback(self, msg):
        """接收機器人里程計資訊"""
        self.robot_pose = msg.pose.pose

    def control_loop(self):
        """主要控制迴圈"""
        if not self.navigation_enabled:
            # 確保在導航關閉時不發布任何運動指令
            if self.should_publish:
                self._publish_stop_command()
                self.should_publish = False
            return

        if self.last_navigation_pose is None:
            # 如果沒有導航資訊，停止發佈指令
            if self.should_publish:
                self._publish_stop_command()
                self.should_publish = False
            return
        else:
            self.should_publish = True

        try:
            # 計算控制指令
            steering_angle, velocity = self._calculate_fuzzy_control_command()

            # 發布控制指令
            self._publish_motion_command(velocity, steering_angle)

        except Exception as e:
            self.get_logger().error(f'Control loop error: {str(e)}')
            self._publish_stop_command()

    def _calculate_fuzzy_control_command(self):
        """使用模糊控制器計算轉向角度和速度"""
        lateral_error = self._calculate_lateral_error()
        angular_error = self._calculate_angular_error()

        # 限制輸入值在定義範圍內
        angular_error = np.clip(angular_error, -15, 16)
        lateral_error = np.clip(lateral_error, -0.2, 0.21)

        # 設定模糊控制器輸入
        self.steering_simulation.input['angular_error'] = angular_error
        self.steering_simulation.input['lateral_error'] = lateral_error

        # 執行模糊推理
        self.steering_simulation.compute()

        # 獲取模糊控制器輸出
        steering_angle = self.steering_simulation.output['steering_angle']

        # 執行模糊推理調試分析
        fuzzy_debug_info = self.debug_fuzzy_inference(
            angular_error, lateral_error, verbose=False)

        # 根據誤差調整速度（保持原有的速度調整邏輯）
        lateral_error_abs = abs(lateral_error)
        if lateral_error_abs > self.max_lateral_error:
            velocity_factor = 0.3  # 大幅減速
        elif lateral_error_abs > self.max_lateral_error * 0.5:
            velocity_factor = 0.6  # 適度減速
        else:
            velocity_factor = 1.0  # 正常速度

        velocity = self.target_velocity * velocity_factor

        # 記錄到CSV
        self._record_to_csv(angular_error, lateral_error,
                            steering_angle, fuzzy_debug_info)

        # 發布矯正資訊
        self._publish_correction_pose(
            lateral_error, angular_error, steering_angle)

        # self.get_logger().info(
        #     f'Fuzzy Control - Angular error: {angular_error:.3f}deg, '
        #     f'Lateral error: {lateral_error:.3f}m, '
        #     f'Steering: {steering_angle:.1f}deg')

        return steering_angle, velocity

    def debug_fuzzy_inference(self, angular_error_val, lateral_error_val, verbose=True):
        """詳細的模糊推理調試函數"""

        # 計算各模糊集合的隸屬度
        angular_memberships = {}
        lateral_memberships = {}

        # 角度誤差的隸屬度
        for label in ['NL', 'NS', 'ZO', 'PS', 'PL']:
            membership = fuzz.interp_membership(
                self.angular_error.universe,
                self.angular_error[label].mf,
                angular_error_val
            )
            angular_memberships[label] = membership

        # 橫向誤差的隸屬度
        for label in ['NL', 'NS', 'ZO', 'PS', 'PL']:
            membership = fuzz.interp_membership(
                self.lateral_error.universe,
                self.lateral_error[label].mf,
                lateral_error_val
            )
            lateral_memberships[label] = membership

        if verbose:
            self.get_logger().info(
                f"[Fuzzy] 輸入值: 角度誤差={angular_error_val:+6.2f}°, 橫向誤差={lateral_error_val:+7.3f}m")

            # 顯示輸入模糊集合的隸屬度
            self.get_logger().info("[Fuzzy] 輸入模糊集合:")
            angular_str = "  角度: " + ", ".join([f"{label}:{angular_memberships[label]:.2f}"
                                                for label in ['NL', 'NS', 'ZO', 'PS', 'PL']])
            self.get_logger().info(angular_str)

            lateral_str = "  橫向: " + ", ".join([f"{label}:{lateral_memberships[label]:.2f}"
                                                for label in ['NL', 'NS', 'ZO', 'PS', 'PL']])
            self.get_logger().info(lateral_str)

        # 找出主要觸發的規則
        max_activation = 0
        main_rule = None
        active_rules = []

        rule_id = 0
        for i, ang_label in enumerate(['NL', 'NS', 'ZO', 'PS', 'PL']):
            for j, lat_label in enumerate(['NL', 'NS', 'ZO', 'PS', 'PL']):
                # 計算規則激活強度 (AND操作使用最小值)
                activation = min(
                    angular_memberships[ang_label], lateral_memberships[lat_label])

                if activation > 0.01:  # 只考慮有意義的激活
                    rule_info = self.rule_mapping[rule_id].copy()
                    rule_info['activation'] = activation
                    active_rules.append(rule_info)

                    if activation > max_activation:
                        max_activation = activation
                        main_rule = rule_info

                rule_id += 1

        if verbose and main_rule:
            print(f"[Fuzzy] 主要觸發規則: {main_rule['description']}")
            print(f"[Fuzzy] 激活強度: {main_rule['activation']:.2f}")

            if len(active_rules) > 1:
                print(f"[Fuzzy] 其他激活規則:")
                # 顯示前3個
                for rule in sorted(active_rules, key=lambda x: x['activation'], reverse=True)[1:4]:
                    if rule['activation'] > 0.05:  # 只顯示較強的激活
                        print(
                            f"        {rule['description']} (激活度: {rule['activation']:.2f})")

        return {
            'angular_memberships': angular_memberships,
            'lateral_memberships': lateral_memberships,
            'active_rules': active_rules,
            'main_rule': main_rule
        }

    def _calculate_lateral_error(self):
        """
        計算橫向偏差
        - 目標在車體左側為正
        - 目標在車體右側為負
        """
        lateral_error = self.last_navigation_pose.pose.position.y

        return lateral_error

    def _calculate_angular_error(self):
        """
        計算角度偏差
        - 目標在車體左側為正
        - 目標在車體右側為負
        """
        # 從導航線姿態中提取角度
        nav_orientation = self.last_navigation_pose.pose.orientation
        angular_error = self._quaternion_to_yaw(nav_orientation)

        # 正規化角度到 [-180, 180]
        # angular_error = self._normalize_angle(angular_error)

        return np.degrees(angular_error)  # 轉換為度

    def _quaternion_to_yaw(self, quaternion):
        """四元數轉換為偏航角"""
        import math
        siny_cosp = 2 * (quaternion.w * quaternion.z +
                         quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y +
                             quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle):
        """正規化角度到 [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _publish_motion_command(self, velocity, steering_angle):
        """發布運動控制指令"""
        cmd = JoyMotionCommand()
        cmd.linear_x = float(velocity)
        cmd.center_rotate_angle = float(steering_angle)
        cmd.turning_mode = 0  # Ackerman 模式

        if not self.should_publish:
            return

        # self.motion_command_pub.publish(cmd)
        self.nav_command_suggestion_pub.publish(cmd)

    def _publish_stop_command(self):
        """發布停止指令"""
        self.get_logger().info("Stop nav command")
        cmd = JoyMotionCommand()
        cmd.linear_x = 0.0
        cmd.center_rotate_angle = 0.0
        cmd.turning_mode = 0

        # self.motion_command_pub.publish(cmd)
        self.nav_command_suggestion_pub.publish(cmd)

    def _publish_correction_pose(self, lateral_error, angular_error, steering_angle):
        """發布矯正方向資訊"""
        correction_pose = PoseStamped()

        # 設定 header
        correction_pose.header.stamp = self.get_clock().now().to_msg()
        correction_pose.header.frame_id = self.robot_frame

        # 位置資訊：使用偏差作為位置指示
        correction_pose.pose.position.x = 0.0  # 車體前方參考點
        correction_pose.pose.position.y = lateral_error  # 橫向偏差
        correction_pose.pose.position.z = 0.0

        # 方向資訊：使用矯正角度
        # 將轉向角度轉換為四元數 (繞z軸旋轉)
        correction_angle_rad = np.radians(steering_angle)
        correction_pose.pose.orientation.x = 0.0
        correction_pose.pose.orientation.y = 0.0
        correction_pose.pose.orientation.z = np.sin(correction_angle_rad / 2.0)
        correction_pose.pose.orientation.w = np.cos(correction_angle_rad / 2.0)

        self.correction_pose_pub.publish(correction_pose)

        self.get_logger().debug(
            f'Published correction pose - Lateral: {lateral_error:.3f}m, '
            f'Steering: {steering_angle:.1f}deg')

    def set_navigation_enabled(self, enabled):
        """設定導航開關"""
        self.navigation_enabled = enabled
        if not enabled:
            # 立即發送停止指令
            self._publish_stop_command()

            # 關閉CSV記錄
            self._close_csv_recording()

            # 清除導航資料，避免殘留
            self.last_navigation_pose = None
            self.should_publish = False

            self.get_logger().info('Navigation disabled - 已清除所有導航資料並停止運動')
        else:
            # 重新初始化CSV記錄（如果需要）
            if not self.recording_enabled:
                self._init_csv_recording()
            self.get_logger().info('Navigation enabled')


def main(args=None):
    rclpy.init(args=args)

    node = VisualNavigationController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Navigation controller shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
