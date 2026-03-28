#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from enum import Enum
from std_msgs.msg import String, Bool, Float32
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from customize_interface.msg import JoyMotionCommand
import math


class NavigationState(Enum):
    MANUAL = "manual"
    ROW_IN_APPROACH = "row_in_approach"
    ROW_FOLLOWING = "row_following"
    WORKING_OPERATION = "working_operation"
    MOVING_TO_NEXT_WORK_POINT = "moving_to_next_work_point"
    ROW_END_APPROACH = "row_end_approach"
    TURNING_PHASE1 = "turning_phase1"
    TURNING_PHASE2 = "turning_phase2"
    TURNING_PHASE3 = "turning_phase3"
    ERROR_RECOVERY = "error_recovery"
    EMERGENCY_STOP = "emergency_stop"  # 新增異常停止狀態


class NavFSMController(Node):
    def __init__(self):
        super().__init__('nav_fsm_controller')

        # 狀態管理
        self.current_state = NavigationState.MANUAL
        self.previous_state = NavigationState.MANUAL

        # 感測器狀態
        self.furrow_state = "unknown"
        self.current_pose = None
        self._vision_nav_state = None

        # 作業追蹤
        self.current_row = 0
        self.total_rows = 10  # 從參數讀取
        self.row_direction = 1  # 1: 正向, -1: 反向
        self.vision_navigation_enabled = False

        # 作業操作相關
        self.work_point_start_position = None
        self.is_first_work_in_row = True  # 是否為該行的第一個作業點

        # 角度追蹤相關
        self.target_rotation_angle = 0.0
        self.rotation_completed = False
        self.current_yaw = 0.0

        # 錯誤追蹤
        self.error_count = 0
        self.max_error_count = 3
        self.last_error_time = None
        self.error_timeout = 10.0  # 10秒內超過3次錯誤進入異常停止

        self._declare_parameters()
        self._init_variables()
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_services()

        # 狀態機定時器 (10Hz)
        self.state_timer = self.create_timer(0.1, self.state_machine_loop)

        self.get_logger().info('導航狀態機控制器已啟動')

    def _declare_parameters(self):
        """宣告ROS2參數"""
        # 基本導航參數
        self.declare_parameter('total_rows', 10)
        self.declare_parameter('row_spacing', 1.1)
        self.declare_parameter('approach_speed', 0.35)
        self.declare_parameter('turn_speed', 0.3)
        self.declare_parameter('out_furrow_distance', 0.35)  # 離開土畦的距離

        # 擴充功能參數（暫時不使用）
        self.declare_parameter('enable_working_operations', False)
        self.declare_parameter('working_interval_distance', 0.5)  # 每0.5米執行一次作業
        # 角度控制參數
        self.declare_parameter('rotation_tolerance', 5.0)  # 角度容忍度（度）
        self.declare_parameter('rotation_timeout', 10.0)   # 旋轉超時時間（秒）

        # 錯誤處理參數
        self.declare_parameter('max_error_count', 3)
        self.declare_parameter('error_timeout', 10.0)
        self.declare_parameter('state_timeout', 30.0)  # 狀態超時時間

        # 速度控制參數
        self.declare_parameter('min_speed', 0.05)  # 最小速度
        self.declare_parameter('deceleration_distance', 0.3)  # 減速開始距離
        self.declare_parameter('speed_smoothing_factor', 0.8)  # 速度平滑係數

    def _init_variables(self):
        """初始化變數"""
        self.approach_speed = self.get_parameter('approach_speed').value
        self.total_rows = self.get_parameter('total_rows').value
        self.turn_speed = self.get_parameter('turn_speed').value
        self.row_spacing = self.get_parameter('row_spacing').value
        self.out_furrow_distance = self.get_parameter(
            'out_furrow_distance').value
        self.working_interval_distance = self.get_parameter(
            'working_interval_distance').value
        self.max_error_count = self.get_parameter('max_error_count').value
        self.error_timeout = self.get_parameter('error_timeout').value
        self.state_timeout = self.get_parameter('state_timeout').value

        # 動態速度控制變數
        self.min_speed = self.get_parameter('min_speed').value
        self.deceleration_distance = self.get_parameter(
            'deceleration_distance').value
        self.speed_smoothing_factor = self.get_parameter(
            'speed_smoothing_factor').value

        # 當前速度追蹤
        self.current_speed = self.approach_speed

    def _setup_subscribers(self):
        """設置訂閱器"""
        # 土畦進入狀態接收
        self.furrow_state_sub = self.create_subscription(
            String,
            '/furrow_state',
            self.furrow_state_callback,
            10)

        # 迴轉角度訂閱
        self.rotation_status_sub = self.create_subscription(
            Bool,
            '/rotation_completed',
            self.rotation_completed_callback,
            10)

        # ICP里程計訂閱
        self.odom_sub = self.create_subscription(
            Odometry,
            '/icp_odom',
            self.odom_callback,
            10)

        # 接收視覺導航命令
        self.visual_nav_suggestion_sub = self.create_subscription(
            JoyMotionCommand,
            '/visual_nav_suggestion',
            self.visual_nav_suggestion_callback,
            10
        )

    def _setup_publishers(self):
        """設置發佈器"""
        # 導航命令發佈器
        self.motion_cmd_pub = self.create_publisher(
            JoyMotionCommand, '/nav_command', 10)

        # 角度控制發佈器
        self.target_angle_pub = self.create_publisher(
            Float32, '/target_rotation_angle', 10)

        self.rotation_start_pub = self.create_publisher(
            Bool, '/start_rotation', 10)

        # 狀態發佈器
        self.state_pub = self.create_publisher(String, '/navigation_state', 10)

    def _setup_services(self):
        """設置服務"""
        self.nav_control_service = self.create_service(
            SetBool,
            '/nav_sys_control',
            self.navigation_control_callback
        )

        #  視覺導航控制服務
        self.vision_nav_client = self.create_client(
            SetBool,
            '/vision_navigation_control',
        )

        # 作業操作控制服務
        self.working_operation_client = self.create_client(
            SetBool,
            '/working_operation_control',
        )

        # %　測試用只拍照
        self.detection_operation_client = self.create_client(
            SetBool,
            '/trigger_detection'
        )


# Topic回調函數---------------------------------------------------------------

    def furrow_state_callback(self, msg):
        """call_function: 畦溝狀態信息"""
        self.furrow_state = msg.data
        self.get_logger().debug(f'Furrow state: {self.furrow_state}')

    def rotation_completed_callback(self, msg):
        """call_function: 旋轉完成信號"""
        self.rotation_completed = msg.data

    def odom_callback(self, msg):
        """call_function: 更新當前角度"""
        self.current_pose = msg.pose.pose.position
        self.current_yaw = self._quaternion_to_yaw(msg.pose.pose.orientation)

    def visual_nav_suggestion_callback(self, msg):
        """接收視覺導航建議命令"""
        self.visual_nav_suggestion = msg
        self.get_logger().debug(
            f'收到視覺導航建議: linear={msg.linear_x:.2f}, angle={msg.center_rotate_angle:.1f}')

# Service呼叫函數-------------------------------------------------------------
    def call_vision_navigation_control(self, active):
        """調用視覺導航控制服務（非阻塞）"""
        if not hasattr(self, '_vision_nav_state') or self._vision_nav_state != active:
            if not self.vision_nav_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('視覺導航服務不可用')
                return False

            request = SetBool.Request()
            request.data = active
            future = self.vision_nav_client.call_async(request)

            # 記錄狀態，避免重複呼叫
            self._vision_nav_state = active
            self.get_logger().info(f'視覺導航{"啟用" if active else "禁用"}請求已發送')
            return True
        return True

    def call_working_operation_control(self, active):
        """調用除草作業服務（異步）"""
        if not self.working_operation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('weeding service不可用')
            return None

        request = SetBool.Request()
        request.data = True  # 啟動拍照除草流程

        try:
            self.get_logger().info('發送除草作業請求...')

            # 異步呼叫服務
            future = self.working_operation_client.call_async(request)

            # 設置回調函數來處理響應
            future.add_done_callback(self._working_service_callback)

            # 記錄 future 以便追蹤
            self._working_future = future

            self.get_logger().info('除草作業請求已發送，等待回應...')
            return future

        except Exception as e:
            self.get_logger().error(f'除草作業服務呼叫錯誤: {str(e)}')
            return None

    def _working_service_callback(self, future):
        """除草作業服務回調處理"""
        try:
            if future.done():
                if future.cancelled():
                    self.get_logger().error('除草作業服務被取消')
                    self._working_result = False
                elif future.exception() is not None:
                    self.get_logger().error(f'除草作業服務異常: {future.exception()}')
                    self._working_result = False
                else:
                    response = future.result()
                    if response is not None:
                        self.get_logger().info(
                            f'除草作業完成 - success: {response.success}, message: {response.message}')
                        self._working_result = response.success
                    else:
                        self.get_logger().error('除草作業回應為空')
                        self._working_result = False

                # 標記服務完成
                self._working_service_done = True
            else:
                self.get_logger().warn('除草服務回調被調用但 future 未完成')
                self._working_result = False
                self._working_service_done = True

        except Exception as e:
            self.get_logger().error(f'處理除草服務回應時發生錯誤: {str(e)}')
            self._working_result = False
            self._working_service_done = True

    def call_detection_operation_control(self, active):
        """調用植栽檢測服務（異步）"""
        # 檢查服務是否可用
        if not self.detection_operation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('trigger_detection service不可用')
            return None

        request = SetBool.Request()
        request.data = active

        try:
            self.get_logger().info(f'發送植栽檢測請求... 參數: {active}')

            # 異步呼叫服務 - 不等待結果
            future = self.detection_operation_client.call_async(request)

            # 設置回調函數來處理響應
            future.add_done_callback(self._detection_service_callback)

            # 記錄 future 以便追蹤
            self._detection_future = future

            self.get_logger().info('植栽檢測請求已發送，等待回應...')
            return future  # 返回 future 對象

        except Exception as e:
            self.get_logger().error(f'植栽檢測service呼叫錯誤: {str(e)}')
            return None

    def _detection_service_callback(self, future):
        """植栽檢測服務回調處理"""
        try:
            if future.done():
                if future.cancelled():
                    self.get_logger().error('植栽檢測服務被取消')
                    self._detection_result = False
                elif future.exception() is not None:
                    self.get_logger().error(f'植栽檢測服務異常: {future.exception()}')
                    self._detection_result = False
                else:
                    response = future.result()
                    if response is not None:
                        self.get_logger().info(
                            f'植栽檢測完成 - success: {response.success}, message: {response.message}')
                        self._detection_result = response.success
                    else:
                        self.get_logger().error('植栽檢測回應為空')
                        self._detection_result = False

                # 標記服務完成
                self._detection_service_done = True
            else:
                self.get_logger().warn('檢測服務回調被調用但 future 未完成')
                self._detection_result = False
                self._detection_service_done = True

        except Exception as e:
            self.get_logger().error(f'處理檢測服務回應時發生錯誤: {str(e)}')
            self._detection_result = False
            self._detection_service_done = True


# Service回調函數-------------------------------------------------------------

    def navigation_control_callback(self, request, response):
        """Service call_function: 導航控制服務回調"""
        try:
            if request.data:  # 啟動導航
                if self.current_state == NavigationState.MANUAL:
                    # 重置錯誤計數器
                    self.error_count = 0
                    self.last_error_time = None

                    self._change_state(NavigationState.ROW_IN_APPROACH)
                    response.success = True
                    response.message = "導航已啟動"
                elif self.current_state == NavigationState.EMERGENCY_STOP:
                    # 從異常停止恢復
                    self.get_logger().info('從異常停止狀態恢復')
                    self.error_count = 0
                    self.last_error_time = None
                    if hasattr(self, '_emergency_logged'):
                        delattr(self, '_emergency_logged')

                    self._change_state(NavigationState.MANUAL)
                    response.success = True
                    response.message = "已從異常停止恢復到手動模式"
                else:
                    response.success = False
                    response.message = f"無法啟動：當前狀態為 {self.current_state.value}"
            else:  # 關閉導航
                self.get_logger().info('關閉自主導航')
                self._change_state(NavigationState.MANUAL)
                self.call_vision_navigation_control(False)
                # 停止所有運動
                self._send_motion_command(0.0, 0.0, 0)

                # 重置錯誤狀態
                self.error_count = 0
                self.last_error_time = None
                if hasattr(self, '_emergency_logged'):
                    delattr(self, '_emergency_logged')

                response.success = True
                response.message = "導航已關閉"

        except Exception as e:
            self.get_logger().error(f'導航控制服務錯誤: {str(e)}')
            self._handle_error(f'導航控制服務錯誤: {str(e)}')
            response.success = False
            response.message = f"服務錯誤: {str(e)}"

        return response

# FSM -------------------------------------------------------------
    def state_machine_loop(self):
        """狀態機主循環"""
        # 發布當前狀態
        state_msg = String()
        state_msg.data = self.current_state.value
        self.state_pub.publish(state_msg)

        # 狀態處理
        if self.current_state == NavigationState.MANUAL:  # 手動模式
            self._handle_manual_state()
        elif self.current_state == NavigationState.ROW_IN_APPROACH:  # 入行接近狀態
            self._handle_row_in_approach_state()
        elif self.current_state == NavigationState.ROW_FOLLOWING:  # 行內跟隨狀態
            self._handle_row_following_state()
        elif self.current_state == NavigationState.WORKING_OPERATION:  # 作業操作狀態
            self._handle_working_operation_state()
        elif self.current_state == NavigationState.MOVING_TO_NEXT_WORK_POINT:  # 移動到下個作業點
            self._handle_moving_to_next_work_point_state()
        elif self.current_state == NavigationState.ROW_END_APPROACH:  # 畦尾接近狀態
            self._handle_row_end_approach_state()
        elif self.current_state == NavigationState.TURNING_PHASE1:  # 轉向旋轉90度
            self._handle_turning_phase1_state()
        elif self.current_state == NavigationState.TURNING_PHASE2:  # 往前開
            self._handle_turning_phase2_state()
        elif self.current_state == NavigationState.TURNING_PHASE3:  # 回正旋轉90度
            self._handle_turning_phase3_state()
        elif self.current_state == NavigationState.EMERGENCY_STOP:  # 緊急停止
            self._handle_emergency_stop_state()

    def _handle_manual_state(self):
        """處理手動模式狀態"""
        pass

    def _handle_row_in_approach_state(self):
        """處理入行接近狀態"""
        try:
            # 檢查是否進入土畦
            if hasattr(self, 'furrow_state') and self.furrow_state == "in_field":
                self.get_logger().info('進入土畦，預備第一次作業操作')
                self.is_first_work_in_row = True
                self._change_state(NavigationState.WORKING_OPERATION)
            else:
                approach_speed = self.approach_speed * 0.8  # 稍微減速以提高精度

                self._send_motion_command(
                    linear_x=approach_speed,
                    turning_mode=0
                )

                self.get_logger().debug(f'接近土畦中，速度: {approach_speed:.3f}m/s')
        except Exception as e:
            self._handle_error(f"入行接近狀態錯誤: {str(e)}")

    def _handle_row_following_state(self):
        """處理行內跟隨狀態"""
        # 檢查是否到達畦尾
        if hasattr(self, 'furrow_state') and self.furrow_state == "out_field":
            self.get_logger().info('到達畦尾，準備轉彎')
            self._change_state(NavigationState.ROW_END_APPROACH)
        else:
            # 繼續跟隨導航線
            pass

    def _handle_working_operation_state(self):
        """田間管理作業操作狀態"""
        if not hasattr(self, '_working_started'):
            # 停止移動並記錄當前位置
            self._send_motion_command(0.0, 0.0, 0)

            work_type = "第一次" if self.is_first_work_in_row else "後續"
            self.get_logger().info(f'開始{work_type}作業操作')

            # 記錄作業開始位置
            if self.current_pose:
                self.work_point_start_position = self.current_pose

            self._working_started = True
            self.get_logger().info('作業操作狀態初始化完成，準備調用服務')

        # 如果還沒有啟動服務，啟動它
        if not hasattr(self, '_detection_service_started'):
            # 調用植栽檢測服務（異步）
            future = self.call_detection_operation_control(True)
            if future is not None:
                self._detection_service_started = True
                self._detection_service_done = False
                self._detection_result = None
                self._service_start_time = self.get_clock().now()
                self.get_logger().info('植栽檢測服務已啟動，等待回應...')
            else:
                self.get_logger().error('無法啟動植栽檢測服務')
                self._handle_service_failure()
                return

        # 檢查服務是否已完成
        if hasattr(self, '_detection_service_done') and self._detection_service_done:
            # 服務已完成，檢查結果
            if hasattr(self, '_detection_result') and self._detection_result:
                self.get_logger().info('植栽檢測作業成功完成')
                self._cleanup_working_operation_state()

                # 檢查是否還在土畦內
                if hasattr(self, 'furrow_state') and self.furrow_state == "out_field":
                    self.get_logger().info('作業完成時已離開土畦，準備轉彎')
                    self._change_state(NavigationState.ROW_END_APPROACH)
                else:
                    self.get_logger().info('作業完成，移動到下一個作業點')
                    self._change_state(
                        NavigationState.MOVING_TO_NEXT_WORK_POINT)
            else:
                self.get_logger().error('植栽檢測作業失敗')
                self._handle_service_failure()

        # 檢查服務超時（30秒）
        elif hasattr(self, '_service_start_time'):
            current_time = self.get_clock().now()
            elapsed_time = (
                current_time - self._service_start_time).nanoseconds / 1e9

            if elapsed_time > 30.0:  # 30秒超時
                self.get_logger().error('植栽檢測服務超時（30秒）')
                self._handle_service_failure()

    def _handle_service_failure(self):
        """處理服務失敗"""
        self._handle_error('植栽檢測作業失敗')
        self._cleanup_working_operation_state()

    def _cleanup_working_operation_state(self):
        """清理作業操作狀態變數"""
        # 清理狀態變數
        attrs_to_clean = [
            '_working_started',
            '_detection_service_started',
            '_detection_service_done',
            '_detection_result',
            '_service_start_time',
            '_detection_future'
        ]

        for attr in attrs_to_clean:
            if hasattr(self, attr):
                delattr(self, attr)

        self.is_first_work_in_row = False

    def _handle_moving_to_next_work_point_state(self):
        """處理移動到下一個作業點狀態"""
        if not hasattr(self, '_movement_started'):
            # 檢查當前位置是否可用
            if self.current_pose is None:
                self.get_logger().error('無法獲取當前位置，無法開始移動到下一個作業點')
                self._handle_error('移動狀態：無法獲取當前位置')
                return

            # 記錄移動開始位置
            self._movement_start_position = self.current_pose
            self._movement_started = True
            self.get_logger().info(
                f'開始移動到下一個作業點（距離：{self.working_interval_distance}公尺）')

        # 檢查是否離開土畦
        if hasattr(self, 'furrow_state') and self.furrow_state == "out_field":
            self.get_logger().info('移動過程中離開土畦，準備轉彎')
            self._cleanup_movement_variables()
            self._change_state(NavigationState.ROW_END_APPROACH)
            return

        # 檢查當前位置和開始位置是否可用
        current_position = self.current_pose
        if current_position is None or not hasattr(self, '_movement_start_position') or self._movement_start_position is None:
            self.get_logger().warn('無法獲取位置信息進行距離計算，繼續前進')
            # 使用備用方案：繼續前進一段時間
            if not hasattr(self, '_movement_start_time'):
                self._movement_start_time = self.get_clock().now()

            current_time = self.get_clock().now()
            elapsed_time = (
                current_time - self._movement_start_time).nanoseconds / 1e9

            # 根據設定距離和速度計算需要的時間
            required_time = self.working_interval_distance / self.approach_speed

            if elapsed_time < required_time:
                # 使用視覺導航建議進行移動（FSM 控制速度）
                
                if hasattr(self, 'visual_nav_suggestion') and self.visual_nav_suggestion is not None:
                    self._send_motion_command(
                        linear_x=self.approach_speed,  # FSM 控制前進速度
                        center_rotate_angle=self.visual_nav_suggestion.center_rotate_angle,  # 使用視覺導航轉向
                        turning_mode=0
                    )
                    self.get_logger().debug(
                        f'使用視覺導航建議移動 - 時間: {elapsed_time:.1f}/{required_time:.1f}s')
                else:
                    # 沒有視覺導航建議時直行
                    self._send_motion_command(
                        linear_x=self.approach_speed,
                        turning_mode=0
                    )
                    self.get_logger().debug(
                        f'直線移動 - 時間: {elapsed_time:.1f}/{required_time:.1f}s')
            else:
                self.get_logger().info(f'移動時間達到{elapsed_time:.1f}秒，開始下一次作業')
                # FSM 立即停止，忽略視覺導航建議
                self._send_motion_command(0.0, 0.0, 0)
                self._cleanup_movement_variables()
                self._change_state(NavigationState.WORKING_OPERATION)
            return

        # 計算已移動距離
        distance_moved = math.sqrt(
            (current_position.x - self._movement_start_position.x) ** 2 +
            (current_position.y - self._movement_start_position.y) ** 2
        )

        # 關鍵修改：提前停止，預留視覺導航延遲補償
        stop_distance_compensation = 0.12  # 預留 12cm 的延遲補償
        effective_target = self.working_interval_distance - stop_distance_compensation

        if distance_moved < effective_target:
            # 計算動態速度：接近目標時減速
            remaining_distance = effective_target - distance_moved
            if remaining_distance < 0.1:  # 最後 10cm 大幅減速
                dynamic_speed = self.approach_speed * 0.3
            elif remaining_distance < 0.2:  # 最後 20cm 適度減速
                dynamic_speed = self.approach_speed * 0.6
            else:
                dynamic_speed = self.approach_speed

            # 使用視覺導航建議進行移動（FSM 控制速度）
            if hasattr(self, 'visual_nav_suggestion') and self.visual_nav_suggestion is not None:
                self._send_motion_command(
                    linear_x=dynamic_speed,  # FSM 控制動態速度
                    center_rotate_angle=self.visual_nav_suggestion.center_rotate_angle,  # 使用視覺導航轉向
                    turning_mode=0
                )
                self.get_logger().debug(
                    f'視覺導航移動 - 進度: {distance_moved:.2f}/{self.working_interval_distance:.2f}m, '
                    f'速度: {dynamic_speed:.3f}m/s, 轉向: {self.visual_nav_suggestion.center_rotate_angle:.1f}°')
            else:
                # 沒有視覺導航建議時直行
                self._send_motion_command(
                    linear_x=dynamic_speed,
                    turning_mode=0
                )
                self.get_logger().debug(
                    f'直線移動 - 進度: {distance_moved:.2f}/{self.working_interval_distance:.2f}m, '
                    f'速度: {dynamic_speed:.3f}m/s')
        else:
            # 到達預停點，FSM 立即停止並停用視覺導航
            self.get_logger().info(f'到達預停點 {distance_moved:.2f}m，停止視覺導航')

            # 立即停止運動（FSM 完全控制）
            self._send_motion_command(0.0, 0.0, 0)

            # 停用視覺導航（避免後續干擾）
            self.call_vision_navigation_control(False)

            # 等待系統穩定
            import time
            time.sleep(0.2)

            # 最終檢查移動距離
            final_position = self.current_pose
            if final_position:
                final_distance = math.sqrt(
                    (final_position.x - self._movement_start_position.x) ** 2 +
                    (final_position.y - self._movement_start_position.y) ** 2
                )
                self.get_logger().info(
                    f'最終移動距離: {final_distance:.2f}公尺，開始下一次作業')

            self._cleanup_movement_variables()
            self._change_state(NavigationState.WORKING_OPERATION)

    def _handle_row_end_approach_state(self):
        """處理畦尾接近狀態 - 繼續前進直到完全離開土畦"""
        if not hasattr(self, '_exit_started'):
            self._exit_started = True
            self._exit_start_position = self.current_pose if self.current_pose else None
            self.get_logger().info('檢測到畦尾, 繼續前進0.5公尺確保完全離開土畦')

        # 檢查是否有有效的位置信息
        if self.current_pose is None or not hasattr(self, '_exit_start_position') or self._exit_start_position is None:
            self.get_logger().warn('無法獲取位置信息，使用時間方式退出')
            # 備用方案：使用時間
            if not hasattr(self, '_exit_start_time'):
                self._exit_start_time = self.get_clock().now()

            current_time = self.get_clock().now()
            elapsed_time = (
                current_time - self._exit_start_time).nanoseconds / 1e9

            if elapsed_time < 2.5:  # 備用時間：2.5秒
                self._send_motion_command(
                    linear_x=self.approach_speed,
                    turning_mode=0
                )
            else:
                # 完全離開後開始轉彎程序
                self.current_row += 1
                if self.current_row >= self.total_rows:
                    self.get_logger().info('所有行作業完成')
                    self._change_state(NavigationState.MANUAL)
                else:
                    self.get_logger().info(
                        f'車體已完全離開土畦，開始轉彎程序，當前第{self.current_row}行')
                    self._change_state(NavigationState.TURNING_PHASE1)
                self._cleanup_exit_variables()
            return

        # 計算已移動的距離
        current_position = self.current_pose
        distance_moved = math.sqrt(
            (current_position.x - self._exit_start_position.x) ** 2 +
            (current_position.y - self._exit_start_position.y) ** 2
        )

        if distance_moved < self.out_furrow_distance:  # 尚未移動到工作距離
            # 計算動態速度：隨著接近退出距離而減速
            dynamic_speed = self._calculate_dynamic_speed(
                distance_to_target=distance_moved,
                target_distance=self.out_furrow_distance,
                base_speed=self.approach_speed
            )

            # 繼續前進
            self._send_motion_command(
                linear_x=dynamic_speed,
                turning_mode=0
            )

            self.get_logger().debug(
                f'退出進度: {distance_moved:.2f}/{self.out_furrow_distance:.2f}m, '
                f'當前速度: {dynamic_speed:.3f}m/s'
            )
        else:
            # 完全離開後開始轉彎程序
            self.current_row += 1

            # 檢查是否完成所有行
            if self.current_row >= self.total_rows:
                self.get_logger().info('所有行作業完成')
                self._change_state(NavigationState.MANUAL)
            else:
                self.get_logger().info(
                    f'車體已完全離開土畦，開始轉彎程序，當前第{self.current_row}行')
                self._change_state(NavigationState.TURNING_PHASE1)

    # @ ---------------- 換行 --------------------

    def _handle_turning_phase1_state(self):
        """處理轉彎第一階段:旋轉90度"""
        if not hasattr(self, '_rotation_started'):
            self._rotation_started = True
            # 根據當前行數決定旋轉方向
            # TODO
            target_angle = 90.0 if self.current_row % 2 == 1 else -90.0
            self._phase1_target_angle = target_angle
            self.get_logger().info(
                f'Phase 1: 第{self.current_row}行，開始旋轉{target_angle}度')

        self._start_rotation(self._phase1_target_angle)

        if self.rotation_completed:
            self.get_logger().info('Phase 1: 第一次旋轉完成')
            self._change_state(NavigationState.TURNING_PHASE2)
            delattr(self, '_rotation_started')
            delattr(self, '_phase1_target_angle')

    def _handle_turning_phase2_state(self):
        """處理轉彎第二階段:前進一段距離"""
        try:
            if not hasattr(self, '_forward_started'):
                # 記錄開始前進時的位置
                if self.current_pose:
                    self._forward_start_position = self.current_pose
                else:
                    self.get_logger().warn('無法獲取導航位置，退回時間控制方式')
                    self._forward_start_time = self.get_clock().now()

                # 重置速度為轉彎速度
                self.current_speed = self.turn_speed

                self._send_motion_command(
                    linear_x=self.turn_speed,
                    turning_mode=0
                )
                self._forward_started = True
                self.get_logger().info('Phase 2: 開始前進到下一行')

            # 優先使用距離判斷
            if hasattr(self, '_forward_start_position') and self.current_pose:
                # 計算已移動的距離
                current_position = self.current_pose
                distance_moved = math.sqrt(
                    (current_position.x - self._forward_start_position.x) ** 2 +
                    (current_position.y - self._forward_start_position.y) ** 2
                )

                # 計算動態速度：隨著接近目標距離而減速
                dynamic_speed = self._calculate_dynamic_speed(
                    distance_to_target=distance_moved,
                    target_distance=self.row_spacing,
                    base_speed=self.turn_speed
                )

                # 更新運動指令
                self._send_motion_command(
                    linear_x=dynamic_speed,
                    turning_mode=0
                )

                self.get_logger().debug(
                    f'Phase 2 進度: {distance_moved:.2f}/{self.row_spacing:.2f}m, '
                    f'當前速度: {dynamic_speed:.3f}m/s'
                )

                if distance_moved >= self.row_spacing:
                    self.get_logger().info(
                        f'Phase 2: 前進完成，移動距離: {distance_moved:.2f}m')
                    self._change_state(NavigationState.TURNING_PHASE3)
                    self._cleanup_forward_variables()
            elif hasattr(self, '_forward_start_time'):
                # 使用時間備用方案
                current_time = self.get_clock().now()
                elapsed_time = (
                    current_time - self._forward_start_time).nanoseconds / 1e9

                # 根據設定距離和速度計算需要的時間
                required_time = self.row_spacing / self.turn_speed

                self._send_motion_command(
                    linear_x=self.turn_speed,
                    turning_mode=0
                )

                self.get_logger().debug(
                    f'Phase 2 進度（時間）: {elapsed_time:.1f}/{required_time:.1f}s'
                )

                if elapsed_time >= required_time:
                    self.get_logger().info(
                        f'Phase 2: 前進完成（時間方式），耗時: {elapsed_time:.1f}s')
                    self._change_state(NavigationState.TURNING_PHASE3)
                    self._cleanup_forward_variables()
            else:
                # 異常情況：沒有開始位置也沒有開始時間
                raise Exception('缺少位置和時間信息')

        except Exception as e:
            self._handle_error(f"轉彎第二階段錯誤: {str(e)}")
            self._cleanup_forward_variables()

    def _handle_turning_phase3_state(self):
        """處理轉彎第三階段:再次旋轉90度"""
        if not hasattr(self, '_rotation3_started'):
            # 第三階段與第一階段旋轉方向相同
            target_angle = 90.0 if self.current_row % 2 == 1 else -90.0
            self._phase3_target_angle = target_angle
            self._rotation3_started = True
            self.get_logger().info(
                f'Phase 3: 第{self.current_row}行，開始第二次旋轉{target_angle}度')

        self._start_rotation(self._phase3_target_angle)

        if self.rotation_completed:
            self.get_logger().info('Phase 3: 第二次旋轉完成')
            self._change_state(NavigationState.ROW_IN_APPROACH)
            delattr(self, '_rotation3_started')
            delattr(self, '_phase3_target_angle')

    def _start_rotation(self, angle_degrees):
        """開始旋轉操作 - 迴轉功能版本"""
        # 檢查是否為新的旋轉任務
        if not hasattr(self, '_rotation_target') or self._rotation_target != angle_degrees:
            # 初始化新的旋轉任務
            self._rotation_target = angle_degrees
            self._start_yaw = math.degrees(self.current_yaw)
            self._accumulated_rotation = 0.0
            self._last_yaw = self._start_yaw

            self.get_logger().info(
                f'開始迴轉 {angle_degrees}°，起始角度: {self._start_yaw:.1f}°')

        # 計算相對於上次的角度變化
        current_yaw_deg = math.degrees(self.current_yaw)
        yaw_change = (current_yaw_deg - self._last_yaw) * -1

        # 處理角度跳躍 (-180° 到 +180° 或反之)
        if yaw_change > 180:
            yaw_change -= 360
        elif yaw_change < -180:
            yaw_change += 360

        # 累積旋轉角度
        self._accumulated_rotation += yaw_change
        self._last_yaw = current_yaw_deg

        # 計算還需要旋轉的角度
        remaining_rotation = self._rotation_target - self._accumulated_rotation

        # 檢查是否完成旋轉
        if abs(remaining_rotation) <= 5.0:  # 5度容忍度
            self.rotation_completed = True
            self._send_motion_command(0.0, 0.0, 0)  # 停止

            final_yaw = math.degrees(self.current_yaw)
            self.get_logger().info(
                f'迴轉完成！累積旋轉: {self._accumulated_rotation:.1f}°, 最終角度: {final_yaw:.1f}°')

            # 清理所有旋轉相關變數
            self._clear_rotation_variables()
        else:
            # 根據還需要旋轉的方向決定旋轉速度
            if remaining_rotation > 0:
                rotation_speed = -0.0  # 逆時針
            else:
                rotation_speed = 0.0   # 順時針

            self._send_motion_command(0.0, rotation_speed, 1)

            # 調試信息
            if abs(remaining_rotation) > 10:
                self.get_logger().info(
                    f'迴轉中 - 當前: {current_yaw_deg:.1f}°, 累積旋轉: {self._accumulated_rotation:.1f}°, '
                    f'目標旋轉: {self._rotation_target:.1f}°, 剩餘: {remaining_rotation:.1f}°')

    def _clear_rotation_variables(self):
        """清理所有旋轉相關變數"""
        attrs_to_clear = ['_rotation_target', '_start_yaw',
                          '_accumulated_rotation', '_last_yaw']
        for attr in attrs_to_clear:
            if hasattr(self, attr):
                delattr(self, attr)

    #! 緊急處理類函式--------------------------------------------------------------
    def _check_state_timeout(self):
        """檢查狀態是否超時"""
        current_time = self.get_clock().now()
        time_in_state = (
            current_time - self.state_start_time).nanoseconds / 1e9

        # 某些狀態允許較長時間
        timeout_exceptions = {
            NavigationState.MANUAL: float('inf'),  # 手動模式無超時
            NavigationState.WORKING_OPERATION: 120.0,  # 作業操作允許2分鐘
            NavigationState.EMERGENCY_STOP: float('inf')  # 異常停止無超時
        }

        timeout_limit = timeout_exceptions.get(
            self.current_state, self.state_timeout)

        if time_in_state > timeout_limit:
            self._handle_error(
                f"狀態 {self.current_state.value} 超時 ({time_in_state:.1f}s)")

    def _handle_error(self, error_message):
        """統一錯誤處理"""
        current_time = self.get_clock().now()

        # 記錄錯誤
        self.get_logger().error(error_message)

        # 重置錯誤計數器如果距離上次錯誤超過timeout時間
        if (self.last_error_time is None or
                (current_time - self.last_error_time).nanoseconds / 1e9 > self.error_timeout):
            self.error_count = 0

        self.error_count += 1
        self.last_error_time = current_time

        # 檢查是否需要進入異常停止
        if self.error_count >= self.max_error_count:
            self.get_logger().error(f'錯誤次數達到上限 ({self.error_count})，進入異常停止狀態')
            self._change_state(NavigationState.EMERGENCY_STOP)
        else:
            self.get_logger().warn(
                f'錯誤計數: {self.error_count}/{self.max_error_count}')

    def _handle_emergency_stop_state(self):
        """處理異常停止狀態"""
        # 立即停止所有運動
        self._send_motion_command(0.0, 0.0, 0)

        # 關閉視覺導航
        if hasattr(self, '_vision_nav_state') and self._vision_nav_state:
            self.call_vision_navigation_control(False)

        # 記錄異常停止日誌
        if not hasattr(self, '_emergency_logged'):
            self.get_logger().error('系統進入異常停止狀態 - 所有運動已停止')
            self.get_logger().error('請檢查系統狀態，手動重置後才能恢復運行')
            self._emergency_logged = True

        # 在異常停止狀態下等待手動重置（通過服務調用）
        self._wait_for_manual_reset()

    def _wait_for_manual_reset(self):
        """等待手動重置 - 在異常停止狀態下調用"""
        # 這個函數在異常停止狀態下被定期調用
        # 實際的重置是通過 navigation_control_callback 服務來處理的
        pass

    def _cleanup_movement_variables(self):
        """清理移動相關的狀態變數"""
        attrs_to_clean = [
            '_movement_started',
            '_movement_start_position',
            '_movement_start_time'
        ]

        for attr in attrs_to_clean:
            if hasattr(self, attr):
                delattr(self, attr)

    def _cleanup_forward_variables(self):
        """清理前進階段的狀態變數"""
        try:
            attrs_to_clean = [
                '_forward_started',
                '_forward_start_position',
                '_forward_start_time'
            ]

            for attr in attrs_to_clean:
                if hasattr(self, attr):
                    delattr(self, attr)
        except Exception as e:
            self.get_logger().warn(f'清理前進變數時發生錯誤: {str(e)}')

    def _cleanup_exit_variables(self):
        """清理退出階段的狀態變數"""
        try:
            if hasattr(self, '_exit_started'):
                delattr(self, '_exit_started')
            if hasattr(self, '_exit_start_position'):
                delattr(self, '_exit_start_position')
            if hasattr(self, '_exit_start_time'):
                delattr(self, '_exit_start_time')
        except Exception as e:
            self.get_logger().warn(f'清理退出變數時發生錯誤: {str(e)}')

    # 功能函數-------------------------------------------------------------------
    def _change_state(self, new_state):
        """狀態轉換"""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = self.get_clock().now()
        self.rotation_completed = False  # 重置旋轉完成標誌

        # 清理旋轉變數（確保狀態切換時清理）
        self._clear_rotation_variables()

        # 根據新狀態控制視覺導航
        if new_state in [NavigationState.ROW_FOLLOWING, NavigationState.MOVING_TO_NEXT_WORK_POINT]:
            self.call_vision_navigation_control(True)
        else:
            self.call_vision_navigation_control(False)

        self.get_logger().info(
            f'State changed: {self.previous_state.value} -> {new_state.value}')

    def _normalize_angle(self, angle):
        """將角度標準化到 [-π, π]"""
        import math
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _quaternion_to_yaw(self, quaternion):
        """四元數轉偏航角"""
        import math
        siny_cosp = 2 * (quaternion.w * quaternion.z +
                         quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y +
                             quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _send_motion_command(self, linear_x=0.0, center_rotate_angle=0.0, turning_mode=0):
        """統一的運動指令發送方法"""
        cmd = JoyMotionCommand()
        cmd.linear_x = float(linear_x)
        cmd.center_rotate_angle = float(center_rotate_angle)
        cmd.turning_mode = turning_mode

        # 記錄當前發送的命令
        self.last_motion_command = cmd

        self.motion_cmd_pub.publish(cmd)
        self.get_logger().debug(
            f'發送運動指令: linear={linear_x:.2f}, angle={center_rotate_angle:.1f}, mode={turning_mode}')

    def _calculate_dynamic_speed(self, distance_to_target, target_distance, base_speed=None):
        """
        計算動態速度：隨著距離目標點越近而逐漸減速

        Args:
            distance_to_target: 到目標點的剩餘距離
            target_distance: 總目標距離
            base_speed: 基礎速度（預設使用 approach_speed）

        Returns:
            計算後的動態速度
        """
        if base_speed is None:
            base_speed = self.approach_speed

        # 計算剩餘距離比例
        remaining_distance = target_distance - distance_to_target

        self.get_logger().info(f"剩餘距離{remaining_distance:.2f}公尺")

        if remaining_distance <= self.deceleration_distance:
            # 在減速區域內，線性減速到最小速度
            speed_ratio = max(remaining_distance / self.deceleration_distance,
                              self.min_speed / base_speed)
            target_speed = base_speed * speed_ratio
        else:
            # 不在減速區域，使用基礎速度
            target_speed = base_speed

        # 平滑速度變化，避免突然加減速
        self.current_speed = (self.speed_smoothing_factor * self.current_speed +
                              (1 - self.speed_smoothing_factor) * target_speed)

        # 確保速度在合理範圍內
        self.current_speed = max(self.min_speed, min(
            self.current_speed, base_speed))

        return self.current_speed

    def _reset_speed(self):
        """重置當前速度為基礎接近速度"""
        self.current_speed = self.approach_speed

    def _get_current_speed_info(self):
        """獲取當前速度資訊用於除錯"""
        return {
            'current_speed': self.current_speed,
            'approach_speed': self.approach_speed,
            'turn_speed': self.turn_speed,
            'min_speed': self.min_speed
        }


def main(args=None):
    """主函數"""
    rclpy.init(args=args)

    try:
        nav_controller = NavFSMController()
        rclpy.spin(nav_controller)
    except KeyboardInterrupt:
        pass
    finally:
        if 'nav_controller' in locals():
            nav_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
