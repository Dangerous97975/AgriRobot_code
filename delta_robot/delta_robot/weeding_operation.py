#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from std_msgs.msg import Bool
import threading
import time


class WeederOperationController(Node):
    def __init__(self):
        super().__init__('weeder_operation_controller')

        # 控制狀態
        self.is_operation_running = False
        self.operation_success = False
        self.operation_message = ""

        # 步驟完成標誌
        self.delta_complete = False

        # 創建服務
        self.operation_service = self.create_service(
            SetBool,
            '/weeder_operation_control',
            self.operation_control_callback
        )

        # 創建客戶端 - 觸發植物檢測
        self.detection_trigger_client = self.create_client(
            SetBool,
            '/trigger_detection'
        )

        # 訂閱者 - 監聽 Delta 執行完成狀態
        self.delta_status_sub = self.create_subscription(
            Bool,
            '/delta_execution_complete',
            self.delta_complete_callback,
            10
        )

        # 初始化完成
        self.get_logger().info('除草作業控制器已啟動')
        self.get_logger().info('等待服務調用: weeder_operation_control')

    def operation_control_callback(self, request, response):
        """處理除草作業控制請求"""

        if not request.data:
            # 停止指令（除草作業不支援中途停止）
            response.success = True
            response.message = '除草作業不支援中途停止'
            self.get_logger().info('收到停止指令，但除草作業不支援中途停止')
            return response

        if self.is_operation_running:
            response.success = False
            response.message = '除草作業正在進行中，請稍後再試'
            self.get_logger().warn('除草作業已在進行中')
            return response

        # 啟動除草作業
        self.get_logger().info('開始除草作業流程')

        # 在新執行緒中執行作業以避免阻塞服務回應
        operation_thread = threading.Thread(
            target=self.execute_weeding_operation)
        operation_thread.daemon = True
        operation_thread.start()

        # 等待作業完成
        timeout = 120.0
        start_time = time.time()

        while self.is_operation_running and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        # 回應結果
        if self.is_operation_running:
            # 超時
            response.success = False
            response.message = '除草作業超時'
            self.get_logger().error('除草作業執行超時')
        else:
            # 正常完成
            response.success = self.operation_success
            response.message = self.operation_message

        return response

    def execute_weeding_operation(self):
        """執行完整的除草作業流程"""

        self.is_operation_running = True
        self.operation_success = False
        self.operation_message = ""

        # 重置完成標誌
        self.delta_complete = False

        try:
            # Step 1: 觸發植物檢測並等待完成
            self.get_logger().info('步驟 1: 執行植物檢測')
            if not self.trigger_and_wait_detection():
                self.operation_success = False
                self.operation_message = '植物檢測失敗'
                return

            # Step 2: 座標轉換已自動完成 (由 trajectory_plan 自動處理)
            self.get_logger().info('步驟 2: 座標轉換已自動完成')

            # Step 3: 等待 Delta 機器人執行完成
            self.get_logger().info('步驟 3: 等待 Delta 機器人執行完成')
            if not self.wait_for_delta_complete():
                self.operation_success = False
                self.operation_message = 'Delta 機器人執行超時或失敗'
                return

            # 所有步驟完成
            self.operation_success = True
            self.operation_message = '除草作業完成成功'
            self.get_logger().info('除草作業流程全部完成')

        except Exception as e:
            self.operation_success = False
            self.operation_message = f'除草作業執行錯誤: {str(e)}'
            self.get_logger().error(f'除草作業執行錯誤: {str(e)}')

        finally:
            self.is_operation_running = False

    def trigger_and_wait_detection(self):
        """觸發plant_detection並同步等待完成，失敗時重試一次"""

        # 等待檢測服務可用
        if not self.detection_trigger_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Plant detection service不可用')
            return False

        # 最多嘗試2次
        max_attempts = 2
        for attempt in range(max_attempts):
            self.get_logger().info(
                f'plant_detection嘗試 {attempt + 1}/{max_attempts}')

            # 發送觸發請求（同步等待完成）
            request = SetBool.Request()
            request.data = True

            try:
                future = self.detection_trigger_client.call_async(request)
                rclpy.spin_until_future_complete(
                    self, future, timeout_sec=30.0)

                if future.done():
                    response = future.result()
                    if response.success:
                        self.get_logger().info(
                            f'plant_detection完成: {response.message}')
                        return True
                    else:
                        self.get_logger().warn(
                            f'plant_detection失敗: {response.message}')

                        # 如果是影像處理失敗且還有重試機會，則繼續重試
                        if "影像處理失敗" in response.message and attempt < max_attempts - 1:
                            self.get_logger().info('檢測到影像處理失敗，準備重試...')
                            time.sleep(1.0)  # 等待1秒後重試
                            continue
                        else:
                            return False
                else:
                    self.get_logger().error('plant_detection服務超時')
                    if attempt < max_attempts - 1:
                        self.get_logger().info('服務超時，準備重試...')
                        time.sleep(1.0)
                        continue
                    else:
                        return False

            except Exception as e:
                self.get_logger().error(f'plant_detection服務錯誤: {str(e)}')
                if attempt < max_attempts - 1:
                    self.get_logger().info('服務錯誤，準備重試...')
                    time.sleep(1.0)
                    continue
                else:
                    return False

        return False

    def delta_complete_callback(self, msg):
        """Delta 執行完成回調"""
        if msg.data:
            self.delta_complete = True
            self.get_logger().info('收到 Delta 執行完成信號')

    def wait_for_detection_complete(self, timeout=30.0):
        """等待檢測完成（已移除，現在使用同步服務調用）"""
        # 此函式已不再需要，檢測完成狀態通過服務回應直接獲得
        pass

    def wait_for_delta_complete(self, timeout=45.0):
        """等待 Delta 機器人完成"""
        start_time = time.time()

        self.get_logger().info('等待 Delta 機器人執行完成...')

        while not self.delta_complete and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

        if self.delta_complete:
            self.get_logger().info('Delta 機器人執行完成')
            return True
        else:
            self.get_logger().error('Delta 機器人執行超時')
            return False

    def destroy_node(self):
        """節點銷毀時的清理工作"""
        self.get_logger().info('正在關閉除草作業控制器...')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    controller = WeederOperationController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('收到中斷信號')
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
