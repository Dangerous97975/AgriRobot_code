#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Point
import numpy as np
import serial
import time
import threading
from queue import Queue


class DeltaXController:
    def __init__(self, port, baudrate=115200):
        """初始化 Delta X 控制器"""
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.is_connected = False

    def connect(self):
        """建立與機器人的連線並初始化"""
        try:
            self.ser = serial.Serial(
                self.port, self.baudrate, timeout=3)  # 增加 timeout
            time.sleep(2)

            initial_response = self.ser.readline().decode().strip()
            print(
                f"Initial response: {initial_response}, \nSuccessfully connected to Delta X on {self.port}")

            self.is_connected = True

            # 等待機器人完全準備好
            time.sleep(3)

            # 清空緩衝區
            self.ser.flushInput()
            self.ser.flushOutput()

            # 初始化後移動到待命位置
            print(
                "Moving to standby position after connection...")
            success = self.move_to_standby_position()
            if not success:
                print(
                    "Warning: Failed to move to standby position initially")
                return False

            return True

        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False

    def send_gcode_with_timing(self, gcode, timeout=10):
        """發送 G-code 指令並等待回應"""
        if not self.is_connected:
            print("Error: Not connected to robot")
            return False, 0

        try:
            start_time = time.time()

            command = gcode + '\n'
            self.ser.write(command.encode())
            print(f"Sent: {gcode}")

            # 使用更長的超時時間來避免卡住
            end_time = start_time + timeout

            while time.time() < end_time:
                if self.ser.in_waiting > 0:  # 檢查是否有數據等待讀取
                    response = self.ser.readline().decode().strip()
                    if response:
                        print(f"Response: {response}")
                        if 'Ok' in response:
                            execution_time = time.time() - start_time
                            print(
                                f"Command completed (Time: {execution_time:.3f}s)")
                            return True, execution_time
                        elif 'error' in response.lower():
                            print(
                                f"Error from robot: {response}")
                            return False, 0

                time.sleep(0.1)  # 避免忙碌等待

            # 超時處理
            print(f"Timeout waiting for response to: {gcode}")
            return False, 0

        except Exception as e:
            print(f"Communication error: {e}")
            return False, 0

    def move_to_standby_position(self, standby_pos=(0, -350, -550)):
        """
        移動到待命位置, 先移動Z軸再移動X、Y軸

        Parameters:
        - standby_pos: 待命位置座標 (x, y, z)
        """
        x, y, z = standby_pos

        print(f"Moving to standby position: ({x}, {y}, {z})")

        try:
            # 先移動Z軸到目標高度，使用較慢的速度
            print(f"Moving Z axis to {z}...")
            success, _ = self.send_gcode_with_timing(
                f"G01 Z{z} F300", timeout=15)
            # 短暫停留
            time.sleep(0.5)
            if not success:
                print("Error: Failed to move Z axis to standby position")
                return False

            # 再移動X、Y軸到目標位置
            print(f"Moving X,Y axes to ({x}, {y})...")
            success, _ = self.send_gcode_with_timing(
                f"G01 X{x} Y{y} F300", timeout=15)
            if not success:
                print("Error: Failed to move X,Y axes to standby position")
                return False

            print("Successfully moved to standby position")
            return True

        except Exception as e:
            print(
                f"Exception in move_to_standby_position: {e}")
            return False

    def move_to_position_only(self, x, y, z, feed_rate=400, gripper_action=None):
        """
        移動到指定位置，可選擇性執行夾爪動作

        Parameters:
        - x, y, z: 目標位置座標
        - feed_rate: 移動速度
        - gripper_action: 'open', 'close', 或 None
        """
        print(f"Moving to position: ({x}, {y}, {z}) at {feed_rate}mm/s")

        # 移動到目標位置
        success, _ = self.send_gcode_with_timing(
            f"G01 X{x} Y{y} Z{z} F{feed_rate}")
        if not success:
            print(f"Error: Failed to move to ({x}, {y}, {z})")
            return False

        # 執行夾爪動作（如果有指定）
        if gripper_action == 'open':
            return self.open_gripper()
        elif gripper_action == 'close':
            return self.close_gripper()
        else:
            pass

        return True

    def move_to_position_safe(self, x, y, z, feed_rate=400):
        """
        安全移動到指定位置（包含待命位置邏輯）

        Parameters:
        - x, y, z: 目標位置座標
        - feed_rate: 移動速度
        """
        print(
            f"Moving to position: ({x}, {y}, {z}) at {feed_rate}mm/s")

        # 移動到目標位置
        success, _ = self.send_gcode_with_timing(
            f"G01 X{x} Y{y} Z{z} F{feed_rate}")
        if not success:
            print(
                f"Error: Failed to move to ({x}, {y}, {z})")
            return False

        time.sleep(5)  # 停留秒

        # 動作完成後返回待命位置
        return self.move_to_standby_position()

    def open_gripper(self):
        """打開夾爪 (M5)"""
        print("Opening gripper...")
        success, _ = self.send_gcode_with_timing('M3')
        if success:
            print("Gripper opened successfully")
            time.sleep(0.5)  # 等待夾爪完全打開
            return True
        else:
            print("Error: Failed to open gripper")
            return False

    def close_gripper(self):
        """關閉夾爪 (M4)"""
        print("Closing gripper...")
        success, _ = self.send_gcode_with_timing('M5')
        if success:
            print("Gripper closed successfully")
            time.sleep(0.5)  # 等待夾爪完全關閉
            return True
        else:
            print("Error: Failed to close gripper")
            return False

    def get_current_position(self):
        """
        獲取當前機器人位置
        """
        if not self.is_connected:
            print("Error: Not connected to robot")
            return None

        try:
            self.ser.write(b'G92\n')  # 假設 G92 可以返回當前位置
            response = self.ser.readline().decode().strip()
            print(f"Current position response: {response}")

            # 解析回應格式，假設格式為 "X:0.00 Y:0.00 Z:0.00"
            position_data = response.split()
            position = {}
            for item in position_data:
                key, value = item.split(':')
                position[key] = float(value)

            return position

        except Exception as e:
            print(f"Error getting current position: {e}")
            return None

    def close_motor(self):
        """ 關閉馬達電源，確保安全"""
        if not self.is_connected:
            print("Error: Not connected to robot")
            return False

        try:
            success, _ = self.send_gcode_with_timing(
                f"G01 X0 Y0 Z-500 F200")
            if not success:
                print("Error: Failed to home robot")
                return False
            time.sleep(1)
            success, _ = self.send_gcode_with_timing(
                'M84')
            response = self.ser.readline().decode().strip()
            print(f"Motor close response: {response}")

            if 'Ok' in response:
                print("Motor closed successfully")
                return True
            else:
                print(f"Error closing motor: {response}")
                return False

        except Exception as e:
            print(f"Error closing motor: {e}")
            return False

    def disconnect(self):
        """斷開連線"""
        if self.ser:
            self.ser.close()
            self.is_connected = False
            print("Disconnected from Delta X")


class DeltaXROS2Node(Node):
    def __init__(self):
        super().__init__('delta_x_controller')
        self._declare_parameters()
        self._init_variables()

        port = self.get_parameter('port').get_parameter_value().string_value
        baudrate = self.get_parameter(
            'baudrate').get_parameter_value().integer_value

        # 初始化 Delta X 控制器
        self.controller = DeltaXController(port, baudrate)

        # 建立Float32MultiArray訂閱者
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'delta_x/target_array',
            self.coordinate_callback,
            10
        )

        # 建立 Point 訂閱者
        self.point_subscription = self.create_subscription(
            Point,
            'delta_x/target_point',
            self.point_callback,
            10
        )

        # 發佈執行完成狀態
        self.execution_complete_pub = self.create_publisher(
            Bool, 'delta_execution_complete', 10)

        # 建立命令佇列和處理執行緒
        self.batch_queue = Queue()  # 新增批次佇列
        self.command_queue = Queue()
        self.processing_thread = threading.Thread(
            target=self.process_commands, daemon=True)

        self.is_processing = False

        self.initialize()

    def _declare_parameters(self):
        self.declare_parameter('port', '/dev/MEGA2560')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('feed_rate', 400)
        self.declare_parameter(
            'drop_position', [-400.0, 0.0, -550.0])  # 新增丟棄位置參數

    def _init_variables(self):
        self.feed_rate = self.get_parameter(
            'feed_rate').get_parameter_value().integer_value
        self.drop_position = self.get_parameter(
            'drop_position').get_parameter_value().double_array_value

    def initialize(self):
        """初始化機器人連線"""
        self.get_logger().info("Initializing Delta X robot...")

        if not self.controller.connect():
            self.get_logger().error("Failed to connect to Delta X robot")
            return False

        # 啟動命令處理執行緒
        self.processing_thread.start()
        self.is_processing = True

        self.get_logger().info(
            "Delta X robot initialized successfully and moved to standby position")
        return True

    def point_callback(self, msg):
        """處理接收到的 Point 訊息"""
        try:
            x, y, z = msg.x, msg.y, msg.z

            self.get_logger().info(
                f"Received Point: ({x:.2f}, {y:.2f}, {z:.2f})")

            # 將座標加入命令佇列
            self.command_queue.put((x, y, z))
            self.get_logger().info(
                f"Added Point to queue: ({x:.2f}, {y:.2f}, {z:.2f})")

        except Exception as e:
            self.get_logger().error(f"Error processing Point message: {e}")

    def coordinate_callback(self, msg):
        """處理接收到的座標指令"""
        try:
            # 解析二維陣列
            coordinates = self.parse_2d_array(msg)

            if coordinates is None or len(coordinates) == 0:
                self.get_logger().warn("Received empty or invalid coordinate data")
                return

            self.get_logger().info(
                f"Received {len(coordinates)} coordinate groups")

            # 建立有效座標列表
            valid_coordinates = []
            for i, coord_group in enumerate(coordinates):
                if len(coord_group) >= 3:  # 確保至少有 x, y, z
                    x, y, z = coord_group[0], coord_group[1], coord_group[2]
                    valid_coordinates.append((x, y, z))
                    self.get_logger().info(
                        f"Added coordinate {i+1}: ({x:.2f}, {y:.2f}, {z:.2f})")
                else:
                    self.get_logger().warn(
                        f"Invalid coordinate group {i+1}: insufficient dimensions")

            # 將整個座標列表作為一個批次加入批次佇列
            if valid_coordinates:
                self.batch_queue.put(valid_coordinates)
                self.get_logger().info(
                    f"Added batch of {len(valid_coordinates)} coordinates to batch queue")

        except Exception as e:
            self.get_logger().error(
                f"Error processing coordinate message: {e}")

    def parse_2d_array(self, msg):
        """解析 Float32MultiArray 為二維陣列"""
        try:
            # 檢查維度資訊
            if len(msg.layout.dim) != 2:
                self.get_logger().error("Expected 2D array, got different dimensions")
                return None

            rows = msg.layout.dim[0].size
            cols = msg.layout.dim[1].size

            # 除錯：檢查接收到的資料
            self.get_logger().info(
                f"Received data layout: rows={rows}, cols={cols}")
            self.get_logger().info(f"Data length: {len(msg.data)}")
            self.get_logger().info(
                f"First few data values: {msg.data[:9] if len(msg.data) >= 9 else msg.data}")

            # 將一維資料重塑為二維陣列
            data_array = np.array(
                msg.data, dtype=np.float64).reshape(rows, cols)

            # 除錯：檢查重塑後的結果
            self.get_logger().info(f"Reshaped array shape: {data_array.shape}")
            if len(data_array) > 0:
                self.get_logger().info(
                    f"First coordinate group: {data_array[0]}")

            return data_array

        except Exception as e:
            self.get_logger().error(f"Error parsing 2D array: {e}")
            return None

    def process_commands(self):
        """處理命令佇列中的移動指令"""
        self.get_logger().info("Command processing thread started")

        loop_count = 0
        while self.is_processing:
            try:
                loop_count += 1

                # 每100次循環打印一次狀態（每10秒）
                if loop_count % 100 == 0:
                    command_queue_size = self.command_queue.qsize()
                    batch_queue_size = self.batch_queue.qsize()
                    self.get_logger().info(
                        f"執行緒存活 - 指令對列: {command_queue_size}, 批次對列: {batch_queue_size}")

                # 優先處理批次佇列
                if not self.batch_queue.empty():
                    self.get_logger().info("Found batch in queue, processing...")
                    coordinate_list = self.batch_queue.get(timeout=1.0)

                    self.get_logger().info(
                        f"Executing batch of {len(coordinate_list)} coordinates")

                    # 將座標列表按每3個一組分組（對應一個目標點的完整動作）
                    groups_of_3 = [coordinate_list[i:i+3]
                                   for i in range(0, len(coordinate_list), 3)]
                    self.get_logger().info(
                        f"Processing {len(groups_of_3)} target groups")

                    success_count = 0
                    for group_idx, group in enumerate(groups_of_3):
                        self.get_logger().info(
                            f"Processing target group {group_idx + 1}/{len(groups_of_3)}")

                        # 處理每組的3個座標點
                        group_success = True
                        for point_idx, coord in enumerate(group):
                            # 確保座標是可以解包的
                            if isinstance(coord, (list, tuple, np.ndarray)) and len(coord) >= 3:
                                x, y, z = float(coord[0]), float(
                                    coord[1]), float(coord[2])
                            else:
                                self.get_logger().error(
                                    f"Invalid coordinate format: {coord}")
                                group_success = False
                                break

                            # 根據點在組中的位置判斷夾爪動作
                            gripper_action = None
                            if point_idx == 0:  # 第一個點（上方，-550）
                                gripper_action = 'open'
                            elif point_idx == 1:  # 第二個點（目標位置，-625）
                                gripper_action = 'close'
                            # 第三個點（拔起，-550）不需要夾爪動作

                            self.get_logger().info(
                                f"Executing point {point_idx+1}/3 in group {group_idx+1}: ({x:.2f}, {y:.2f}, {z:.2f})")

                            success = self.controller.move_to_position_only(
                                x, y, z, self.feed_rate, gripper_action)

                            if success:
                                self.get_logger().info(
                                    f"Successfully moved to point {point_idx+1}/3")
                            else:
                                self.get_logger().error(
                                    f"Failed to move to point {point_idx+1}/3")
                                group_success = False
                                break

                        # 每組完成後移動到丟棄位置
                        if group_success:
                            self.get_logger().info(
                                f"Group {group_idx+1} completed, moving to drop position...")
                            drop_x, drop_y, drop_z = self.drop_position
                            drop_success = self.controller.move_to_position_only(
                                drop_x, drop_y, drop_z, self.feed_rate, 'open')

                            if drop_success:
                                self.get_logger().info(
                                    f"Successfully dropped object from group {group_idx+1}")
                                success_count += 1
                            else:
                                self.get_logger().error(
                                    f"Failed to drop object from group {group_idx+1}")
                        else:
                            self.get_logger().error(
                                f"Group {group_idx+1} failed, skipping drop")

                    # 批次執行完畢後返回待命位置
                    self.get_logger().info("Batch execution completed, returning to standby position...")
                    standby_success = self.controller.move_to_standby_position()

                    if standby_success:
                        self.get_logger().info(
                            f"Batch completed successfully! {success_count}/{len(coordinate_list)} points executed")
                        # 發佈執行完成信號
                        complete_msg = Bool()
                        complete_msg.data = True
                        self.execution_complete_pub.publish(complete_msg)
                    else:
                        self.get_logger().error("Failed to return to standby position after batch execution")

                    # 標記批次任務完成
                    self.batch_queue.task_done()

                # 處理單點指令佇列（Point 訊息）
                elif not self.command_queue.empty():
                    self.get_logger().info("Found single command in queue, processing...")
                    x, y, z = self.command_queue.get(timeout=1.0)

                    self.get_logger().info(
                        f"Executing single move to: ({x:.2f}, {y:.2f}, {z:.2f})")

                    # 使用 move_to_position_safe（會返回待命位置）
                    success = self.controller.move_to_position_safe(
                        x, y, z, self.feed_rate)

                    if success:
                        self.get_logger().info(
                            f"Successfully completed single move to ({x:.2f}, {y:.2f}, {z:.2f})")
                    else:
                        self.get_logger().error(
                            f"Failed to move to ({x:.2f}, {y:.2f}, {z:.2f})")

                    # 標記任務完成
                    self.command_queue.task_done()
                else:
                    time.sleep(0.1)  # 避免忙碌等待

            except Exception as e:
                self.get_logger().error(f"Error in command processing: {e}")
                time.sleep(0.1)

        self.get_logger().info("Command processing thread stopped")

    def destroy_node(self):
        """節點銷毀時的清理工作"""
        self.get_logger().info("Shutting down Delta X controller...")

        # 停止命令處理
        self.is_processing = False

        # 等待佇列清空
        if not self.command_queue.empty():
            self.get_logger().info("Waiting for remaining commands to complete...")
            self.command_queue.join()

        self.controller.close_motor()

        # 斷開機器人連線
        self.controller.disconnect()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    # 創建並運行節點
    node = DeltaXROS2Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Received shutdown signal")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
