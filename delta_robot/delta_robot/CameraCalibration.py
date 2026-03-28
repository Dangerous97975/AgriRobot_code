#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os


class ExtrinsicCalibrationNode(Node):
    def __init__(self):
        super().__init__('extrinsic_calibration_node')

        # 初始化 CV Bridge
        self.bridge = CvBridge()

        # 棋盤格參數
        self._Chessboard = (9, 6)  # 棋盤格的長寬數量
        self._Chessboard_squareSize = 24  # 棋盤格小格的長度 mm

        # 相機參數變數
        self.camera_matrix = None
        self.dist_coeffs = None
        self.current_image = None
        self.camera_info_received = False

        # 訂閱器
        self.image_sub = self.create_subscription(
            Image, '/agri_bot/D455f/color/image_raw', self.image_callback, 10)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/agri_bot/D455f/color/camera_info', self.camera_info_callback, 10)

        self.get_logger().info('外參標定節點已啟動')
        self.get_logger().info('請確保棋盤格在視野中，然後按 Enter 開始標定')

    def camera_info_callback(self, msg):
        """接收相機內參與畸變參數"""
        if not self.camera_info_received:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.camera_info_received = True
            self.get_logger().info('已接收相機內參與畸變參數')
            self.get_logger().info(f'相機內參矩陣: \n{self.camera_matrix}')

    def image_callback(self, msg):
        """接收圖像數據"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'圖像轉換失敗: {e}')

    def text_print(self, frame, text):
        """在畫面中顯示文字"""
        word_color = (0, 255, 255)  # 字的顏色
        position = (0, 20)          # 字的位置
        cv2.rectangle(frame, (0, 0), (655, 25), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, text, position,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, word_color, 1)

    def standard_catch(self):
        """拍攝標定用棋盤格"""
        if self.current_image is None:
            self.get_logger().warn('尚未接收到圖像數據')
            return None

        while True:
            if self.current_image is not None:
                frame = self.current_image.copy()
                text = 'Place the Calibration board and press Enter'
                self.text_print(frame, text)
                cv2.imshow("Standard", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter 鍵
                    standard_image = self.current_image.copy()
                    # cv2.imwrite("Standard.png", standard_image)
                    # self.get_logger().info('標定圖像已儲存為 Standard.png')
                    cv2.destroyAllWindows()
                    return standard_image
                elif key == 27:  # ESC 鍵
                    cv2.destroyAllWindows()
                    return None

            # 讓 ROS2 回調函數有機會執行
            rclpy.spin_once(self, timeout_sec=0.01)

    def calculate_extrinsics(self):
        """計算外參 - 改用手動標記四個關鍵點"""
        if not self.camera_info_received:
            self.get_logger().error('尚未接收到相機內參，無法進行標定')
            return False

        # 拍攝標定圖像
        std_img = self.standard_catch()
        if std_img is None:
            self.get_logger().info('標定已取消')
            return False

        # 轉換為灰階
        gray = cv2.cvtColor(std_img, cv2.COLOR_BGR2GRAY)

        # 尋找棋盤格角點
        ret, corners = cv2.findChessboardCorners(gray, self._Chessboard, None)

        if ret:
            self.get_logger().info('成功找到棋盤格角點')

            # 精細化角點位置
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)

            # 選擇四個關鍵角點進行標定
            key_corners = self.select_key_corners(corners, std_img)

            if key_corners is not None:
                # 手動輸入這四個點的實際Delta座標
                world_points = self.input_delta_coordinates()

                if world_points is not None:
                    # 使用四個關鍵點計算外參
                    success, rvecs, tvecs = cv2.solvePnP(
                        world_points, key_corners, self.camera_matrix, self.dist_coeffs)

                    if success:
                        # 驗證標定結果
                        if self.verify_calibration_result(key_corners, world_points, rvecs, tvecs):
                            # 儲存外參到 YAML 檔案
                            self.save_extrinsics_to_yaml(rvecs, tvecs)
                            self.get_logger().info('外參標定完成')
                            return True
                        else:
                            self.get_logger().error('標定結果驗證失敗，請重新標定')
                            return False
                    else:
                        self.get_logger().error('solvePnP 計算失敗')
                        return False
                else:
                    self.get_logger().info('座標輸入已取消')
                    return False
            else:
                self.get_logger().info('角點選擇已取消')
                return False
        else:
            self.get_logger().error('無法找到棋盤格角點，請重新調整棋盤格位置')
            return False

    def select_key_corners(self, corners, img):
        """選擇四個關鍵角點"""
        # 選擇棋盤格的四個角落點
        # corners 是按照棋盤格從左到右、從上到下的順序排列
        # 注意：棋盤格是 (cols, rows)
        rows, cols = self._Chessboard[1], self._Chessboard[0]

        key_indices = [
            0,                    # 左上角
            cols - 1,            # 右上角
            (rows - 1) * cols,   # 左下角
            rows * cols - 1      # 右下角
        ]

        key_corners = np.array([corners[i][0]
                                for i in key_indices], dtype=np.float32)

        # 在圖像上標記這四個點
        display_img = img.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                  (255, 255, 0)]  # 紅、綠、藍、黃
        labels = ['Point 1 (Top-Left)', 'Point 2 (Top-Right)',
                  'Point 3 (Bottom-Left)', 'Point 4 (Bottom-Right)']

        for i, (corner, color, label) in enumerate(zip(key_corners, colors, labels)):
            cv2.circle(
                display_img, (int(corner[0]), int(corner[1])), 8, color, -1)
            cv2.putText(display_img, f'{i+1}',
                        (int(corner[0])+15, int(corner[1])-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 顯示說明
        y_offset = 30
        for i, label in enumerate(labels):
            cv2.putText(display_img, label, (10, y_offset + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

        cv2.putText(display_img, 'Press ENTER to confirm, ESC to cancel',
                    (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Select Key Corners', display_img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 13:  # Enter鍵
            return key_corners
        else:
            return None

    def input_delta_coordinates(self):
        """輸入四個點的實際Delta座標"""
        print("\n=== 請用Delta手臂測量四個標記點的實際座標 ===")
        print("點1: 左上角 (紅色)")
        print("點2: 右上角 (綠色)")
        print("點3: 左下角 (藍色)")
        print("點4: 右下角 (黃色)")
        print("\n請依序輸入四個點的Delta座標 (單位: mm)")

        world_points = []

        for i in range(4):
            try:
                print(f"\n點{i+1}的座標:")
                x = float(input(f"  X座標 (mm): "))
                y = float(input(f"  Y座標 (mm): "))
                z = float(input(f"  Z座標 (mm, 通常為0): "))

                world_points.append([x, y, z])
                print(f"  已記錄: ({x:.1f}, {y:.1f}, {z:.1f})")

            except ValueError:
                print("輸入格式錯誤，標定已取消")
                return None
            except KeyboardInterrupt:
                print("\n標定已取消")
                return None

        return np.array(world_points, dtype=np.float32)

    def verify_calibration_result(self, image_points, world_points, rvecs, tvecs):
        """驗證標定結果"""
        # 將世界座標投影回像素座標
        projected_points, _ = cv2.projectPoints(
            world_points, rvecs, tvecs, self.camera_matrix, self.dist_coeffs)

        # 計算重投影誤差
        total_error = 0
        print(f"\n=== 標定結果驗證 ===")
        print(f"{'點':<5} {'原始像素':<15} {'投影像素':<15} {'誤差(像素)':<10}")
        print("-" * 50)

        for i in range(len(image_points)):
            error = np.linalg.norm(image_points[i] - projected_points[i][0])
            total_error += error
            print(f"點{i+1:<4} ({image_points[i][0]:.1f}, {image_points[i][1]:.1f})   "
                  f"({projected_points[i][0][0]:.1f}, {projected_points[i][0][1]:.1f})   "
                  f"{error:.2f}")

        mean_error = total_error / len(image_points)
        print(f"\n平均重投影誤差: {mean_error:.2f} 像素")

        # 如果平均誤差小於2像素，認為標定成功
        if mean_error < 2.0:
            print("✓ 標定結果良好")
            return True
        else:
            print("✗ 標定誤差過大，建議重新標定")
            return False

    def save_extrinsics_to_yaml(self, rvecs, tvecs):
        """儲存外參到 YAML 檔案"""
        # 將旋轉向量轉換為旋轉矩陣
        rotation_matrix, _ = cv2.Rodrigues(rvecs)

        extrinsic_data = {
            'extrinsic_calibration': {
                'rotation_vector': rvecs.flatten().tolist(),
                'translation_vector': tvecs.flatten().tolist(),
                'rotation_matrix': rotation_matrix.tolist(),
                'chessboard_size': list(self._Chessboard),
                'square_size_mm': self._Chessboard_squareSize
            }
        }

        yaml_filename = 'camera_extrinsics.yaml'
        with open(yaml_filename, 'w') as yaml_file:
            yaml.dump(extrinsic_data, yaml_file, default_flow_style=False)

        self.get_logger().info(f'外參已儲存至 {yaml_filename}')
        self.get_logger().info(f'旋轉向量: {rvecs.flatten()}')
        self.get_logger().info(f'平移向量: {tvecs.flatten()}')


def main(args=None):
    rclpy.init(args=args)

    calibration_node = ExtrinsicCalibrationNode()

    try:
        # 等待相機資訊
        while not calibration_node.camera_info_received:
            rclpy.spin_once(calibration_node, timeout_sec=1.0)
            calibration_node.get_logger().info('等待相機內參資訊...')

        # 開始標定
        calibration_node.calculate_extrinsics()

    except KeyboardInterrupt:
        calibration_node.get_logger().info('程式已中斷')
    finally:
        calibration_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
