#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R
import threading
import queue
import tf_transformations


class CoordinateTransformTester(Node):
    def __init__(self):
        super().__init__('coordinate_transform_tester')

        # 初始化變數
        self.bridge = CvBridge()
        self.current_image = None
        self.click_points = []
        self.transformed_points = []
        self.mouse_callback_set = False

        self.current_depth = None

        # TF2相關
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 相機參數 (你的參數)
        self.setup_camera_params()

        # 訂閱相機影像
        self.image_sub = self.create_subscription(
            Image, '/agri_bot/D455f/color/image_raw', self.image_callback, 10)

        self.depth_sub = self.create_subscription(
            Image, '/agri_bot/D455f/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        self.cord_pub = self.create_publisher(
            Point, 'delta_x/target_point', 10
        )

        # 建立視窗
        cv2.namedWindow('Camera Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Transform Visualization', cv2.WINDOW_AUTOSIZE)

        self.get_logger().info("座標轉換測試器已啟動")
        self.get_logger().info("點擊影像中的點來測試座標轉換")

    def setup_camera_params(self):
        """設定相機參數"""
        self.K = np.array([
            [646.4586181640625, 0.0, 642.8865966796875],
            [0.0, 645.4234619140625, 364.8355712890625],
            [0.0, 0.0, 1.0]], dtype=np.float32)

        self.D = np.array([
            -0.05416392162442207,
            0.06253961473703384,
            -2.7602534828474745e-05,
            0.0001689740311121568,
            -0.020422106608748436
        ], dtype=np.float32)

        self.rvec = np.array([
            3.1229417858397155,
            0.015100941411833508,
            -0.039669120067705635
        ], dtype=np.float32)

    def verify_camera_parameters(self):
        """驗證相機內參和畸變參數"""
        print(f"\n=== 相機參數驗證 ===")
        print(f"內參矩陣 K:")
        print(self.K)
        print(f"畸變係數 D: {self.D}")

        # 測試中心點和四個角落的去畸變
        test_points = np.array([
            [[self.cx, self.cy]],           # 光心
            [[0, 0]],                       # 左上角
            [[self.K[0, 2]*2, 0]],          # 右上角 (假設影像寬度)
            [[0, self.K[1, 2]*2]],          # 左下角 (假設影像高度)
            [[self.K[0, 2]*2, self.K[1, 2]*2]]  # 右下角
        ], dtype=np.float32)

        undistorted = cv2.undistortPoints(test_points, self.K, self.D)

        print(f"\n去畸變測試:")
        test_names = ["光心", "左上角", "右上角", "左下角", "右下角"]
        for i, (name, original, undist) in enumerate(zip(test_names, test_points, undistorted)):
            print(
                f"{name}: ({original[0][0]:.1f}, {original[0][1]:.1f}) -> ({undist[0][0]:.3f}, {undist[0][1]:.3f})")

            # 檢查是否在合理範圍內
            if abs(undist[0][0]) > 2 or abs(undist[0][1]) > 2:
                print(f"  ⚠️  警告: {name}的去畸變結果可能異常")

    def get_camera2delta_tf(self):
        """獲取相機到Delta的轉換關係"""
        # try:
        # 這裡使用你的TF2邏輯
        delta1_tf = self.tf_buffer.lookup_transform(
            'base_link', 'upper_Frameconnector1', rclpy.time.Time())
        delta2_tf = self.tf_buffer.lookup_transform(
            'base_link', 'upper_Frameconnector2', rclpy.time.Time())
        delta3_tf = self.tf_buffer.lookup_transform(
            'base_link', 'upper_Frameconnector3', rclpy.time.Time())

        # 計算理論Delta原點
        delta_center_x = (delta1_tf.transform.translation.x +
                          delta2_tf.transform.translation.x +
                          delta3_tf.transform.translation.x) / 3
        delta_center_y = (delta1_tf.transform.translation.y +
                          delta2_tf.transform.translation.y +
                          delta3_tf.transform.translation.y) / 3
        delta_center_z = np.mean([
            delta1_tf.transform.translation.z,
            delta2_tf.transform.translation.z,
            delta3_tf.transform.translation.z
        ])

        d455_tf = self.tf_buffer.lookup_transform(
            'base_link', 'D455f_color_optical_frame', rclpy.time.Time())
        camera_quat = d455_tf.transform.rotation

        rvec = self.quaternion_to_rotation_vector([camera_quat.x,
                                                   camera_quat.y,
                                                   camera_quat.z,
                                                   camera_quat.w])

        # 計算D455相對於Delta中心的位置
        d455_relative_x = d455_tf.transform.translation.x - delta_center_x
        d455_relative_y = d455_tf.transform.translation.y - delta_center_y
        d455_relative_z = d455_tf.transform.translation.z - delta_center_z

        tvec = np.array([d455_relative_x * 1000,
                         d455_relative_y * 1000,
                        d455_relative_z * 1000], dtype=np.float32)

        # self.get_logger().info(
        #     f"Delta原點與base_link的平移向量 x={delta_center_x}, y={delta_center_y}, z={delta_center_z}")

        return tvec, rvec

    def quaternion_to_rotation_vector(self, quaternion):
        """將四元數轉換為旋轉向量"""
        try:
            # 使用scipy進行轉換
            rotation = R.from_quat(quaternion)  # [x, y, z, w]

            # 獲取旋轉向量 (Rodrigues representation)
            rotvec = rotation.as_rotvec()

            return rotvec.tolist()

        except Exception as e:
            self.get_logger().error(f"旋轉轉換失敗: {e}")
            return [0.0, 0.0, 0.0]

            return rotations

    def image_callback(self, msg):
        """影像回調函式"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 設定滑鼠回調（只設定一次）
            if not self.mouse_callback_set:
                cv2.setMouseCallback('Camera Feed', self.mouse_callback)
                self.mouse_callback_set = True

            # 顯示影像和點擊點
            self.display_image()

        except Exception as e:
            self.get_logger().error(f"影像處理錯誤: {e}")

    def depth_callback(self, msg):
        """深度影像回調函式"""
        try:
            # 深度影像通常是16位元，單位為毫米
            self.current_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"深度影像處理錯誤: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        """滑鼠點擊回調"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 記錄點擊點
            self.click_points.append((x, y))
            self.get_logger().info(f"像素座標: ({x}, {y})")

            # 執行座標轉換
            transformed_point = self.transform_pixel_to_delta(x, y)
            self.transformed_points.append(transformed_point)

            # 更新視覺化
            self.update_visualization()

    def transform_pixel_to_delta(self, u, v):
        """執行像素到Delta座標的轉換"""
        # 檢查是否有深度資訊
        if self.current_depth is None:
            self.get_logger().warn("沒有深度資訊，使用固定深度650mm")
            depth_mm = 650.0
        else:
            # 從深度影像取得該像素的深度值
            depth_mm = float(self.current_depth[v, u])  # 注意：深度影像是[y, x]

            # 檢查深度值是否有效
            if depth_mm == 0 or np.isnan(depth_mm) or np.isinf(depth_mm):
                self.get_logger().warn(
                    f"像素({u}, {v})深度值無效: {depth_mm}，使用固定深度650mm")
                depth_mm = 650.0

        pixel_points = np.array([[[float(u), float(v)]]], dtype=np.float32)

        undistorted_points = cv2.undistortPoints(
            pixel_points,
            self.K,
            self.D
        )

        # 取得校正後的像素座標
        pixel_cord_corrected = [undistorted_points[0]
                                [0][0], undistorted_points[0][0][1]]

        # self.get_logger().info(
        #     f"去畸變後像素座標: {pixel_cord_corrected[0]}, {pixel_cord_corrected[1]}")

        cam_cords = [pixel_cord_corrected[0] * depth_mm,
                     pixel_cord_corrected[1] * depth_mm,
                     -depth_mm]

        cam_cords = np.array(cam_cords, dtype=np.float32)

        # self.get_logger().info(
        #     f"相機座標: {cam_cords[0]:.1f}, {cam_cords[1]:.1f}, {cam_cords[2]:.1f} mm")

        tvec, _ = self.get_camera2delta_tf()
        R, _ = cv2.Rodrigues(self.rvec)
        delta_cords = np.dot(R.T, cam_cords - tvec)

        # self.get_logger().info(
        #     f"偏移前Delta座標: {delta_cords[0]:.1f}, {delta_cords[1]:.1f} mm")

        Point_msg = Point()
        Point_msg.x = float(delta_cords[0])  # 轉換為毫米
        Point_msg.y = float(delta_cords[1])
        Point_msg.z = float(-550.0)
        # self.get_logger().info(
        #     f"Delta座標: {Point_msg.x:.1f}, {Point_msg.y:.1f}mm")

        self.get_logger().info(
            f"{u}, {v}, {pixel_cord_corrected[0]}, {pixel_cord_corrected[1]}, {cam_cords[2]}, {cam_cords[0]}, {cam_cords[1]}, {delta_cords[0]}, {delta_cords[1]}")

        self.cord_pub.publish(Point_msg)
        # measurement_msg = Point()
        # measurement_msg.x = float(delta_cords[0])
        # measurement_msg.y = float(delta_cords[1])
        # measurement_msg.z = float(cam_cords[2]+50)

        # # 發布到測量topic
        # self.measurement_pub = self.create_publisher(
        #     Point, 'delta_x/measurement_point', 10)
        # self.measurement_pub.publish(measurement_msg)

        return [Point_msg.x, Point_msg.y, Point_msg.z]

    def display_image(self):
        """顯示影像和標記點"""
        if self.current_image is None:
            return

        display_img = self.current_image.copy()

        # 繪製點擊點
        for i, point in enumerate(self.click_points):
            cv2.circle(display_img, point, 5, (0, 0, 255), -1)  # 紅色圓點
            cv2.putText(display_img, f'P{i+1}',
                        (point[0]+10, point[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 顯示說明文字
        cv2.putText(display_img, 'Left click to test coordinate transform',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, 'Press ESC to exit, C to clear points',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Camera Feed', display_img)

    def update_visualization(self):
        """更新座標轉換視覺化圖表"""
        if not self.transformed_points:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

        # 左圖：XY平面視圖
        points = np.array(self.transformed_points)
        ax1.scatter(points[:, 0], points[:, 1], c='red', s=50)
        for i, point in enumerate(points):
            ax1.annotate(f'P{i+1}', (point[0], point[1]),
                         xytext=(5, 5), textcoords='offset points')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_title('Delta Coordinate System - XY View')
        ax1.grid(True)
        ax1.axis('equal')

        # 右圖：XZ平面視圖
        ax2.scatter(points[:, 0], points[:, 2], c='blue', s=50)
        for i, point in enumerate(points):
            ax2.annotate(f'P{i+1}', (point[0], point[2]),
                         xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Z (mm)')
        ax2.set_title('Delta Coordinate System - XZ View')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('/tmp/coordinate_transform_test.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        # 載入並顯示圖表
        viz_img = cv2.imread('/tmp/coordinate_transform_test.png')
        if viz_img is not None:
            cv2.imshow('Transform Visualization', viz_img)

    def print_summary(self):
        """列印轉換結果摘要"""
        if not self.click_points:
            return

        print("\n=== 座標轉換測試結果 ===")
        print(f"{'點':<5} {'像素座標':<15} {'Delta座標 (mm)':<30}")
        print("-" * 55)

        for i, (pixel, delta) in enumerate(zip(self.click_points, self.transformed_points)):
            print(f"P{i+1:<4} ({pixel[0]:<3}, {pixel[1]:<3})      "
                  f"({delta[0]:<7.1f}, {delta[1]:<7.1f}, {delta[2]:<7.1f})")

    def run(self):
        """主執行迴圈"""
        while rclpy.ok():
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC鍵退出
                break
            elif key == ord('c') or key == ord('C'):  # C鍵清除點
                self.click_points.clear()
                self.transformed_points.clear()
                self.get_logger().info("已清除所有測試點")
            elif key == ord('s') or key == ord('S'):  # S鍵顯示摘要
                self.print_summary()

            # 處理ROS2訊息
            rclpy.spin_once(self, timeout_sec=0.01)

        # 列印最終結果
        self.print_summary()

        # 清理
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)

    try:
        tester = CoordinateTransformTester()
        tester.run()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
