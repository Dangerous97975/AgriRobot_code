import numpy as np
from scipy.ndimage import gaussian_filter1d


class FurrowKalmanFilter:
    """畦溝位置的卡爾曼濾波器"""

    def __init__(self, initial_position=None):
        self.is_initialized = False

        # 狀態向量: [position, velocity]
        self.state = np.array([0.0, 0.0])

        # 狀態協方差矩陣
        self.P = np.array([[1.0, 0.0],
                          [0.0, 1.0]])

        # 狀態轉移矩陣 (假設等速模型)
        self.F = np.array([[1.0, 1.0],
                          [0.0, 1.0]])

        # 過程噪聲協方差
        self.Q = np.array([[0.01, 0.0],
                          [0.0, 0.05]])

        # 觀測矩陣
        self.H = np.array([[1.0, 0.0]])

        if initial_position is not None:
            self.initialize(initial_position)

    def initialize(self, position):
        """初始化濾波器"""
        self.state[0] = position
        self.state[1] = 0.0  # 初始速度為0
        self.is_initialized = True

    def predict(self):
        """預測步驟"""
        if not self.is_initialized:
            return None

        # 預測狀態
        self.state = self.F @ self.state

        # 預測協方差
        self.P = self.F @ self.P @ self.F.T + self.Q

        # 確保返回標量位置值
        position = self.state[0]
        if isinstance(position, np.ndarray):
            return float(position.item()) if position.size == 1 else float(position[0])
        else:
            return float(position)

    def update(self, measurements, measurement_covariances):
        """更新步驟 - 支援多個測量值"""
        if not self.is_initialized:
            if len(measurements) > 0:
                self.initialize(measurements[0])
            # 確保返回標量
            if self.is_initialized:
                return float(self.state[0]) if isinstance(self.state[0], np.ndarray) else self.state[0]
            else:
                return None

        if len(measurements) == 0:
            # 確保返回標量
            return float(self.state[0]) if isinstance(self.state[0], np.ndarray) else self.state[0]

        # 處理多個測量值
        for i, (measurement, R) in enumerate(zip(measurements, measurement_covariances)):
            if measurement is None:
                continue

            # 計算卡爾曼增益
            S = self.H @ self.P @ self.H.T + R
            K = self.P @ self.H.T / S

            # 更新狀態
            y = measurement - self.H @ self.state  # 創新
            self.state = self.state + K * y

            # 更新協方差
            I_KH = np.eye(2) - K @ self.H
            self.P = I_KH @ self.P

        # 確保返回標量位置值
        position = self.state[0]
        if isinstance(position, np.ndarray):
            return float(position.item()) if position.size == 1 else float(position[0])
        else:
            return float(position)

    def get_uncertainty(self):
        """獲取位置不確定性"""
        if not self.is_initialized:
            return float('inf')

        uncertainty = np.sqrt(self.P[0, 0])
        if isinstance(uncertainty, np.ndarray):
            return float(uncertainty.item()) if uncertainty.size == 1 else float(uncertainty[0])
        else:
            return float(uncertainty)


class FusedFurrowDetector:
    """融合多種方法的畦溝檢測器"""

    def __init__(self):
        self.left_filter = FurrowKalmanFilter()
        self.right_filter = FurrowKalmanFilter()

    def reset_filters(self):
        """重置濾波器"""
        self.left_filter = FurrowKalmanFilter()
        self.right_filter = FurrowKalmanFilter()

    def fuse_measurements(self, hist_left, hist_right, ridge_left, ridge_right,
                          hist_confidence_left, hist_confidence_right,
                          ridge_confidence_left, ridge_confidence_right,
                          logger=None):
        """使用卡爾曼濾波器融合測量值"""

        # 預測步驟
        pred_left = self.left_filter.predict()
        pred_right = self.right_filter.predict()

        # 準備左側測量值
        left_measurements = []
        left_covariances = []

        if hist_left is not None and hist_confidence_left > 20:
            left_measurements.append(hist_left)
            # 信心度越高，測量噪聲越小
            left_covariances.append(
                0.05 * (100 - hist_confidence_left) / 100 + 0.01)

        if ridge_left is not None and ridge_confidence_left > 20:
            left_measurements.append(ridge_left)
            left_covariances.append(
                0.05 * (100 - ridge_confidence_left) / 100 + 0.01)

        # 準備右側測量值
        right_measurements = []
        right_covariances = []

        if hist_right is not None and hist_confidence_right > 20:
            right_measurements.append(hist_right)
            right_covariances.append(
                0.05 * (100 - hist_confidence_right) / 100 + 0.01)

        if ridge_right is not None and ridge_confidence_right > 20:
            right_measurements.append(ridge_right)
            right_covariances.append(
                0.05 * (100 - ridge_confidence_right) / 100 + 0.01)

        # 更新步驟
        fused_left = self.left_filter.update(
            left_measurements, left_covariances)
        fused_right = self.right_filter.update(
            right_measurements, right_covariances)

        # 記錄融合過程 - 修復格式化問題
        if logger is not None:
            # 安全地轉換為 Python 標量
            if fused_left is not None:
                if isinstance(fused_left, np.ndarray):
                    left_val = float(fused_left.item()) if fused_left.size == 1 else float(
                        fused_left[0])
                else:
                    left_val = float(fused_left)
            else:
                left_val = None

            if fused_right is not None:
                if isinstance(fused_right, np.ndarray):
                    right_val = float(fused_right.item()) if fused_right.size == 1 else float(
                        fused_right[0])
                else:
                    right_val = float(fused_right)
            else:
                right_val = None

            # logger.info(
            #     f'卡爾曼融合: 左側測量值={len(left_measurements)}, '
            #     f'右側測量值={len(right_measurements)}, '
            #     f'融合結果: left={left_val:.3f if left_val is not None else "None"}, '
            #     f'right={right_val:.3f if right_val is not None else "None"}'
            # )

        return fused_left, fused_right

    def get_uncertainties(self):
        """獲取當前位置的不確定性"""
        left_uncertainty = self.left_filter.get_uncertainty()
        right_uncertainty = self.right_filter.get_uncertainty()
        return left_uncertainty, right_uncertainty


def calculate_method_confidence(result, method_type, heights, centers,
                                ridge_reference_height=None):
    """計算單個方法的信心度"""

    if result is None:
        return 0.0

    confidence = 50.0  # 基礎分數

    # 檢查結果是否在有效範圍內
    if abs(result) > 1.0:  # 超出1米範圍
        return 0.0

    try:
        if method_type == 'histogram':
            # 直方圖方法：檢查是否為真正的谷值
            result_idx = np.argmin(np.abs(centers - result))
            if 0 < result_idx < len(heights) - 1:
                left_height = heights[result_idx - 1]
                right_height = heights[result_idx + 1]
                current_height = heights[result_idx]

                # 如果確实是谷值，增加信心度
                if current_height < left_height and current_height < right_height:
                    valley_depth = min(left_height - current_height,
                                       right_height - current_height)
                    confidence += min(30.0, valley_depth * 1000)  # 谷深度獎勵
                else:
                    confidence -= 20.0  # 不是谷值懲罰

        elif method_type == 'ridge':
            # 土畦方法：檢查高度差是否合理
            if ridge_reference_height is not None:
                result_idx = np.argmin(np.abs(centers - result))
                if 0 <= result_idx < len(heights):
                    furrow_height = heights[result_idx]
                    height_diff = ridge_reference_height - furrow_height

                    # 高度差在合理範圍內增加信心度
                    if 0.08 <= height_diff <= 0.20:
                        confidence += 30.0
                    elif 0.05 <= height_diff <= 0.25:
                        confidence += 15.0
                    elif 0.03 <= height_diff <= 0.30:
                        confidence += 5.0
                    else:
                        confidence -= 10.0  # 高度差不合理懲罰

        # 額外檢查：距離中心的合理性
        distance_from_center = abs(result)
        if 0.2 <= distance_from_center <= 0.8:
            confidence += 10.0  # 在合理範圍內
        elif distance_from_center < 0.2:
            confidence -= 15.0  # 太靠近中心
        elif distance_from_center > 0.8:
            confidence -= 20.0  # 太遠離中心

    except (IndexError, ValueError):
        confidence = 0.0

    return max(0.0, min(100.0, confidence))


def fuse_furrow_detection_with_kalman(histogram_info, ridge_info, fused_detector,
                                      histogram_centers, histogram_heights,
                                      logger=None, slice_id=0):
    """使用卡爾曼濾波器融合畦溝檢測結果"""

    # 計算土畦參考高度
    center_mask = (histogram_centers >= -0.4) & (histogram_centers <= 0.4)
    if np.any(center_mask):
        ridge_reference_height = np.median(histogram_heights[center_mask])
    else:
        ridge_reference_height = np.max(histogram_heights) if len(
            histogram_heights) > 0 else None

    # 計算各方法的信心度
    hist_conf_left = calculate_method_confidence(
        histogram_info['left_furrow'], 'histogram',
        histogram_info['heights_smooth'], histogram_centers)

    hist_conf_right = calculate_method_confidence(
        histogram_info['right_furrow'], 'histogram',
        histogram_info['heights_smooth'], histogram_centers)

    ridge_conf_left = calculate_method_confidence(
        ridge_info['left_furrow'], 'ridge',
        ridge_info['heights_smooth'], histogram_centers,
        ridge_reference_height)

    ridge_conf_right = calculate_method_confidence(
        ridge_info['right_furrow'], 'ridge',
        ridge_info['heights_smooth'], histogram_centers,
        ridge_reference_height)

    # 使用卡爾曼濾波器融合
    fused_left, fused_right = fused_detector.fuse_measurements(
        histogram_info['left_furrow'], histogram_info['right_furrow'],
        ridge_info['left_furrow'], ridge_info['right_furrow'],
        hist_conf_left, hist_conf_right,
        ridge_conf_left, ridge_conf_right,
        logger)

    # 記錄詳細信息
    if logger is not None:
        # logger.info(
        #     f'切片 {slice_id} 信心度評估: '
        #     f'直方圖(左={hist_conf_left:.1f}, 右={hist_conf_right:.1f}), '
        #     f'土畦(左={ridge_conf_left:.1f}, 右={ridge_conf_right:.1f})'
        # )

        uncertainties = fused_detector.get_uncertainties()
        # 修復不確定性值的格式化
        left_uncertainty = float(uncertainties[0]) if uncertainties[0] != float(
            'inf') else float('inf')
        right_uncertainty = float(uncertainties[1]) if uncertainties[1] != float(
            'inf') else float('inf')

        # logger.info(
        #     f'切片 {slice_id} 不確定性: '
        #     f'左={left_uncertainty:.3f if left_uncertainty != float("inf") else "inf"}, '
        #     f'右={right_uncertainty:.3f if right_uncertainty != float("inf") else "inf"}'
        # )

    # 為可視化準備融合後的資訊
    fused_furrow_info = {
        'left_furrow': fused_left,
        'right_furrow': fused_right,
        'valleys': get_fused_valleys(fused_left, fused_right, histogram_centers),
        'heights_smooth': ridge_info['heights_smooth'],
        'histogram_info': histogram_info,
        'ridge_info': ridge_info,
        'confidences': {
            'hist_left': hist_conf_left,
            'hist_right': hist_conf_right,
            'ridge_left': ridge_conf_left,
            'ridge_right': ridge_conf_right
        }
    }

    return fused_left, fused_right, fused_furrow_info


def get_fused_valleys(left_furrow, right_furrow, centers):
    """獲取融合結果的valleys索引用於可視化"""
    valleys = []

    if left_furrow is not None:
        # 提取位置值（如果是陣列的話）
        if isinstance(left_furrow, np.ndarray):
            left_pos = left_furrow[0] if left_furrow.size > 0 else left_furrow
        else:
            left_pos = left_furrow

        left_idx = np.where(np.abs(centers - left_pos) < 0.01)[0]
        if len(left_idx) > 0:
            valleys.append(left_idx[0])

    if right_furrow is not None:
        # 提取位置值（如果是陣列的話）
        if isinstance(right_furrow, np.ndarray):
            right_pos = right_furrow[0] if right_furrow.size > 0 else right_furrow
        else:
            right_pos = right_furrow

        right_idx = np.where(np.abs(centers - right_pos) < 0.01)[0]
        if len(right_idx) > 0:
            valleys.append(right_idx[0])

    return np.array(valleys)
