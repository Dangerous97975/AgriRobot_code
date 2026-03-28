# 視覺導航性能監控使用說明

## 概述
此工具為 `vis_nav_line_detector_ridge.py` 添加了詳細的性能監控功能，可以幫助分析每個處理步驟的執行時間。

## 功能特點

### 1. 時間測量範圍
- **總處理時間**: 整個點雲處理週期的總時間
- **TF轉換時間**: 座標系轉換的時間
- **點雲處理時間**: 點雲數據讀取和轉換的時間
- **滑動窗口時間**: 滑動窗口分析的時間
- **導航生成時間**: 導航線生成的時間
- **視覺化時間**: 標記發佈和視覺化的時間

### 2. 統計輸出頻率
- **每50次處理**: 輸出各階段的平均處理時間統計
- **每100次處理**: 輸出滑動窗口的詳細統計
- **每200個切片**: 輸出單個切片的處理詳情

### 3. 統計數據包含
- 平均時間 (毫秒)
- 最小時間 (毫秒)
- 最大時間 (毫秒)
- 標準差 (毫秒)
- 處理頻率 (Hz)

## 使用方法

### 方法1: 直接運行節點
```bash
# 啟動主節點
ros2 run farm_navigation vis_nav_line_detector_ridge

# 在另一個終端啟動性能監控器
ros2 run farm_navigation performance_monitor

# 啟用視覺導航
ros2 service call /vision_navigation_control std_srvs/srv/SetBool '{data: true}'
```

### 方法2: 使用 Launch 文件
```bash
# 同時啟動節點和性能監控器
ros2 launch farm_navigation vis_nav_performance_monitor.launch.py

# 啟用視覺導航
ros2 service call /vision_navigation_control std_srvs/srv/SetBool '{data: true}'

# 停用性能監控器
ros2 launch farm_navigation vis_nav_performance_monitor.launch.py enable_performance_monitor:=false
```

## 性能分析指標

### 1. 正常性能基準
- **總處理時間**: 應 < 100ms (目標: 10Hz 處理頻率)
- **TF轉換時間**: 應 < 5ms
- **點雲處理時間**: 應 < 30ms
- **滑動窗口時間**: 應 < 50ms
- **導航生成時間**: 應 < 10ms
- **視覺化時間**: 應 < 5ms

### 2. 性能瓶頸識別
- 如果 **滑動窗口時間** 過長，考慮:
  - 減少體素化密度 (增加 `voxel_size` 參數)
  - 減少切片數量 (增加 `slice_thickness` 參數)
  - 縮小 Y 軸範圍 (`y_minimum`, `y_maximum` 參數)

- 如果 **點雲處理時間** 過長，考慮:
  - 降低點雲發佈頻率
  - 減少點雲密度

### 3. 實時監控
查看節點日誌中的性能輸出:
```bash
ros2 topic echo /rosout | grep "處理時間統計"
```

## 調試技巧

### 1. 啟用詳細日誌
```bash
ros2 run farm_navigation vis_nav_line_detector_ridge --ros-args --log-level debug
```

### 2. 查看處理頻率
```bash
ros2 topic hz /navigation_pose
```

### 3. 監控系統資源
```bash
htop  # 查看 CPU 使用率
```

## 參數調優建議

### 效能優先配置
```yaml
voxel_size: 0.02        # 較大的體素尺寸
slice_thickness: 0.08   # 較厚的切片
window_step: 0.05       # 較大的窗口步進
```

### 精度優先配置
```yaml
voxel_size: 0.005       # 較小的體素尺寸
slice_thickness: 0.03   # 較薄的切片
window_step: 0.01       # 較小的窗口步進
```

## 故障排除

### 1. 性能過慢
- 檢查點雲數據量是否過大
- 調整參數以降低計算複雜度
- 確認硬體資源充足

### 2. 統計數據異常
- 重啟節點清除統計緩存
- 檢查是否有其他高負載程序

### 3. 無性能數據輸出
- 確認視覺導航已啟用
- 檢查點雲數據是否正常輸入
- 查看節點是否有錯誤日誌
