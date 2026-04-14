/**
 * ===================================================================================
 * 專案名稱: 4WIS 底盤控制韌體 - 狀態機與轉向動力解耦版 
 * 核心功能: 
 * 1. 四輪獨立轉向與動力驅動 (Independent Steering & Driving)
 * 2. 支援 ROS2 (cmd_vel) 與 藍牙手把雙模式
 * 3. 具備阿克曼、螃蟹、原地旋轉等多種運動模式
 * 4. 轉向最佳化邏輯：確保轉向角在 ±90 度內，必要時自動反轉輪胎轉向以求最快反應。
 * ===================================================================================
 */

#include <Arduino.h>
#include <micro_ros_platformio.h>
#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <geometry_msgs/msg/twist.h>
#include <FastAccelStepper.h> 
#include <ESP32Servo.h>
#include <Bluepad32.h>
#include <math.h>

// ==========================================
// 1. 系統參數設定
// ==========================================
#define ENABLE_MICRO_ROS true      // 是否啟用 Micro-ROS 連線
#define BINDING_MODE_ENABLED false  

// 操作安全性參數
#define JOYSTICK_DEADZONE 80       // 搖桿死區（防止抖動造成的誤觸發）
#define INVERT_STEERING true       // 是否反轉轉向方向
#define MANUAL_SPEED_LIMIT 0.2     // 手把模式下的最高速度限制 (預設僅 30% 功率以維護安全)

// 步進馬達 (轉向) 物理參數
#define MAX_STEPPER_SPEED 5000     // 步進馬達最高頻率 (Hz)
#define STEPPER_ACCEL 6000         // 步進馬達加速度 (steps/s^2)

// 嚴格物理限位與減速比計算
#define STEER_LIMIT_DEG 90.0       // 物理限制最大轉向角 ±90 度
#define STEP_ANGLE 1.8             // 步進馬達步距角
#define MICRO_STEP 16.0            // 驅動器細分數
#define STEP_GEAR_RATIO 20.0       // 轉向齒輪箱減速比 (20:1)
// 計算每度需要的脈衝數：(360/1.8) * 16 * 20 / 360 = 177.77
#define PULSES_PER_DEG ((360.0/STEP_ANGLE) * MICRO_STEP * STEP_GEAR_RATIO / 360.0)

// 腳位定義：LF(左前), LR(左後), RF(右前), RR(右後)
#define DIR_LF 12 
#define STEP_LF 13
#define BLDC_LF_PIN 18 

#define DIR_LR 25
#define STEP_LR 26
#define BLDC_LR_PIN 19

#define DIR_RF 27
#define STEP_RF 14
#define BLDC_RF_PIN 17

#define DIR_RR 22
#define STEP_RR 23
#define BLDC_RR_PIN 16

// BLDC 電調 (動力) 參數 (使用 PPM 訊號控制)
#define PPM_MIN 1000               // 最小脈寬 (全後退)
#define PPM_MAX 2000               // 最大脈寬 (全前進)
#define PPM_MID 1500               // 中位數 (停止)
#define PPM_DEAD 80                // 電調死區 (避免靜止時微動)

// 底盤幾何參數 (公尺)
#define TRACK_W 0.560              // 輪距 (左右輪距離)
#define WHEELBASE_L 0.660          // 軸距 (前後輪距離)
#define MAX_VEL 1.0                // 系統標定最大速度 (m/s)

// ==========================================
// 2. 狀態機與運動學類別
// ==========================================

// 運動模式狀態機
enum DriveMode {
    MODE_ACKERMANN,   // 阿克曼轉向 (一般車輛行進)
    MODE_CRAB_LEFT,   // 左橫移 (螃蟹模式)
    MODE_CRAB_RIGHT,  // 右橫移 (螃蟹模式)
    MODE_SPIN_CCW,    // 原地逆時針旋轉 (坦克轉向)
    MODE_SPIN_CW,     // 原地順時針旋轉 (坦克轉向)
    MODE_RESET        // 重置歸零
};

// 儲存單一輪子的狀態
struct WheelState {
    float speed;      // 目標速度 (-1.0 ~ 1.0 或 m/s)
    float angle;      // 目標角度 (度)
    bool reverse;     // 是否需要反轉馬達 (用於最佳化路徑)
};

// 儲存 ROS 接收到的指令
struct ROSCommand {
    float vx;    // 線速度 X
    float vy;    // 線速度 Y
    float omega; // 角速度 (旋轉)
};

class Kinematics {
private:
    float x_offset[4]; // 輪胎座標 X
    float y_offset[4]; // 輪胎座標 Y

public:
    Kinematics() {
        // 定義四個輪子相對於底盤中心的幾何位置
        x_offset[0] = WHEELBASE_L/2.0;  y_offset[0] = TRACK_W/2.0;  // LF
        x_offset[1] = WHEELBASE_L/2.0;  y_offset[1] = -TRACK_W/2.0; // RF
        x_offset[2] = -WHEELBASE_L/2.0; y_offset[2] = TRACK_W/2.0;  // LR
        x_offset[3] = -WHEELBASE_L/2.0; y_offset[3] = -TRACK_W/2.0; // RR
    }

    /**
     * 全向運動學計算 (Swerve/4WIS Drive)
     * 依據 vx, vy, omega 計算四輪各自的向量
     */
    void compute(float vx, float vy, float omega, WheelState out[4]) {
        if (fabs(vx) < 0.001 && fabs(vy) < 0.001 && fabs(omega) < 0.001) {
             for(int i=0; i<4; i++) { out[i].speed = 0; out[i].angle = 0; out[i].reverse = false; }
             return;
        }
        for(int i=0; i<4; i++) {
            float wx = vx - omega * y_offset[i];
            float wy = vy + omega * x_offset[i];
            out[i].speed = sqrt(wx*wx + wy*wy);
            out[i].angle = atan2(wy, wx) * 180.0 / PI; 
            out[i].reverse = false;
        }
    }

    /**
     * 阿克曼轉向計算 (Ackermann Steering)
     * 主要用於手把前進後退 + 轉向
     */
    void computeAckermann(float throttle, float steer_input, WheelState out[4]) {
        float clamped_steer = constrain(steer_input, -45.0, 45.0);
        if (abs(clamped_steer) < 1.0) {
            for(int i=0; i<4; i++) { out[i].speed = throttle; out[i].angle = 0; out[i].reverse = false; }
            return;
        }
        float tan_delta = tan(clamped_steer * PI / 180.0);
        if (fabs(tan_delta) < 0.0001) tan_delta = 0.0001;
        float R = WHEELBASE_L / tan_delta; // 轉彎半徑

        // 依據幾何計算內外輪角度差
        out[0].angle = atan(WHEELBASE_L / (R - TRACK_W/2.0)) * 180.0 / PI; // 左前
        out[1].angle = atan(WHEELBASE_L / (R + TRACK_W/2.0)) * 180.0 / PI; // 右前
        out[2].angle = 0; out[3].angle = 0; // 後輪不動 (標準阿克曼)
        
        out[2].speed = throttle; out[3].speed = throttle;
        out[0].speed = throttle / cos(out[0].angle * PI / 180.0);
        out[1].speed = throttle / cos(out[1].angle * PI / 180.0);
        for(int i=0; i<4; i++) out[i].reverse = false;
    }

    /**
     * 轉向角度最佳化 (核心邏輯)
     * 為了避免線材纏繞並縮短反應時間，轉向角度限制在 ±90 度。
     * 若目標角度超過 90 度，則將輪胎轉向「目標角度-180度」，並將動力馬達反轉。
     */
    static float optimizeAngleStrict90(float current_pos, float target_deg, bool &reverse) {
        while (target_deg > 180.0) target_deg -= 360.0;
        while (target_deg <= -180.0) target_deg += 360.0;

        if (target_deg > 90.0) { target_deg -= 180.0; reverse = true; } 
        else if (target_deg < -90.0) { target_deg += 180.0; reverse = true; } 
        else { reverse = false; }

        return constrain(target_deg, -STEER_LIMIT_DEG, STEER_LIMIT_DEG);
    }
};

// ==========================================
// 3. 全域物件與變數
// ==========================================
Kinematics kinematics; 
WheelState target_states[4]; 
FastAccelStepperEngine engine;
FastAccelStepper* steppers[4];
Servo bldcs[4];
ControllerPtr myController = nullptr;

DriveMode current_bt_mode = MODE_ACKERMANN; 
ROSCommand ros_cmd = {0, 0, 0};    
bool is_system_locked = false;  // 系統緊急鎖定狀態

unsigned long last_ros_time = 0, last_bt_time = 0;
const unsigned long CMD_TIMEOUT = 500; // 指令逾時保護 (500ms)

// Micro-ROS 相關物件
rcl_subscription_t subscriber;
geometry_msgs__msg__Twist msg;
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
bool micro_ros_online = false;

// ROS 訂閱回呼函數
void subscription_callback(const void * msgin) {
    const geometry_msgs__msg__Twist * m = (const geometry_msgs__msg__Twist *)msgin;
    ros_cmd.vx = m->linear.x; 
    ros_cmd.vy = m->linear.y; 
    ros_cmd.omega = m->angular.z;
    last_ros_time = millis();
}

// 藍牙手把事件
void onConnectedController(ControllerPtr ctl) { if (myController == nullptr) myController = ctl; }
void onDisconnectedController(ControllerPtr ctl) { if (myController == ctl) myController = nullptr; }

// ==========================================
// 4. 初始化與主迴圈
// ==========================================

void setup() {
    Serial.begin(115200);
    engine.init();

    // 1. 初始化轉向步進馬達
    int stepPins[4] = {STEP_LF, STEP_RF, STEP_LR, STEP_RR};
    int dirPins[4] = {DIR_LF, DIR_RF, DIR_LR, DIR_RR};
    for(int i=0; i<4; i++) {
        steppers[i] = engine.stepperConnectToPin(stepPins[i]);
        if (steppers[i]) {
            steppers[i]->setDirectionPin(dirPins[i]);
            steppers[i]->setAutoEnable(true);
            steppers[i]->setSpeedInHz(MAX_STEPPER_SPEED); 
            steppers[i]->setAcceleration(STEPPER_ACCEL);
        }
    }

    // 2. 初始化動力 BLDC 電調
    int bldcPins[4] = {BLDC_LF_PIN, BLDC_RF_PIN, BLDC_LR_PIN, BLDC_RR_PIN};
    for(int i=0; i<4; i++) { 
        bldcs[i].attach(bldcPins[i], PPM_MIN, PPM_MAX); 
        bldcs[i].writeMicroseconds(PPM_MID); // 確保開機時停止
    }
    
    // 3. 藍牙與 Micro-ROS 初始化
    BP32.setup(&onConnectedController, &onDisconnectedController);
    if(ENABLE_MICRO_ROS) {
        set_microros_serial_transports(Serial);
        allocator = rcl_get_default_allocator();
        if (rclc_support_init(&support, 0, NULL, &allocator) == RCL_RET_OK) {
            rclc_node_init_default(&node, "esp32_4wis", "", &support);
            rclc_subscription_init_default(&subscriber, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist), "cmd_vel");
            rclc_executor_init(&executor, &support.context, 1, &allocator);
            rclc_executor_add_subscription(&executor, &subscriber, &msg, &subscription_callback, ON_NEW_DATA);
            micro_ros_online = true;
        }
    }
}

void loop() {
    unsigned long current_time = millis();
    
    // 處理 ROS 通訊
    if (ENABLE_MICRO_ROS && micro_ros_online) rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1));
    
    // 更新手把狀態
    BP32.update(); 

    float actual_throttle = 0, steer_rx = 0;

    // --- A. 手把輸入解析 ---
    if (myController && myController->isConnected()) {
        int ly = myController->axisY(), rx = myController->axisRX();
        bool btn_lb = myController->l1(), btn_rb = myController->r1();
        bool btn_lt = myController->brake() > 100, btn_rt = myController->throttle() > 100;
        bool btn_x = myController->x(); // 鎖定鍵
        bool btn_a = myController->a(); // 解鎖鍵

        // 安全鎖定邏輯
        if (btn_x) {
            is_system_locked = true;
            current_bt_mode = MODE_RESET;
            Serial.println("!!! SYSTEM LOCKED (Forced Stop) !!!");
        } else if (btn_a) {
            is_system_locked = false;
            Serial.println(">>> SYSTEM UNLOCKED (Normal Mode)");
        }

        if (!is_system_locked) {
            // 解析油門與轉向
            actual_throttle = (abs(ly) > JOYSTICK_DEADZONE) ? -(ly / 512.0) * MAX_VEL * MANUAL_SPEED_LIMIT : 0;
            steer_rx = -(rx / 512.0) * 45.0;

            // 模式切換邏輯
            if (btn_lb) current_bt_mode = MODE_CRAB_LEFT;
            else if (btn_rb) current_bt_mode = MODE_CRAB_RIGHT;
            else if (btn_lt) current_bt_mode = MODE_SPIN_CCW;
            else if (btn_rt) current_bt_mode = MODE_SPIN_CW;
            else if (abs(rx) > JOYSTICK_DEADZONE) current_bt_mode = MODE_ACKERMANN;

            // 更新最後通訊時間
            if (btn_lb || btn_rb || btn_lt || btn_rt || abs(rx) > JOYSTICK_DEADZONE || abs(ly) > JOYSTICK_DEADZONE) {
                last_bt_time = current_time;
            }
        }
    }

    // --- B. 運動學計算 (每 20ms 執行一次，50Hz) ---
    static unsigned long last_kin = 0;
    if (current_time - last_kin >= 20) {
        
        if (is_system_locked) {
            // 鎖定狀態：強制角度歸零、速度歸零
            for(int i=0; i<4; i++) { target_states[i].angle = 0; target_states[i].speed = 0; target_states[i].reverse = false; }
        } 
        else if (current_time - last_bt_time < CMD_TIMEOUT) {
            // 手把模式優先
            switch (current_bt_mode) {
                case MODE_RESET: for(int i=0; i<4; i++) { target_states[i].angle = 0; target_states[i].speed = 0; } break;
                case MODE_ACKERMANN: 
                    kinematics.computeAckermann(1.0, steer_rx, target_states); 
                    for(int i=0; i<4; i++) target_states[i].speed *= actual_throttle; 
                    break;
                case MODE_CRAB_LEFT: kinematics.compute(0, 1.0, 0, target_states); for(int i=0; i<4; i++) target_states[i].speed = actual_throttle; break;
                case MODE_CRAB_RIGHT: kinematics.compute(0, -1.0, 0, target_states); for(int i=0; i<4; i++) target_states[i].speed = actual_throttle; break;
                case MODE_SPIN_CCW: kinematics.compute(0, 0, 1.0, target_states); for(int i=0; i<4; i++) target_states[i].speed = actual_throttle; break;
                case MODE_SPIN_CW: kinematics.compute(0, 0, -1.0, target_states); for(int i=0; i<4; i++) target_states[i].speed = actual_throttle; break;
            }
        } 
        else if (current_time - last_ros_time < CMD_TIMEOUT) {
            // ROS 模式 (當手把無輸入時)
            kinematics.compute(ros_cmd.vx, ros_cmd.vy, ros_cmd.omega, target_states);
        } 
        else {
            // 閒置狀態：停止動力，保持轉向
            for(int i=0; i<4; i++) { target_states[i].speed = 0; }
        }

        // --- C. 執行硬體控制 (輸出脈衝與 PWM) ---
        for(int i=0; i<4; i++) {
            if (steppers[i]) {
                // 1. 取得當前角度並計算轉向最佳化
                float current_deg = steppers[i]->getCurrentPosition() / 177.77;
                bool rev = false;
                float target_angle_corrected = -target_states[i].angle; // 考慮硬體安裝方向
                float final_deg = Kinematics::optimizeAngleStrict90(current_deg, target_angle_corrected, rev);
                
                // 2. 更新反轉狀態 (若轉向最佳化判定需要反轉，則疊加反轉訊號)
                target_states[i].reverse = target_states[i].reverse != rev; 
                
                // 3. 步進馬達移動
                steppers[i]->moveTo(long(final_deg * 177.77)); 
            }

            // 4. 動力馬達控制
            float final_speed = target_states[i].speed * (target_states[i].reverse ? -1.0 : 1.0); 
            float ratio = constrain(final_speed / MAX_VEL, -1.0, 1.0);
            
            // 將比例轉為 PPM 微秒訊號
            int pwm_val = PPM_MID;
            if (abs(ratio) > 0.01) {
                if (ratio > 0) pwm_val = (PPM_MID + PPM_DEAD) + (int)(ratio * (PPM_MAX - (PPM_MID + PPM_DEAD)));
                else pwm_val = (PPM_MID - PPM_DEAD) + (int)(ratio * ((PPM_MID - PPM_DEAD) - PPM_MIN));
            }
            bldcs[i].writeMicroseconds(pwm_val);
        }
        last_kin = current_time;
    }
}