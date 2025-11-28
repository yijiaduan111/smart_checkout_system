#!/usr/bin/env python3
# 导入所有必需的库和自定义的功能模块。
# ==============================================================================

# --- 系统与基础库 ---
import sys
import os
import time

# --- 计算机视觉与硬件加速库 ---
import cv2          # 用于图像处理 (加载、缩放图片)
import numpy as np  # 用于图像数据的数组操作
import jetson.utils # NVIDIA Jetson 平台专用工具库，用于高效访问摄像头和显示

# --- 自定义功能模块 ---
# 将 'modules' 文件夹的路径添加到Python解释器的搜索列表中
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from object_detector import ObjectDetector
from hand_tracker import HandTracker
from ui_manager import UIManager
from gesture_recognizer import GestureRecognizer
from modules.voice_announcer import say as announcer_say


# ==============================================================================
# 全局配置 (Global Configurations)
# ------------------------------------------------------------------------------
# 定义整个程序中使用的常量和配置参数。
# ==============================================================================

# --- 商品价格数据库 ---
# 使用字典结构存储，键为小写水果名，值为单价。
PRICE_LIST = {
    "apple": 2.50, "orange": 1.80, "banana": 1.20, "strawberry": 4.00,
    "grape": 5.50, "pear": 2.00, "pineapple": 3.50, "watermelon": 8.00
}

# 定义每隔多少帧进行一次手势识别
GESTURE_CHECK_INTERVAL = 3
def main():
    """
    程序的主函数，封装了所有的初始化、主循环和资源管理。
    """
    
    # --------------------------------------------------------------------------
    # 初始化
    # --------------------------------------------------------------------------
    
    print(" Starting Smart Fruit Stall System (Final Optimized Version)...")

    # --- 实例化所有自定义模块 ---
    fruit_detector = ObjectDetector(model_path="models/fruit/ssd-mobilenet.onnx", labels_path="models/fruit/labels.txt")
    hand_tracker = HandTracker()
    ui = UIManager()
    gesture_recognizer = GestureRecognizer()
    
    # --- 加载静态资源 (二维码图片) ---
    qr_image_path = "payment_qr.png"
    qr_code_img = cv2.imread(qr_image_path)
    if qr_code_img is not None:
        qr_code_img = cv2.resize(qr_code_img, (300, 300))
    else:
        # 如果图片加载失败，创建一个带错误提示的黑色方块，保证程序健壮性
        qr_code_img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(qr_code_img, "QR NOT FOUND", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- 初始化硬件接口 (摄像头与显示) ---
    fruit_cam = jetson.utils.videoSource("csi://1", argv=['--input-width=640', '--input-height=480'])
    hand_cam = jetson.utils.videoSource("csi://0", argv=['--input-width=320', '--input-height=240'])
    display = jetson.utils.videoOutput("display://0", argv=[f'--width={ui.width}', f'--height={ui.height}'])
    
    # --- 初始化语音播报 ---
    announcer_say("Welcome")

    # --- 初始化程序状态变量 ---
    shopping_cart = {}             # 购物车
    last_detected_fruits = set()   # “记忆”：上一帧的水果
    addition_history = []          # “历史记录”：用于撤销
    checkout_mode = False          # “开关”：是否为结账模式
    last_action_time = 0           # “计时器”：用于手势防抖
    
    # --- 初始化性能优化所需变量 ---
    frame_counter = 0              # 帧计数器
    last_known_gesture = "No Hand" # “记忆”：上一次有效的手势结果
    last_hand_results = None       # “记忆”：上一次的骨骼数据，用于平滑显示


    # ==========================================================================
    # 主循环 
    
    while display.IsStreaming():
        
        # ----------------------------------------------------------------------
        #  数据采集 (Data Acquisition)
        # ----------------------------------------------------------------------
        frame_counter += 1
        fruit_img = fruit_cam.Capture()
        hand_img = hand_cam.Capture()
        if fruit_img is None or hand_img is None: continue

        # ----------------------------------------------------------------------
        # 步骤 4.2: 核心处理 (Core Processing)
        # ----------------------------------------------------------------------

        # a) 水果检测 (每帧都执行，由GPU加速，非常快)
        detections = fruit_detector.detect_and_draw(fruit_img)
        fruit_frame_np = jetson.utils.cudaToNumpy(fruit_img)
        
        # b) 手势识别 
        if frame_counter % GESTURE_CHECK_INTERVAL == 0:
            hand_frame_np_rgb = cv2.cvtColor(jetson.utils.cudaToNumpy(hand_img), cv2.COLOR_RGBA2RGB)
            hand_frame_flipped = cv2.flip(hand_frame_np_rgb, 1)#镜像
            results = hand_tracker.process_frame(hand_frame_flipped)
            last_hand_results = results # 更新骨骼数据
            
            if results.multi_hand_landmarks:
                last_known_gesture = gesture_recognizer.recognize(results.multi_hand_landmarks[0])
            else:
                last_known_gesture = "No Hand"
        
        current_gesture = last_known_gesture # 使用最近一次的有效结果

        # ----------------------------------------------------------------------
        # 交互逻辑 (Interaction Logic - State Machine)
        # ----------------------------------------------------------------------
        time_since_last_action = time.time() - last_action_time
        
        # --- 状态一: 结账模式 (Checkout Mode) ---
        if checkout_mode:
            if current_gesture == "thumb_up" and time_since_last_action > 1.5:
                announcer_say("Thank you. Cart is now clear.")
                # 重置所有状态，为下一位顾客准备
                shopping_cart.clear(); addition_history.clear(); last_detected_fruits.clear()
                checkout_mode = False
                last_action_time = time.time()
        
        # --- 状态二: 购物模式 (Shopping Mode) ---
        else:
            # 自动添加商品、大小写转换
            current_detected_fruits = {fruit_detector.net.GetClassDesc(det.ClassID).strip().lower() for det in detections if fruit_detector.net.GetClassDesc(det.ClassID).strip().lower() != 'background'}
            newly_appeared_fruits = current_detected_fruits - last_detected_fruits
            if newly_appeared_fruits:
                first_new_fruit = list(newly_appeared_fruits)[0]#只念最新加的一种水果
                announcer_say(f"{first_new_fruit} added.")
                for fruit in newly_appeared_fruits:
                    if fruit in shopping_cart: shopping_cart[fruit]['count'] += 1#数量加一
                    else: shopping_cart[fruit] = {'count': 1, 'price': PRICE_LIST[fruit]}
                    addition_history.append(fruit)#加入历史记录
            last_detected_fruits = current_detected_fruits#更新记忆

            # 手势操作
            if current_gesture == "pointing" and shopping_cart and time_since_last_action > 1.5:
                total_price = sum(item['count'] * item['price'] for item in shopping_cart.values())
                announcer_say(f"Total price is {total_price:.2f} dollars. Please scan to pay.")
                checkout_mode = True#切换到结账模式
                last_action_time = time.time()
            elif time_since_last_action > 1.5:
                if current_gesture == "thumb_up" and shopping_cart:
                    announcer_say("Cart cleared.")
                    shopping_cart.clear(); addition_history.clear(); last_detected_fruits.clear()
                    last_action_time = time.time()
                elif current_gesture == "open_palm" and addition_history:
                    last_added_fruit = addition_history.pop()
                    announcer_say(f"Undo {last_added_fruit}.")
                    if last_added_fruit in shopping_cart:
                        shopping_cart[last_added_fruit]['count'] -= 1#数量减一
                        if shopping_cart[last_added_fruit]['count'] == 0:
                            del shopping_cart[last_added_fruit]
                    last_action_time = time.time()

        # ----------------------------------------------------------------------
        # 界面渲染 (UI Rendering)
        # ----------------------------------------------------------------------
        total_price = sum(item['count'] * item['price'] for item in shopping_cart.values())

        # 准备手势窗口的实时画面
        hand_frame_np_rgb = cv2.cvtColor(jetson.utils.cudaToNumpy(hand_img), cv2.COLOR_RGBA2RGB)
        hand_frame_to_draw = cv2.flip(hand_frame_np_rgb, 1)
        hand_tracker.draw_landmarks(hand_frame_to_draw, last_hand_results) # 用记住的骨骼数据绘制
        cv2.putText(hand_frame_to_draw, f"Gesture: {current_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 调用UI管理器进行最终合成
        background = ui.create_background()
        ui.draw_video_frames(background, fruit_frame_np, hand_frame_to_draw)
        ui.draw_shopping_cart(background, shopping_cart, total_price)

        if checkout_mode:
            background = ui.draw_qr_code(background, qr_code_img)

        # 将最终画面显示到屏幕
        final_img_cuda = jetson.utils.cudaFromNumpy(background)
        display.Render(final_img_cuda)
        display.SetStatus(f"Smart Fruit Stall | FPS: {fruit_detector.get_network_fps():.1f}")


# ==============================================================================
# 区域 5: 程序启动入口
# ------------------------------------------------------------------------------
# 确保这个脚本被直接执行时，会调用 `main` 函数。
# ==============================================================================

if __name__ == "__main__":
    main()