#!/usr/bin/env python3
# modules/hand_tracker.py

# ==============================================================================
# 导入库
# ==============================================================================
import cv2          # 用于颜色空间转换和绘图 (单独测试要用)
import numpy as np  # Python中处理数组和矩阵的基础库
import jetson.utils # 用于与Jetson硬件交互 (单独测试用)
import mediapipe as mp # 实现手部追踪

# --- 仅在独立测试时需要导入 ---
# 为了让这个脚本可以独立运行测试，我们需要能够在这里也调用手势识别器。
# 用一个try-except结构来导入，这样即使主程序用不到它，也不会报错。
try:
    from gesture_recognizer import GestureRecognizer
except ImportError:
    # 如果是在主程序中被导入，这个导入可能会失败，但没关系。
    pass

# ==============================================================================
# 定义HandTracker类
# ==============================================================================

class HandTracker:
    """
    直接封装了Mediapipe手部追踪功能
    从一帧图像中找21个手势关键点
    """

    # --------------------------------------------------------------------------
    # 初始化方法 (__init__)
    # --------------------------------------------------------------------------
    def __init__(self, static_mode=False, max_hands=1, min_detect_conf=0.8, min_track_conf=0.8):
        """
        初始化手部追踪器。
        
        static_mode: 是否为静态图片模式。False表示视频
        max_hands: 最多检测1只手，测试了一下两个手太卡
        min_detect_conf: 检测置信度
        min_track_conf: 追踪置信度
        """
        print("Initializing HandTracker...")
        
        # --- 加载Mediapipe手部模型 ---
        self.mp_hands = mp.solutions.hands # 获取手部解决方案模块solutions.hands是Mediapipe中手部解决方案
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detect_conf,
            min_tracking_confidence=min_track_conf
        )
        
        # --- 加载绘图工具 ---
        self.mp_drawing = mp.solutions.drawing_utils # 绘图工具模块，用于画骨骼
        print("HandTracker initialized.")

    # --------------------------------------------------------------------------
    # 核心处理方法 (process_frame)
    # --------------------------------------------------------------------------
    def process_frame(self, np_img_rgb):
        """
        接收一帧图像，跑Mediapipe的手部处理，进行手部关键点检测
        np_img_rgb: Numpy格式的、RGB颜色空间的图像。
        return: 一个'results'对象，里面包含了检测到的所有手部信息。
        """
        # `process`接收图像并运行神经网
        # Mediapipe内部处理只读图像，不会修改传入的`np_img_rgb`
        return self.hands.process(np_img_rgb)

    # --------------------------------------------------------------------------
    # 绘制关节点方法 
    # --------------------------------------------------------------------------
    def draw_landmarks(self, np_img_to_draw_on, results):
        """
        根据检测结果，在给定的图像上绘制手部骨骼。
        
        np_img_to_draw_on: 需要被绘制的Numpy图像。
         results: 从`process_frame`方法返回的结果对象。
        """
        # --- 安全检查 ---
        # 1. 检查`results`本身是否有效 (不是None)。
        # 2. 检查`results.multi_hand_landmarks`是否存在，是否检测到了手
        if results and results.multi_hand_landmarks:
            # 遍历检测到的手
            for hand_landmarks in results.multi_hand_landmarks:
                # 直接调用Mediapipe提供的绘图函数，将关节点和连接线画在图像上。
                self.mp_drawing.draw_landmarks(
                    np_img_to_draw_on,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS # `HAND_CONNECTIONS`预定义了哪两个点之间需要画线
                )
    
    # --------------------------------------------------------------------------
    # 关闭资源方法 (close)
    # --------------------------------------------------------------------------
    def close(self):
        """
        释放Mediapipe模型占用的资源。
        """
        self.hands.close()
        
# ==============================================================================
# 单独测试代码
# ==============================================================================
# 运行 `python3 modules/hand_tracker.py` 时执行。
# 测试追踪手势和识别手势有没有问题
if __name__ == '__main__':
    print("\n-------------------------------------------")
    print("Running HandTracker in Standalone Test Mode...")
    print("-------------------------------------------")
    
    # 初始化手部追踪模块
    tracker = HandTracker()
    
    # 单独测试所以要一个识别器
    try:
        recognizer = GestureRecognizer()
    except NameError:
        print("\n[ERROR] GestureRecognizer not found. Please ensure gesture_recognizer.py is in the same directory.")
        exit()

    # 打开手势摄像头
    camera = jetson.utils.videoSource("csi://0", argv=['--input-width=320', '--input-height=240'])
    # 打开一个显示窗口
    display = jetson.utils.videoOutput("display://0")

    print("Camera is active. Showing hand tracking and gesture recognition.")
    while display.IsStreaming():
        img = camera.Capture()
        if img is None: continue

        # --- 数据准备 ---
        frame_np = jetson.utils.cudaToNumpy(img)
        frame_np_rgb = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2RGB)
        frame_np_rgb_flipped = cv2.flip(frame_np_rgb, 1) # 摄像头是镜像的翻转一下方便看手的动作

        # --- 核心处理 ---
        results = tracker.process_frame(frame_np_rgb_flipped)
        
        current_gesture = "No Hand"
        if results.multi_hand_landmarks:
            # 识别手势
            hand_landmarks = results.multi_hand_landmarks[0]
            current_gesture = recognizer.recognize(hand_landmarks)
            
            # 绘制骨骼
            tracker.draw_landmarks(frame_np_rgb_flipped, results)

        # --- 绘制额外信息 ---
        # 在屏幕上显示识别出的手势
        cv2.putText(frame_np_rgb_flipped, f"Gesture: {current_gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # --- 显示 ---
        output_img_cuda = jetson.utils.cudaFromNumpy(frame_np_rgb_flipped)
        display.Render(output_img_cuda)
        display.SetStatus(f"Hand Tracking & Gesture Recognition | {display.GetFrameRate():.1f} FPS")

    tracker.close()