#!/usr/bin/env python3
# modules/gesture_recognizer.py (Final Version)

# ==============================================================================
# 导入必要的库
# ==============================================================================
import mediapipe as mp 
import math          # 用于计算平方根和距离

# ==============================================================================
# 定义GestureRecognizer类
# ==============================================================================

class GestureRecognizer:
    """
    依据手部关键点识别手势
    """

    # --------------------------------------------------------------------------
    # 初始化方法 (__init__)
    # --------------------------------------------------------------------------
    def __init__(self):
        print("Initializing GestureRecognizer (Final Version)...")
        self.mp_hands = mp.solutions.hands
        
        # --- 为了方便，我们预先定义好需要用到的关节点 ---
        # 指尖的ID列表
        self.finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,        # 拇指指尖 (ID: 4)
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP, # 食指指尖 (ID: 8)
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,# 中指指尖 (ID: 12)
            self.mp_hands.HandLandmark.RING_FINGER_TIP,  # 无名指指尖 (ID: 16)
            self.mp_hands.HandLandmark.PINKY_TIP         # 小指指尖 (ID: 20)
        ]
        # 手指中间关节(PIP)的ID列表 (拇指除外，用IP代替)
        self.finger_pip = [
            self.mp_hands.HandLandmark.THUMB_IP,         # 拇指IP关节点 (ID: 3)
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP, # 食指PIP关节点 (ID: 6)
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,# 中指PIP关节点 (ID: 10)
            self.mp_hands.HandLandmark.RING_FINGER_PIP,  # 无名指PIP关节点 (ID: 14)
            self.mp_hands.HandLandmark.PINKY_PIP         # 小指PIP关节点 (ID: 18)
        ]
        print("GestureRecognizer initialized.")
        
    # --------------------------------------------------------------------------
    # 辅助方法（主要利用距离）
    # --------------------------------------------------------------------------
    def _get_distance(self, p1, p2):
        """计算两个关节点在2D平面上的欧几里得距离"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    # --------------------------------------------------------------------------
    # 核心识别方法 (recognize)
    # --------------------------------------------------------------------------
    def recognize(self, hand_landmarks):
        """
        接收一个手(21个)的关节点，判断它是什么手势。
        
        :param hand_landmarks: Mediapipe返回的单手关节点列表。
        :return: 手势名称字符串 ( "pointing", "thumb_up", "open_palm", "unknown")
        """
        if not hand_landmarks:
            return None

        landmarks = hand_landmarks.landmark # landmarks是一个包含21个点的列表
        
        # --- 第一步：判断每一根手指是“伸直”还是“弯曲” ---
        fingers_straight = []
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST] # 获取手腕关节点(ID: 0)作为基准点
        
        # 这个循环遍历五根手指
        for i in range(5):
            tip = landmarks[self.finger_tips[i]] # 获取当前手指的指尖
            pip = landmarks[self.finger_pip[i]] # 获取当前手指的中间关节
            
            # --- 核心逻辑 ---
            # 如果“指尖到手腕的距离”大于“中间关节到手腕的距离”，认为这根手指是伸直的，否则是弯曲的
            if self._get_distance(tip, wrist) > self._get_distance(pip, wrist):
                fingers_straight.append(True)
            else:
                fingers_straight.append(False)
        
        # `fingers_straight` 现在是一个布尔列，表示手指是不是直的，例如 [True, False, False, False, False] 表示只有大拇指伸直。

        # --- 第二步：根据手指伸直/弯曲的组合来定义手势 ---
        # 这里的if/elif顺序很重要，决定了判断的优先级。
        
        # 1. 指向 (结账)
        #只有食指(索引1)是伸直的，并且中指(2)、无名指(3)、小指(4)都是弯曲的。
        if fingers_straight[1] and not fingers_straight[2] and not fingers_straight[3] and not fingers_straight[4]:
            return "pointing"
            
        # 2. 点赞 (清空)
        # 条件：只有大拇指(索引0)是伸直的，其余四指都是弯曲的。
        if fingers_straight[0] and not fingers_straight[1] and not fingers_straight[2] and not fingers_straight[3] and not fingers_straight[4]:
            return "thumb_up"
            
        # 3. 五指张开 (撤销)
        # 条件：所有五根手指(索引0到4)都是伸直的。
        if all(fingers_straight):
            return "open_palm"
        
        # 如果以上条件都不满足，则返回“未知”
        return "unknown"