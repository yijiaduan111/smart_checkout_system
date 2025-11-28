# modules/ui_manager.py (Final Version - More Items)

import cv2
import numpy as np

class UIManager:
    def __init__(self, window_width=960, window_height=800):
        print("🎨 Initializing UIManager (Final Version)...")
        self.width = window_width
        self.height = window_height
        
        self.fruit_cam_rect = (0, 0, 640, 480)
        self.hand_cam_rect = (640, 0, 320, 240)
        self.cart_rect = (0, 480, 960, 320)
        
        self.colors = {
            'bg': (30, 30, 30), 'header': (255, 200, 0), 'text': (255, 255, 255),
            'line': (80, 80, 80), 'total_bg': (40, 40, 40)
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        print("✅ UIManager initialized.")

    def create_background(self):
        return np.full((self.height, self.width, 3), self.colors['bg'], dtype=np.uint8)

    def draw_video_frames(self, background, fruit_frame, hand_frame):
        # ... (这部分无变化) ...
        fruit_frame_resized = cv2.resize(fruit_frame, (self.fruit_cam_rect[2], self.fruit_cam_rect[3]))
        background[self.fruit_cam_rect[1]:self.fruit_cam_rect[1]+self.fruit_cam_rect[3], self.fruit_cam_rect[0]:self.fruit_cam_rect[0]+self.fruit_cam_rect[2]] = fruit_frame_resized
        hand_frame_resized = cv2.resize(hand_frame, (self.hand_cam_rect[2], self.hand_cam_rect[3]))
        background[self.hand_cam_rect[1]:self.hand_cam_rect[1]+self.hand_cam_rect[3], self.hand_cam_rect[0]:self.hand_cam_rect[0]+self.hand_cam_rect[2]] = hand_frame_resized
        
    def draw_shopping_cart(self, background, shopping_cart, total_price):
        x, y, w, h = self.cart_rect
        cv2.rectangle(background, (x, y), (x + w, y + h), (50, 50, 50), -1)
        cv2.putText(background, "Shopping Cart", (x + 20, y + 40), self.font, 1.2, self.colors['header'], 2)
        cv2.line(background, (x + 20, y + 60), (x + w - 20, y + 60), self.colors['line'], 1)
        
        item_y_start = y + 85 # 稍微向上移动一点起始位置
        if not shopping_cart:
            cv2.putText(background, "Your cart is empty.", (x + 30, item_y_start + 10), self.font, 0.7, (150, 150, 150), 1)
        else:
            for i, (item_name, details) in enumerate(shopping_cart.items()):
                # --- 关键改动: 减小行距和字体，增加最大行数 ---
                line_height = 28 # 行间距
                font_scale = 0.7 # 字体大小
                max_items_to_show = 9 # 最多显示9行
                
                if i >= max_items_to_show:
                    cv2.putText(background, "...", (x + 40, item_y_start + i * line_height), self.font, font_scale, self.colors['text'], 1)
                    break
                item_text_left = f"- {item_name.capitalize()} x{details['count']}"
                cv2.putText(background, item_text_left, (x + 40, item_y_start + i * line_height), self.font, font_scale, self.colors['text'], 1)
                
                item_text_right = f"${details['price'] * details['count']:.2f}"
                text_size_right = cv2.getTextSize(item_text_right, self.font, font_scale, 1)[0]
                cv2.putText(background, item_text_right, (x + w - text_size_right[0] - 40, item_y_start + i * line_height), self.font, font_scale, self.colors['text'], 1)
        
        total_bar_y = y + h - 60
        cv2.rectangle(background, (x, total_bar_y), (x + w, y + h), self.colors['total_bg'], -1)
        cv2.line(background, (x, total_bar_y), (x+w, total_bar_y), self.colors['line'], 1)
        total_price_text = f"Total: ${total_price:.2f}"
        text_size_total = cv2.getTextSize(total_price_text, self.font, 1.1, 2)[0]
        cv2.putText(background, total_price_text, (x + w - text_size_total[0] - 30, total_bar_y + 40), self.font, 1.1, self.colors['header'], 2)

    def draw_qr_code(self, background, qr_image):
        # ... (这部分无变化) ...
        overlay = background.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        alpha = 0.7
        background = cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0)
        qr_h, qr_w, _ = qr_image.shape
        x_offset = (self.width - qr_w) // 2
        y_offset = (self.height - qr_h) // 2
        background[y_offset:y_offset+qr_h, x_offset:x_offset+qr_w] = qr_image
        msg = "Scan to pay. Make a THUMB UP to cancel."
        text_size = cv2.getTextSize(msg, self.font, 0.8, 2)[0]
        cv2.putText(background, msg, ((self.width - text_size[0]) // 2, y_offset + qr_h + 40), self.font, 0.8, (255, 255, 255), 2)
        return background