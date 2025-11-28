#!/usr/bin/env python3
# 导入所有必需的库。
# ==============================================================================
import os
import pyttsx3      # 核心的离线文本转语音库
import threading    # 用于实现异步播放，防止阻塞主程序

# ==============================================================================
# 核心功能实现
# ------------------------------------------------------------------------------
# 将所有功能封装在一个简洁的模块级函数中。
# ==============================================================================

# 全局引擎实例 ---
# 模块加载时就初始化一个全局的pyttsx3引擎实例。
#   避免在每次调用`say`函数时都重复初始化，提高了效率。保证整个程序只有一个引擎实例在工作
try:
    engine = pyttsx3.init()
    # --- 调整语音属性 ---
    rate = engine.getProperty('rate')  # 获取当前语速
    engine.setProperty('rate', rate + 50) # 在默认基础上增加50，让语速更快
    
    # 创建一个线程锁，用于在多线程环境下安全地访问引擎
    engine_lock = threading.Lock()
    print("Global pyttsx3 engine initialized successfully.")

except Exception as e:
    # 如果因为某些原因（如系统音频服务问题）初始化失败，
    # 我们将engine设为None，并在后续调用中安全地跳过播报，保证主程序不会崩溃。
    engine = None
    print(f"ERROR: Failed to initialize global pyttsx3 engine: {e}")


# ---  核心播报函数 `say()` ---
def say(text):
    """
    用于在后台线程中异步播报文本。
    :param text: 需要播报的字符串。
    """
    # --- 安全检查 ---
    # 如果引擎初始化失败或传入的文本为空，则直接返回。
    if not engine or not text:
        return

    # --- 内部函数 `run()` ---
    # 这个函数包含了实际的、会阻塞的语音播报逻辑。
    # 我们将把它放到一个独立的线程中去执行。
    def run():
        # `with engine_lock:` 确保在同一时刻只有一个线程可以操作引擎，避免冲突。
        with engine_lock:
            print(f"    Saying: '{text}'")
            try:
                # --- 打断机制 ---
                # 如果引擎当前正在忙（播放上一条语音），先让它强制停止。
                # 这保证了用户总是能听到最新的、最及时的反馈。
                if engine.isBusy():
                    engine.stop()
                
                # --- 核心播报指令 ---
                engine.say(text)        # 1. 将文本放入播放队列
                engine.runAndWait()     # 2. 开始播报，并“阻塞”直到播报完毕
            except Exception as e:
                print(f"[VoiceAnnouncer ERROR] Failed to say '{text}': {e}")

    # --- 异步执行 ---
    # 创建一个新的线程，目标是执行我们上面定义的`run`函数。
    thread = threading.Thread(target=run)
    thread.daemon = True # 如果主程序退出了，这个线程也会被强制结束。
    thread.start()       # 启动线程。主程序会立即继续执行下一行代码，不会在此等待。


# ==============================================================================
#  独立测试代码
if __name__ == '__main__':
    import time
    print("\n--- Running Voice Announcer Test (Pure pyttsx3) ---")
    if engine:
        say("This is a long sentence that will be interrupted by the next one.")
        time.sleep(1) # 只等1秒，这句话肯定没说完
        say("Interrupted!") # 这个声音会打断并取代上一句话
        
        # 等待最后一个语音播放完毕
        while engine.isBusy():
            time.sleep(0.1)
        
        print("\nTest finished.")