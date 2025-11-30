# 智能水果摊自助结算系统 (Smart Fruit Stall Checkout System)

## 1. 项目简介 (Introduction)

本项目是一个基于 **NVIDIA Jetson Nano** 平台的计算机视觉自助结算系统，旨在模拟一个现代化、无接触的水果摊购物体验。系统通过实时视频流，利用深度学习和传统图像处理技术，实现了自动商品识别与手势指令交互两大核心功能。**（用于2025年秋季DSP验收）**

用户只需将选购的水果在摄像头前展示，系统即可自动将其加入虚拟购物车。随后，用户可以通过一系列直观的手势（如张手、点赞、指向）来管理购物车（撤销、清空）和完成结账流程，全程无需接触任何实体按键或屏幕，提供了一种新颖、卫生且高效的交互方式。

本项目充分利用了 Jetson Nano 的 GPU 加速能力，并通过精心的软件架构设计，在保证功能完整性的同时，实现了流畅、高帧率的用户体验，是一个集成了AI应用、硬件交互和软件工程实践的综合性嵌入式项目。

## 2. 环境配置与运行 (Setup & Execution)

### 2.1 硬件连接
1.  **水果识别摄像头**: 连接到 Jetson Nano 主板上（CSI-1）。
2.  **手势识别摄像头**: 连接到 Jetson Nano 主板上（CSI-0）
3.  **显示与音频**: 通过 **HDMI** 连接一个带有扬声器的显示器。
    *   ⚠️ **重要**: 连接后，请进入 Jetson Nano 的桌面 `Sound` 设置，确保 **Output Device** (输出设备) 被设置为 **HDMI / DisplayPort**。

### 2.2 部署步骤

**1. 放置项目文件**
确保项目文件夹位于 Nano 的用户根目录下：
```bash
/home/nano/smart_checkout_system/
2. 安装系统依赖
语音模块依赖底层的 espeak 引擎：
sudo apt-get update
sudo apt-get install espeak
3. 安装 Python 依赖
(注: jetson-inference 和 jetson-utils 为 JetPack 预装库，无需安装)
pip3 install opencv-python numpy pyttsx3==2.71 -i https://pypi.tuna.tsinghua.edu.cn/simple
2.3 运行项目
1. 进入项目目录
cd /home/nano/smart_checkout_system/
2. 设置显示环境 (如果是SSH远程连接)
export DISPLAY=:0
3. 启动主程序
python3 main.py
（如果报模型相关的错误，可能是因为TensorRT优化后的模型在不同设备之间不通用，新设备需要重新优化，现在的代码中已经有一个engine文件了，而这个文件是基于我当时的板子优化来的，可以把现在的engine删除，然后重新运行python3 main.py，这次运行TensorRT会对ONNX模型重新优化，会花费较长的时间，最后会生成新的engine文件，这个engine文件是针对当前板子生成的，下次运行就会非常快了）
2.4 操作流程指南
🍎 添加商品: 将一个水果（或显示水果图片的平板/手机）在主摄像头前展示一下（即从画面外移入）。系统识别后会自动添加，并有语音提示。要增加同一水果的数量，只需将其移出画面再重新移入即可。
🖐️ 撤销操作: 在手势摄像头前做出 张开手掌 的手势，系统会撤销上一次的添加操作。
👍 清空购物车: 做出 点赞 手势，购物车内的所有商品将被清空。
👆 发起结账: 当购物车不为空时，做出 指向 手势，系统会播报总价，并在屏幕中央弹出支付二维码。
✅ 完成交易: 在二维码界面，再次做出 点赞 手势，系统会播报感谢语，然后自动清空购物车，恢复到初始购物状态，准备为下一位顾客服务。
3. 功能亮点 (Key Features)
自动商品识别 (出现即添加): 模拟超市收银台的“扫码”行为。当一个水果首次出现在摄像头画面中时，系统会立即识别其种类并自动加入购物车。
丰富的无接触手势交互:
撤销 (Undo): 张开手掌 (open_palm)
清空 (Clear Cart): 点赞 (thumb_up)
结账 (Checkout): 指向 (pointing)
完成/取消 (Finish): 点赞 (thumb_up)
实时语音反馈 (Voice Announcer): 系统的每一步关键操作，如“添加商品”、“清空购物车”、“播报总价”等，都会有清晰的语音提示。这极大地增强了交互的确认感。
高性能与流畅体验:
双摄像头架构: 物理上分离了水果识别（CSI-1）和手势识别（CSI-0）的数据流，避免遮挡和焦点切换问题。
GPU加速推理: 水果识别利用 TensorRT 引擎在 GPU 上执行，保证 UI 刷新率稳定在 40-50 FPS。
CPU负载优化: 手势识别采用帧间隔采样技术（每3帧处理1次），将CPU负载降低了约66%，有效避免卡顿，保证跟手性。
专业且信息丰富的用户界面 (UI): 多窗口布局、实时数据列表显示、统一的美观设计。
4. 技术架构与实现 (Tech Stack & Architecture)
4.1 硬件平台
核心: NVIDIA Jetson Nano Developer Kit
视觉输入: 2 x CSI 摄像头 
4.2 软件技术栈
操作系统: Jetson-based Linux (Ubuntu 18.04)
编程语言: Python 3.6
核心AI框架:
jetson-inference: 基于 TensorRT 的高性能 GPU 物体检测。
mediapipe: Google 出品的跨平台机器学习应用库，用于手部关键点检测。
关键Python库:
OpenCV: 图像预处理与UI绘制。
numpy: 高效图像数组操作。
pyttsx3: 完全离线的文本转语音功能，保证语音播报的可靠性。
底层系统工具:
eSpeak: Linux 系统下的底层语音合成引擎。
4.3 模块化设计
项目遵循“高内聚、低耦合”的设计原则，将核心功能拆分为5个独立的Python模块，由 main.py 统一调度。
.
├── main.py                   # 🚀 负责主循环、状态管理和模块调度。
├── models/                   # 🧠 AI模型库：存放训练好的水果识别模型。
│   └── fruit/
│       ├── ssd-mobilenet.onnx
│       └── labels.txt
├── modules/                  # 🛠️ 核心功能模块库。
│   ├── object_detector.py    # 🍓 封装了jetson-inference，专职识别水果。
│   ├── hand_tracker.py       # 🖐️ 封装了mediapipe，专职定位手部21个关键点。
│   ├── gesture_recognizer.py # 👍 通过几何学分析关键点，解读手势含义。
│   ├── ui_manager.py         # 🎨 负责绘制所有UI元素，美化界面。
│   └── voice_announcer.py    # 🗣️ 封装了pyttsx3，专职将文本转换为语音。
├── payment_qr.png            # 💳 结账时显示的二维码图片。
└── README.md                 # 📄 本项目说明文件。

作者: 段奕嘉 
项目日期: 2025年11月
联系方式: yijiaduan116@gmail.com
