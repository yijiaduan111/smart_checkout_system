#!/usr/bin/env python3
# modules/object_detector.py

# ==============================================================================
# 导入必要的库
# ==============================================================================

# --- Jetson Inference 核心库 ---
# NVIDIA为Jetson平台提供的、用于高性能AI推理的Python库。
# etson Inference封装了TensorRT，能以极高的效率运行AI模型。
import jetson.inference 
import jetson.utils      # 导入工具库，再侧睡代码中用的

# --- 系统库 ---
import sys

# ==============================================================================
# 定义ObjectDetector类
# ==============================================================================

class ObjectDetector:
    """
    一个专门用于水果识别的检测器类，这个脚本不能使用OpenCV，OpenCV会和jetson.inference冲突，后面画框都直接在GPU上画，节省时间还避免了opencv
    - 只依赖NVIDIA官方库，不引入OpenCV，避免潜在的OpenCV库冲突
    - 充分利用TensorRT进行GPU加速推理。
    """
    # --------------------------------------------------------------------------
    # 初始化方法 (__init__)
    # --------------------------------------------------------------------------
    def __init__(self, model_path, labels_path, threshold=0.8):
        """
        当创建ObjectDetector对象时，这个方法会被调用。
        它负责加载并初始化AI模型。
        
        :param model_path: ONNX模型文件的路径 
        :param labels_path: 标签文件的路径 
        :param threshold: 置信度阈值
        """
        print("正在初始化水果识别模型...")
        
        # --- 构造命令行参数 ---
        # jetson.inference本来需要命令行参数        
        # 我们可以通过一个列表来模拟命令行参数，以配置模型的加载方式。
        argv = [
            f"--model={model_path}",     # 指定ONNX模型文件的位置
            f"--labels={labels_path}",    # 指定标签文件的位置
            "--input-blob=input_0",      # 告诉网络输入的节点名称 (通常在模型转换时定义)
            "--output-cvg=scores",     # 告诉网络输出分数的节点名称
            "--output-bbox=boxes"      # 告诉网络输出边界框的节点名称
        ]
        
        # --- 加载网络并触发TensorRT优化 ---
        # 当第一次运行的时候：解析ONNX、TensorRT优化、生成Engine
        # 当第二次及以后运行时，直接加载缓存的Engine文件，推理会非常快
        self.net = jetson.inference.detectNet(argv=argv, threshold=threshold)
        
        print("水果识别模型初始化完毕。")

    # --------------------------------------------------------------------------
    # 检测与绘制方法 (detect_and_draw)
    # --------------------------------------------------------------------------
    def detect_and_draw(self, original_img):
        """
        接收一帧图像，执行物体检测，并将结果直接绘制在原始图像上。
        
        :param original_img: 从jetson.utils.videoSource捕获的原始CUDA图像 (在GPU显存中)。
        :return: 一个包含所有检测结果的列表 (detections)。
        """
        # --- GPU快速推理 ---
        # self.net.Detect() 直接将GPU中的图像直接送入TensorRT。
        # 所有的计算都在GPU上完成
        # 'overlay'可以使得在函数在完成检测后，直接在原始图像上绘制边界框(box)、标签(labels)和置信度(conf)，相当于直接在GPU完成，避免把数据拷贝到CPU在用OpenCV绘制
        detections = self.net.Detect(original_img, overlay='box,labels,conf')
        
        # 返回检测结果列表。每个结果是一个对象，包含了类别ID、置信度、边界框坐标等信息。
        return detections

    # --------------------------------------------------------------------------
    # 获取性能指标方法 (get_network_fps)
    # --------------------------------------------------------------------------
    def get_network_fps(self):
        """
        获取AI模型本身的推理速度（FPS - 每秒帧数）。
        这个值只反应GPU处理模型的速度，不管手势那边的检测速度
        """
        return self.net.GetNetworkFPS()

# ==============================================================================
# 3单独测试的代码（单独检测水果识别有没有问题）
# ==============================================================================
# 这部分代码在直接运行 `python3 modules/object_detector.py` 时执行。
if __name__ == '__main__':
    print("以独立模式运行水果检测模块 (纯净版)...")

    # 定义模型文件所在的路径
    model_dir = "../models/fruit/"
    model_path = model_dir + "ssd-mobilenet.onnx"
    labels_path = model_dir + "labels.txt"

    # 创建检测器实例
    detector = ObjectDetector(model_path, labels_path, threshold=0.5)

    # 打开水果摄像头
    camera = jetson.utils.videoSource("csi://1") 
    # 打开一个显示窗口
    display = jetson.utils.videoOutput("display://0")

    print("摄像头已开启，开始检测水果...")
    # 循环捕获、检测和显示
    while display.IsStreaming():
        img = camera.Capture()
        if img is None: continue
            
        # 调用我们上面定义的核心功能
        detections = detector.detect_and_draw(img)
        
        if len(detections) > 0:
            print(f"检测到 {len(detections)} 个水果!")

        # 将带有检测框的图像渲染到屏幕上
        display.Render(img)
        
        # 在窗口标题栏显示模型的推理帧率
        display.SetStatus("智能水果店 | {:.1f} FPS".format(detector.get_network_fps()))
            
    print("退出测试模式。")