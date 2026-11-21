import collections
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from mss import mss
import win32gui
import win32process
import psutil
import time
import platform
import sys
import subprocess
import os
from typing import Optional, Dict, Any, Tuple
import importlib
from abc import ABC, abstractmethod
import threading
import queue
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== 加速后端实现 ====================

class AccelerationBackend(ABC):
    """加速后端抽象基类"""
    def __init__(self):
        self.name = "CPU"
        self.initialized = False
        self.backend_info = {}
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化加速后端"""
        pass
    
    @abstractmethod
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """处理帧并返回阈值化差异图像"""
        pass
    
    @abstractmethod
    def release(self):
        """释放资源"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """获取后端信息"""
        return self.backend_info

class CUDABackend(AccelerationBackend):
    """使用OpenCV CUDA加速"""
    def __init__(self):
        super().__init__()
        self.name = "CUDA"
        self.stream: Optional[cv2.cuda_Stream] = None
        self.gpu_frame1: Optional[cv2.cuda_GpuMat] = None
        self.gpu_frame2: Optional[cv2.cuda_GpuMat] = None
    
    def initialize(self) -> bool:
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() == 0:
                return False
            
            self.stream = cv2.cuda_Stream()
            self.gpu_frame1 = cv2.cuda_GpuMat()
            self.gpu_frame2 = cv2.cuda_GpuMat()
            self.backend_info["cuda_devices"] = cv2.cuda.getCudaEnabledDeviceCount()
            self.initialized = True
            return True
        except:
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized or self.stream is None:
            return super().process_frames(frame1, frame2, threshold)
        
        try:
            self.gpu_frame1.upload(frame1, self.stream)
            self.gpu_frame2.upload(frame2, self.stream)
            
            gpu_gray1 = cv2.cuda.cvtColor(self.gpu_frame1, cv2.COLOR_BGR2GRAY, stream=self.stream)
            gpu_gray2 = cv2.cuda.cvtColor(self.gpu_frame2, cv2.COLOR_BGR2GRAY, stream=self.stream)
            
            gpu_diff = cv2.cuda.absdiff(gpu_gray1, gpu_gray2, stream=self.stream)
            _, gpu_thresh = cv2.cuda.threshold(gpu_diff, threshold, 255, cv2.THRESH_BINARY, stream=self.stream)
            
            thresh = gpu_thresh.download(stream=self.stream)
            self.stream.waitForCompletion()
            return thresh
        except:
            self.initialized = False
            return super().process_frames(frame1, frame2, threshold)
    
    def release(self):
        if self.stream is not None:
            self.stream = None
        self.gpu_frame1 = None
        self.gpu_frame2 = None

class OpenCLBackend(AccelerationBackend):
    """使用OpenCL加速"""
    def __init__(self):
        super().__init__()
        self.name = "OpenCL"
    
    def initialize(self) -> bool:
        try:
            if not cv2.ocl.haveOpenCL():
                return False
            
            cv2.ocl.setUseOpenCL(True)
            if not cv2.ocl.useOpenCL():
                return False
            
            self.backend_info["opencl_available"] = True
            self.initialized = True
            return True
        except:
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized:
            return super().process_frames(frame1, frame2, threshold)
        
        try:
            umat1 = cv2.UMat(frame1)
            umat2 = cv2.UMat(frame2)
            
            gray1 = cv2.cvtColor(umat1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(umat2, cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            return cv2.UMat.get(thresh) if isinstance(thresh, cv2.UMat) else thresh
        except:
            self.initialized = False
            return super().process_frames(frame1, frame2, threshold)
    
    def release(self):
        """释放OpenCL资源"""
        cv2.ocl.setUseOpenCL(False)
        self.initialized = False

class NumbaBackend(AccelerationBackend):
    """使用Numba JIT编译加速"""
    def __init__(self):
        super().__init__()
        self.name = "Numba"
        
    def initialize(self) -> bool:
        try:
            import numba
            from numba import jit
            self.backend_info["numba_version"] = numba.__version__
            
            @jit(nopython=True, nogil=True)
            def numba_process(frame1, frame2, threshold):
                diff = np.abs(frame1.astype(np.int16) - frame2.astype(np.int16))
                return (diff > threshold).astype(np.uint8) * 255
            
            self._numba_process = numba_process
            self.initialized = True
            return True
        except Exception as e:
            print(f"Numba初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized:
            return super().process_frames(frame1, frame2, threshold)
        
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            return self._numba_process(gray1, gray2, threshold)
        except:
            self.initialized = False
            return super().process_frames(frame1, frame2, threshold)
    
    def release(self):
        pass

class PyTorchBackend(AccelerationBackend):
    """使用PyTorch GPU加速"""
    def __init__(self):
        super().__init__()
        self.name = "PyTorch"
        self.device = None
        self._torch_available = False
    
    def initialize(self) -> bool:
        try:
            import torch
            self._torch_available = True
            self.backend_info["torch_version"] = torch.__version__
            
            # 添加更详细的错误处理
            if not torch.cuda.is_available():
                print("PyTorch CUDA不可用，将使用CPU模式")
                self.device = torch.device('cpu')
                return False
            
            try:
                self.device = torch.device('cuda')
                # 测试CUDA是否真的可用
                test_tensor = torch.tensor([1.0]).cuda()
                self.initialized = True
                return True
            except Exception as e:
                print(f"PyTorch CUDA测试失败: {str(e)}")
                self.device = torch.device('cpu')
                return False
            
        except Exception as e:
            print(f"PyTorch初始化失败: {str(e)}")
            self._torch_available = False
            return False

    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self._torch_available or not self.initialized or self.device is None:
            return super().process_frames(frame1, frame2, threshold)
        
        try:
            import torch
            tensor1 = torch.from_numpy(frame1).permute(2, 0, 1).float().to(self.device)
            tensor2 = torch.from_numpy(frame2).permute(2, 0, 1).float().to(self.device)
            
            gray1 = (0.299 * tensor1[0] + 0.587 * tensor1[1] + 0.114 * tensor1[2]).to(torch.uint8)
            gray2 = (0.299 * tensor2[0] + 0.587 * tensor2[1] + 0.114 * tensor2[2]).to(torch.uint8)
            
            diff = torch.abs(gray1.int() - gray2.int())
            thresh = (diff > threshold).to(torch.uint8) * 255
            
            return thresh.cpu().numpy()
        except Exception as e:
            print(f"PyTorch处理失败: {str(e)}")
            self.initialized = False
            return super().process_frames(frame1, frame2, threshold)
    
    def release(self):
        if hasattr(self, 'device'):
            import torch
            torch.cuda.empty_cache()

class CPUBackend(AccelerationBackend):
    """CPU后备方案"""
    def __init__(self):
        super().__init__()
        self.name = "CPU"
    
    def initialize(self) -> bool:
        self.initialized = True
        return True
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def release(self):
        pass

class AccelerationManager:
    """加速后端管理器"""
    def __init__(self):
        self.backends = [
            CUDABackend(),
            OpenCLBackend(),
            NumbaBackend(),
            CPUBackend(),  # 最后一个是CPU后备
            PyTorchBackend()
        ]
        self.current_backend: Optional[AccelerationBackend] = None
    
    def detect_best_backend(self) -> AccelerationBackend:
        """检测并返回最佳可用的加速后端"""
        for backend in self.backends:
            if backend.initialize():
                print(f"检测到加速后端: {backend.name}")
                print(f"后端信息: {backend.get_info()}")
                return backend
        print("未检测到加速后端，使用CPU")
        return self.backends[-1]  # 返回CPU后端
    
    def set_backend(self, backend_name: str) -> bool:
        """设置特定的加速后端"""
        for backend in self.backends:
            if backend.name.lower() == backend_name.lower():
                if backend.initialize():
                    if self.current_backend:
                        self.current_backend.release()
                    self.current_backend = backend
                    return True
        return False
    
    def get_current_backend(self) -> AccelerationBackend:
        """获取当前加速后端"""
        if self.current_backend is None:
            self.current_backend = self.detect_best_backend()
        return self.current_backend
    
    def release_all(self):
        """释放所有后端资源"""
        for backend in self.backends:
            backend.release()
        self.current_backend = None

# ==================== 多线程处理类 ====================

class FrameProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)  # 4核线程池
        self.frame_queue = Queue(maxsize=10)  # 限制队列大小防止内存溢出
        self.result_queue = Queue(maxsize=10)
        self.running = True
        self.use_multithread = True  # 默认启用多线程
        
    def set_multithread(self, enabled: bool):
        """设置是否使用多线程"""
        self.use_multithread = enabled
        
    def process_frame_task(self, frame1, frame2, threshold, backend):
        """线程任务函数"""
        try:
            thresh = backend.process_frames(frame1, frame2, threshold)
            return thresh
        except Exception as e:
            print(f"线程处理失败: {e}")
            return None
    
    def process_frames(self, frame1, frame2, threshold, backend):
        """处理帧，根据设置选择多线程或单线程"""
        if not self.use_multithread:
            # 单线程处理
            return backend.process_frames(frame1, frame2, threshold)
            
        try:
            # 多线程处理
            self.frame_queue.put((frame1, frame2))
            if not self.result_queue.empty():
                future = self.result_queue.get()
                new_thresh = future.result()
                if new_thresh is not None:
                    if new_thresh.dtype != np.uint8:
                        new_thresh = new_thresh.astype(np.uint8)
                    return new_thresh
            return backend.process_frames(frame1, frame2, threshold)  # 回退到单线程
        except Exception as e:
            print(f"多线程处理失败: {str(e)}，使用单线程")
            return backend.process_frames(frame1, frame2, threshold)
    
    def start_processing(self, threshold, backend):
        """启动处理线程"""
        def worker():
            while self.running:
                try:
                    frame1, frame2 = self.frame_queue.get(timeout=0.1)
                    try:
                        future = self.executor.submit(
                            self.process_frame_task, 
                            frame1, frame2, threshold, backend
                        )
                        self.result_queue.put(future, timeout=0.1)
                    except Exception as e:
                        print(f"任务提交失败: {str(e)}")
                        # 将帧重新放回队列以便重试
                        self.frame_queue.put((frame1, frame2))
                except queue.Empty:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"工作线程错误: {str(e)} - {type(e).__name__}")
                    continue
        
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
    
    def stop_processing(self):
        """停止处理线程"""
        self.running = False
        self.executor.shutdown(wait=True)
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join()

# === 核心AI优化引擎 ===

class AIOptimizer:
    def __init__(self):
        self.last_tune_time = 0
        self.model = None
        # 尝试加载轻量级TensorFlow Lite模型
        try:
            from tensorflow.lite.python.interpreter import Interpreter as tflite
            try:
                self.model = tflite.Interpreter(model_path="ai_optimizer.tflite")
                self.model.allocate_tensors()
            except Exception as e:
                print(f"AI模型加载失败: {e}")
        except ImportError:
            print("警告: 未安装tflite_runtime，AI优化功能将受限")
            # 如果没有tflite_runtime，使用简化版的AI分析
            self.simple_analyze = lambda frame: (0.5, 0.5)  # 返回默认值

    def analyze_frame(self, frame):
        """评估画面特征"""
        if self.model is None:
            if hasattr(self, 'simple_analyze'):
                return self.simple_analyze(frame)
            return 0.5, 0.5  # 默认值
        
        # 预处理帧数据
        input_data = cv2.resize(frame, (128, 128)).astype(np.float32) / 255.0
        self.model.set_tensor(0, [input_data])
        self.model.invoke()
        complexity, motion = self.model.get_tensor(1)[0]
        return complexity, motion

    def auto_tune(self):
        """动态调整参数"""
        if time.time() - self.last_tune_time < auto_tune_interval:
            return
        
        # 智能参数计算
        new_fps = int(max_fps_limit * (1 - motion_intensity*0.3))
        new_threshold = int(30 + (1 - frame_complexity)*20)
        
        # 应用AI建议
        globals().update(
            frame_rate=max(min_fps_limit, new_fps),
            threshold=new_threshold,
            resolution_scale=max(0.3, 1 - motion_intensity*0.4)
        )
        self.last_tune_time = time.time()

def toggle_ai_optimizer():
    if ai_optimizer_enabled:
        ai_optimizer = AIOptimizer()
        threading.Thread(target=ai_optimizer.auto_tune, daemon=True).start()

# ===  多级帧缓存系统 ===

class FrameCache:
    def __init__(self):
        self.cache = []
        self.lock = threading.Lock()

    def add_frame(self, frame):
        if enable_cache_compression:
            frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]
        with self.lock:
            if len(self.cache) >= frame_cache_size:
                self.cache.pop(0)
            self.cache.append(frame)

    def get_latest(self):
        with self.lock:
            if not self.cache:
                return None
            frame = self.cache[-1]
            return cv2.imdecode(frame, 1) if enable_cache_compression else frame

frame_cache = FrameCache()

# ==================== 主程序 ====================

# 全局变量
version = "v70.2.30"  # 版本
author = "杜玛"
copyrigh = "Copyright © 杜玛. All rights reserved."
threshold = 30  # 变化检测阈值
min_contour_area = 500  # 最小变化区域
monitoring_process = None  # 当前监控的进程
monitoring_camera = None  # 当前监控的摄像头
sct = mss()  # 屏幕截图对象
frame_rate = 30  # 帧率
lock_aspect_ratio = True  # 是否锁定宽高比
monitor_area_roi = None  # 监控区域的ROI
font_path = "msyh.ttf"
max_recommended_fps = 30
performance_test_done = False
last_update_time = time.time()  # 用于计算实际帧率
actual_fps = 0          # 记录实际帧率
cpu_monitor_enabled = True  # 是否启用CPU监控
frame_count = 0  # 用于计数帧数
last_boxes = []  # 存储上一帧的检测框
fade_effect_enabled = False  # 是否启用渐隐效果
fade_frames = 10  # 渐隐帧数，控制渐隐速度
fade_boxes = []  # 存储渐隐框的列表
use_multithread = True  # 是否使用多线程处理
acceleration_manager = AccelerationManager()
frame_processor = FrameProcessor()  # 多线程处理器
# ==== 性能调节变量 ====
adaptive_fps_enabled = True  # 是否启用自适应帧率
max_fps_limit = 60          # 帧率上限
min_fps_limit = 5           # 帧率下限
performance_mode = "balanced"  # 性能模式：performance/balanced/quality
dynamic_threshold_enabled = False  # 动态阈值调节
current_load_factor = 1.0    # 当前负载系数（用于动态调节）
# === 进阶性能优化变量 === 进阶优化方案 ===
frame_skip_enabled = True     # 是否启用跳帧处理
max_skip_frames = 2           # 最大跳帧数
resolution_scale = 1.0        # 分辨率缩放因子 (0.1-1.0)
enable_hardware_accel = True  # 强制硬件加速
background_priority = False   # 后台优先级模式
motion_sensitivity = 0.7      # 运动敏感度 (0-1)
# ==================== AI智能调节系统 ====================
ai_optimizer_enabled = True          # 启用AI动态优化
auto_tune_interval = 30             # 自动调参间隔(秒)
frame_complexity = 0.5              # 画面复杂度评估(0-1)
motion_intensity = 0.5              # 运动强度评估(0-1)
ai_model = None                     # 加载的轻量级AI模型

# 多级缓存配置
frame_cache_size = 3                # 帧缓存数量
enable_cache_compression = True     # 启用缓存压缩

# === 硬件级加速优化 ===

def init_directx_acceleration():
    """Windows平台DirectX加速"""
    if platform.system() == "Windows":
        try:
            import dxgi
            dxgi.create_d3d11_device()
            cv2.ocl.setUseOpenCL(True)  # 强制启用DX11+OpenCL互操作
            return True
        except:
            return False
    return False

# 在程序初始化时调用
directx_enabled = init_directx_acceleration()


def adjust_performance_settings():
    """根据当前负载动态调节参数"""
    global threshold, frame_rate, current_load_factor
    
    if not adaptive_fps_enabled:
        return
    
    # 获取当前CPU/GPU负载
    cpu_load = psutil.cpu_percent() / 100
    mem_load = psutil.virtual_memory().percent / 100
    current_load = max(cpu_load, mem_load)
    
    # 计算负载系数（0.5-1.5范围）
    current_load_factor = 0.5 + current_load
    
    # 根据性能模式调节
    if performance_mode == "performance":
        frame_rate = int(max_fps_limit * (1.8 - current_load_factor))
        if dynamic_threshold_enabled:
            threshold = min(100, int(30 * current_load_factor))
    elif performance_mode == "quality":
        frame_rate = max(min_fps_limit, int(max_fps_limit * (1.2 - current_load_factor/2)))
    else:  # balanced
        frame_rate = int(max_fps_limit * (1.5 - current_load_factor))
    
    # 确保在合理范围内
    frame_rate = max(min_fps_limit, min(max_fps_limit, frame_rate))
    threshold = max(5, min(100, threshold))

# 创建主窗口
root = tk.Tk()
root.after(100, lambda: (
    detect_max_fps(),
    status_bar.config(text=f"系统检测: 推荐最大帧率 {max_recommended_fps}FPS | 加速方式: {acceleration_manager.get_current_backend().name}")
))
root.title(f"智能变化检测系统 ({version} | {author} | {copyrigh})")
root.geometry("500x800")  # 设置窗口大小
root.minsize(100, 100)  # 设置最小窗口大小

# 创建主框架
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# 创建视频显示区域
video_frame = tk.Frame(main_frame, bg='black')
video_frame.pack(fill=tk.BOTH, expand=True)

# 在视频显示区域中添加画布
canvas = tk.Canvas(video_frame, bg='black')
canvas.pack(fill=tk.BOTH, expand=True)

# 创建控制面板区域
control_panel_frame = tk.Frame(main_frame, bd=2, relief=tk.RAISED)
control_panel_frame.pack(fill=tk.X)

# 控制面板内容框架
control_panel = tk.Frame(control_panel_frame)
control_panel.pack(fill=tk.X, padx=5, pady=5)

# 控制面板开关按钮
toggle_button = tk.Button(
    main_frame, 
    text="▲ 显示控制面板 ▲", 
    command=lambda: toggle_control_panel(),
    relief=tk.RAISED,
    bd=1
)
toggle_button.pack(fill=tk.X, side=tk.BOTTOM)

# 状态栏
status_bar = tk.Label(root, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(fill=tk.X, side=tk.BOTTOM)

# 用于存储当前显示的图像引用
current_image = None

# 控制面板初始状态
control_panel_visible = True

def toggle_control_panel():
    global control_panel_visible
    if control_panel_visible:
        control_panel_frame.pack_forget()
        toggle_button.config(text="▼ 显示控制面板 ▼")
        control_panel_visible = False
    else:
        control_panel_frame.pack(fill=tk.X, before=toggle_button)
        toggle_button.config(text="▲ 隐藏控制面板 ▲")
        control_panel_visible = True

# 获取所有可见窗口的进程列表
def get_process_windows():
    process_list = []
    
    def callback(hwnd, hwnd_list):
        if win32gui.IsWindowVisible(hwnd):
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                process = psutil.Process(pid)
                window_title = win32gui.GetWindowText(hwnd)
                if window_title:  # 只包含有标题的窗口
                    hwnd_list.append({
                        'hwnd': hwnd,
                        'pid': pid,
                        'title': window_title,
                        'name': process.name()
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return True
    
    win32gui.EnumWindows(callback, process_list)
    return process_list

# 获取可用的摄像头列表
def get_available_cameras(max_test=5):
    cameras = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras

# 获取窗口的矩形区域
def get_window_rect(hwnd):
    rect = win32gui.GetWindowRect(hwnd)
    return {
        "left": rect[0],
        "top": rect[1],
        "width": rect[2] - rect[0],
        "height": rect[3] - rect[1]
    }

# === 进阶方案Start ===

# ==================== 新增优化工具函数 ====================
def set_process_priority(enable):
    """设置进程优先级（Windows/Mac/Linux兼容方案）"""
    try:
        if platform.system() == "Windows":
            import win32api, win32process, win32con
            priority = win32process.IDLE_PRIORITY_CLASS if enable else win32process.NORMAL_PRIORITY_CLASS
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, os.getpid())
            win32process.SetPriorityClass(handle, priority)
        elif platform.system() == "Linux":
            os.nice(19 if enable else 0)
    except Exception as e:
        print(f"优先级设置失败: {e}")

def apply_resolution_scale(frame):
    """动态调整分辨率"""
    if resolution_scale < 0.99:
        h, w = frame.shape[:2]
        return cv2.resize(frame, (int(w*resolution_scale), int(h*resolution_scale)))
    return frame

def should_skip_frame():
    """智能跳帧决策"""
    if not frame_skip_enabled:
        return False
    # 基于运动检测的跳帧逻辑（当画面静止时跳过更多帧）
    global last_frame_diff
    if 'last_frame_diff' not in globals():
        last_frame_diff = 0
    skip = min(max_skip_frames, int(last_frame_diff * (1-motion_sensitivity) * 3))
    return skip > 0

# === 进阶方案End ===


# 创建进程选择窗口
def create_process_selection_window():
    selection_window = tk.Toplevel(root)
    selection_window.title(f"选择需要监控的程序 ({version} | {author} | {copyrigh})")
    selection_window.geometry("750x500")
    
    # 创建搜索框
    search_frame = tk.Frame(selection_window)
    search_frame.pack(fill=tk.X, padx=10, pady=5)
    
    search_label = tk.Label(search_frame, text="搜索:")
    search_label.pack(side=tk.LEFT)
    
    search_var = tk.StringVar()
    search_entry = tk.Entry(search_frame, textvariable=search_var)
    search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # 填充进程列表的函数
    def update_process_list(treeview):
        processes = get_process_windows()
        treeview.delete(*treeview.get_children())
        
        for process in processes:
            treeview.insert('', 'end', values=(process['pid'], process['name'], process['title']))
    
    refresh_button = tk.Button(search_frame, text="刷新", command=lambda: update_process_list(tree))
    refresh_button.pack(side=tk.RIGHT, padx=5)
    
    # 创建进程列表
    tree = ttk.Treeview(selection_window, columns=('pid', 'name', 'title'), show='headings')
    tree.heading('pid', text='PID', command=lambda: sort_treeview(tree, 'pid', False))
    tree.heading('name', text='进程名', command=lambda: sort_treeview(tree, 'name', False))
    tree.heading('title', text='窗口标题', command=lambda: sort_treeview(tree, 'title', False))
    tree.column('pid', width=80, anchor='center')
    tree.column('name', width=150, anchor='center')
    tree.column('title', width=350, anchor='w')
    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    # 添加滚动条
    scrollbar = ttk.Scrollbar(selection_window, orient="vertical", command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # 初始填充列表
    update_process_list(tree)
    
    # 搜索功能
    def on_search(*args):
        query = search_var.get().lower()
        processes = get_process_windows()
        tree.delete(*tree.get_children())
        for process in processes:
            if (query in str(process['pid']).lower() or 
                query in process['name'].lower() or 
                query in process['title'].lower()):
                tree.insert('', 'end', values=(process['pid'], process['name'], process['title']))
    
    search_var.trace("w", on_search)
    
    # 排序功能
    def sort_treeview(treeview, col, reverse):
        data = [(treeview.set(child, col), child) for child in treeview.get_children('')]
        
        # 尝试转换为数字排序
        try:
            data.sort(key=lambda x: int(x[0]), reverse=reverse)
        except ValueError:
            data.sort(reverse=reverse)
            
        for index, (val, child) in enumerate(data):
            treeview.move(child, '', index)
        
        treeview.heading(col, command=lambda: sort_treeview(treeview, col, not reverse))
    
    # 确定按钮
    def on_select():
        selected_item = tree.focus()
        if not selected_item:
            messagebox.showwarning("警告", "请先选择一个进程")
            return
        
        item_data = tree.item(selected_item)
        global monitoring_process, monitoring_camera
        monitoring_process = {
            'hwnd': next(p['hwnd'] for p in get_process_windows() if p['pid'] == int(item_data['values'][0])),
            'pid': int(item_data['values'][0]),
            'name': item_data['values'][1],
            'title': item_data['values'][2]
        }
        monitoring_camera = None  # 清除摄像头监控
        
        # 更新按钮状态
        process_button.config(text=f"停止监控 {monitoring_process['name']}")
        camera_button.config(text="选择镜头监控")
        
        selection_window.destroy()
        status_bar.config(text=f"已开始监控进程: {monitoring_process['name']}")
    
    # 取消按钮
    def on_cancel():
        selection_window.destroy()
    
    button_frame = tk.Frame(selection_window)
    button_frame.pack(pady=10)
    
    select_button = tk.Button(button_frame, text="确定", command=on_select)
    select_button.pack(side=tk.LEFT, padx=10)
    
    cancel_button = tk.Button(button_frame, text="取消", command=on_cancel)
    cancel_button.pack(side=tk.LEFT, padx=10)
    
    # 绑定双击事件
    def on_double_click(event):
        on_select()
    
    tree.bind("<Double-1>", on_double_click)

# 创建摄像头选择窗口
def create_camera_selection_window():
    selection_window = tk.Toplevel(root)
    selection_window.title(f"选择监控镜头 ({version} | {author} | {copyrigh})")
    selection_window.geometry("400x300")
    
    # 获取可用摄像头
    cameras = get_available_cameras()
    if not cameras:
        status_bar.config(text="没有检测到可用的摄像头")
        selection_window.destroy()
        return
    
    # 创建摄像头列表
    tree = ttk.Treeview(selection_window, columns=('id', 'status'), show='headings')
    tree.heading('id', text='摄像头ID')
    tree.heading('status', text='状态')
    tree.column('id', width=100)
    tree.column('status', width=200)
    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 填充摄像头列表
    for cam_id in cameras:
        tree.insert('', 'end', values=(cam_id, "可用"))
    
    # 确定按钮
    def on_select():
        selected_item = tree.focus()
        if not selected_item:
            status_bar.config(text="请先选择一个摄像头")
            return
        
        item_data = tree.item(selected_item)
        global monitoring_camera, monitoring_process
        monitoring_camera = int(item_data['values'][0])
        monitoring_process = None  # 清除进程监控
        
        # 更新按钮状态
        camera_button.config(text=f"停止监控 摄像头{monitoring_camera}")
        process_button.config(text="选择监控进程")
        
        selection_window.destroy()
        status_bar.config(text=f"已开始监控摄像头: {monitoring_camera}")
    
    # 取消按钮
    def on_cancel():
        selection_window.destroy()
    
    button_frame = tk.Frame(selection_window)
    button_frame.pack(pady=10)
    
    select_button = tk.Button(button_frame, text="确定", command=on_select)
    select_button.pack(side=tk.LEFT, padx=10)
    
    cancel_button = tk.Button(button_frame, text="取消", command=on_cancel)
    cancel_button.pack(side=tk.LEFT, padx=10)

# 开始/停止监控进程
def toggle_process_monitoring():
    global monitoring_process, monitoring_camera
    
    if monitoring_process:
        # 停止监控
        monitoring_process = None
        process_button.config(text="选择监控进程")
        status_bar.config(text="已停止监控")
    else:
        # 开始监控 - 打开进程选择窗口
        monitoring_camera = None  # 清除摄像头监控
        camera_button.config(text="选择镜头监控")
        create_process_selection_window()

# 开始/停止监控摄像头
def toggle_camera_monitoring():
    global monitoring_camera, monitoring_process
    
    if monitoring_camera is not None:
        # 停止监控
        monitoring_camera = None
        camera_button.config(text="选择镜头监控")
        status_bar.config(text="已停止监控")
    else:
        # 开始监控 - 打开摄像头选择窗口
        monitoring_process = None  # 清除进程监控
        process_button.config(text="选择监控进程")
        create_camera_selection_window()

def select_monitor_area():
    global monitor_area_roi
    
    if not monitoring_process:
        messagebox.showwarning("警告", "请先选择要监控的进程")
        return
    
    # 获取窗口句柄
    hwnd = monitoring_process['hwnd']
    
    # 创建临时窗口用于选择区域
    temp_window = tk.Toplevel(root)
    temp_window.title("请在目标窗口上选择监控区域")
    temp_window.geometry("300x100")
    
    # 添加说明标签
    label = tk.Label(temp_window, text="请在目标窗口上拖动鼠标选择监控区域\n释放鼠标左键自动确认")
    label.pack(pady=20)
    
    # 设置窗口置顶
    temp_window.attributes('-topmost', True)
    
    # 获取窗口矩形
    rect = win32gui.GetWindowRect(hwnd)
    
    # 创建全屏透明窗口用于选择
    selection_window = tk.Toplevel(temp_window)
    selection_window.overrideredirect(True)
    selection_window.geometry(f"{rect[2]-rect[0]}x{rect[3]-rect[1]}+{rect[0]}+{rect[1]}")
    selection_window.attributes('-alpha', 0.3)
    selection_window.attributes('-topmost', True)
    
    # 选择区域变量
    start_x, start_y = 0, 0
    end_x, end_y = 0, 0
    rect_id = None
    
    def on_mouse_down(event):
        nonlocal start_x, start_y, rect_id
        start_x, start_y = event.x, event.y
        if rect_id:
            canvas.delete(rect_id)
        rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2)
    
    def on_mouse_move(event):
        nonlocal end_x, end_y, rect_id
        end_x, end_y = event.x, event.y
        if rect_id:
            canvas.coords(rect_id, start_x, start_y, end_x, end_y)
    
    def on_mouse_up(event):
        nonlocal end_x, end_y
        end_x, end_y = event.x, event.y
    
        # 鼠标释放后自动确认选择
        global monitor_area_roi
        x1, y1 = min(start_x, end_x), min(start_y, end_y)
        x2, y2 = max(start_x, end_x), max(start_y, end_y)
        
        # 确保选择的区域有效
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # 最小10像素的宽度和高度
            monitor_area_roi = (x1, y1, x2-x1, y2-y1)
            status_bar.config(text=f"已设置监控区域: {monitor_area_roi}")
        else:
            status_bar.config(text="选择区域太小，请重新选择")
            return
        
        temp_window.destroy()
        selection_window.destroy()
    
    # 创建画布
    canvas = tk.Canvas(selection_window, cursor="cross")
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # 绑定事件
    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)
    
    # 确保画布可以接收键盘输入
    canvas.focus_set()

# 更新视频显示
def update_video_display(frame):
    global current_image
    
    # 将OpenCV图像转换为PIL格式
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    
    # 计算缩放比例以保持宽高比
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    
    if canvas_width <= 1 or canvas_height <= 1:
        return
    
    img_width, img_height = img.size
    
    if lock_aspect_ratio:
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
    else:
        new_width = canvas_width
        new_height = canvas_height
    
    # 缩放图像
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # 转换为Tkinter PhotoImage
    current_image = ImageTk.PhotoImage(image=img)
    
    # 清除画布并显示新图像
    canvas.delete("all")
    canvas.create_image(
        (canvas_width - new_width) // 2,
        (canvas_height - new_height) // 2,
        anchor=tk.NW,
        image=current_image
    )

# 性能检测函数
def detect_max_fps():
    global max_recommended_fps, performance_test_done
    test_duration = 2  # 测试2秒
    start_time = time.time()
    frame_count = 0
    
    # 模拟实际工作负载
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    backend = acceleration_manager.get_current_backend()
    
    while time.time() - start_time < test_duration:
        # 使用当前加速后端处理
        _ = backend.process_frames(test_frame, test_frame, threshold)
        frame_count += 1
    
    max_recommended_fps = min(360, max(30, frame_count // test_duration))
    performance_test_done = True
    return max_recommended_fps

# 添加帧时间记录
frame_times = collections.deque(maxlen=60)  # 记录最近60帧时间

# 定义更新帧的函数
def update_frame():
    global frame_count, last_frame_diff, frame_complexity, motion_intensity, last_update_time, actual_fps
    global ai_optimizer_enabled

    frame_count += 1
    
    # 初始化时间变量
    current_time = time.time()
    if last_update_time is None:
        last_update_time = current_time
    
    # 计算实际FPS
    time_diff = current_time - last_update_time
    if time_diff > 0:
        actual_fps = 0.9 * actual_fps + 0.1 * (1 / time_diff)
    last_update_time = current_time
    
    # 1. 获取原始帧（优先使用缓存）
    raw_frame = None
    if frame_cache_size > 0:
        raw_frame = frame_cache.get_latest()
    
    if raw_frame is None:
        if monitoring_camera is not None:
            cap = cv2.VideoCapture(monitoring_camera)
            ret, raw_frame = cap.read()
            cap.release()
            if not ret:
                status_bar.config(text="摄像头读取失败")
                root.after(30, update_frame)
                return
        elif monitoring_process:
            try:
                rect = get_window_rect(monitoring_process['hwnd'])
                monitor_area = {
                    "left": rect["left"],
                    "top": rect["top"],
                    "width": rect["width"],
                    "height": rect["height"]
                }
                if monitor_area_roi:
                    x, y, w, h = monitor_area_roi
                    monitor_area["left"] += x
                    monitor_area["top"] += y
                    monitor_area["width"] = w
                    monitor_area["height"] = h
                
                raw_frame = np.array(sct.grab(monitor_area))
                raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2BGR)
                frame_cache.add_frame(raw_frame.copy())
            except Exception as e:
                status_bar.config(text=f"截屏失败: {str(e)}", fg="red")
                root.after(1000, update_frame)
                return
        else:
            # 无信号时的黑屏提示
            raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(raw_frame, "未选择监控源", (150, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    # 2. AI实时分析（确保raw_frame已获取）
    global ai_optimizer_enabled
    if ai_optimizer_enabled and frame_count % 15 == 0:
        if 'ai_optimizer' not in globals():
            globals()['ai_optimizer'] = AIOptimizer()
        try:
            frame_complexity, motion_intensity = ai_optimizer.analyze_frame(raw_frame)
            if ai_optimizer_enabled:
                ai_optimizer.auto_tune()
        except ImportError as e:
            print(f"AI模块导入失败: {e}\n请尝试安装 tflite_runtime: pip install tflite-runtime")
            ai_optimizer_enabled = False
            status_bar.config(text="AI优化需要安装 tflite_runtime 库", fg="red")
        except Exception as e:
            print(f"AI分析失败: {e}")
            # 临时禁用AI优化以避免持续报错
            ai_optimizer_enabled = False
            status_bar.config(text="AI优化已临时禁用")

    # 3. 智能跳帧决策
    if should_skip_frame() and frame_count % max_skip_frames != 0:
        root.after(1, update_frame)  # 快速跳过帧
        return
    if frame_cache_size > 0:
        raw_frame = frame_cache.get_latest()
    
    if raw_frame is None:
        if monitoring_camera is not None:
            cap = cv2.VideoCapture(monitoring_camera)
            ret, raw_frame = cap.read()
            cap.release()
            if not ret:
                status_bar.config(text="摄像头读取失败")
                root.after(30, update_frame)
                return
        elif monitoring_process:
            try:
                rect = get_window_rect(monitoring_process['hwnd'])
                monitor_area = {
                    "left": rect["left"],
                    "top": rect["top"],
                    "width": rect["width"],
                    "height": rect["height"]
                }
                if monitor_area_roi:
                    x, y, w, h = monitor_area_roi
                    monitor_area["left"] += x
                    monitor_area["top"] += y
                    monitor_area["width"] = w
                    monitor_area["height"] = h
                
                raw_frame = np.array(sct.grab(monitor_area))
                raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2BGR)
                frame_cache.add_frame(raw_frame.copy())
            except Exception as e:
                status_bar.config(text=f"截屏失败: {str(e)}", fg="red")
                root.after(1000, update_frame)
                return
        else:
            # 无信号时的黑屏提示
            raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(raw_frame, "未选择监控源", (150, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    # 4. 动态分辨率处理
    work_frame = apply_resolution_scale(raw_frame)
    
    # 5. 运动敏感度处理
    if motion_sensitivity < 0.5:
        work_frame = cv2.GaussianBlur(work_frame, (3,3), 0)
    
    # 6. 硬件加速路径
    if directx_enabled:
        work_frame = cv2.UMat(work_frame)

    # 7. 变化检测核心逻辑
    backend = acceleration_manager.get_current_backend()
    next_frame = get_next_frame()  # 获取下一帧
    
    if next_frame is None:  # 如果没有获取到下一帧
        root.after(30, update_frame)  # 稍后重试
        return
    
    start_time = time.time()
    if use_multithread:
        thresh = frame_processor.process_frames(work_frame, next_frame, threshold, backend)
    else:
        thresh = backend.process_frames(work_frame, next_frame, threshold)
    proc_time = (time.time() - start_time) * 1000
    
    # 8. 动态阈值补偿
    if dynamic_threshold_enabled:
        adj_threshold = min(100, int(threshold * (1 + motion_intensity)))
        _, thresh = cv2.threshold(thresh, adj_threshold, 255, cv2.THRESH_BINARY)

    # 9. 轮廓检测与绘制
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    display_frame = raw_frame.copy()
    
    # 绘制带渐隐效果的检测框
    current_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            current_boxes.append((x, y, w, h))
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 渐隐效果处理
    if fade_effect_enabled:
        fade_boxes[:] = [(box, fade_frames) for box in current_boxes] + \
                       [(box, count-1) for box, count in fade_boxes if count > 1]
        
        for box, count in fade_boxes:
            if box not in current_boxes:
                alpha = count / fade_frames
                color = (0, int(255*alpha), int(255*(1-alpha)))
                cv2.rectangle(display_frame, 
                            (box[0], box[1]), 
                            (box[0]+box[2], box[1]+box[3]), 
                            color, 
                            1 + int(2*alpha))

    # 10. 更新显示
    update_video_display(display_frame)
    
    # 11. 性能监控与状态更新
    current_time = time.time()
    if last_update_time:
        actual_fps = 0.9 * actual_fps + 0.1 * (1 / (current_time - last_update_time))
    last_update_time = current_time
    
    # 状态栏信息
    status_text = (
        f"FPS: {actual_fps:.1f}/{frame_rate} | "
        f"负载: {current_load_factor*100:.0f}% | "
        f"模式: {performance_mode.upper()}"
    )
    if ai_optimizer_enabled:
        status_text += f" | AI: 运动{motion_intensity*100:.0f}%"
    status_bar.config(text=status_text)

    # 12. 自适应帧率控制
    effective_fps = min(frame_rate, max_fps_limit) if adaptive_fps_enabled else frame_rate
    delay = max(1, int(1000 / (effective_fps * (1 + 0.5 * (max_skip_frames - 1)))))
    root.after(delay, update_frame)


def get_next_frame():
    """获取用于比较的下一帧"""
    if monitoring_camera is not None:
        cap = cv2.VideoCapture(monitoring_camera)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)  # 跳转到下一帧
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    elif monitoring_process is not None:
        return np.array(sct.grab(get_window_rect(monitoring_process['hwnd'])))
    else:
        return None

def create_control_panel():
    global frame_rate_scale, threshold_scale, min_area_scale
    global frame_rate_entry, threshold_entry, min_area_entry
    
    # 统一间距设置（更紧凑）
    padx_val = 2  # 从3减小到2
    pady_val = 1  # 从2减小到1
    
    # ===== 1. 顶部固定区域（不可折叠）=====
    top_frame = tk.Frame(control_panel)
    top_frame.pack(fill=tk.X, pady=(0,3))
    
    # 监控源选择按钮（紧凑排列）
    btn_frame = tk.Frame(top_frame)
    btn_frame.pack(fill=tk.X)
    
    buttons = [
        ("进程", toggle_process_monitoring),
        ("摄像头", toggle_camera_monitoring),
        ("区域", select_monitor_area)
    ]
    
    for col, (text, cmd) in enumerate(buttons):
        btn = tk.Button(btn_frame, text=text, command=cmd, width=6)
        btn.pack(side=tk.LEFT, padx=1)
        if text == "进程":
            global process_button
            process_button = btn
        elif text == "摄像头":
            global camera_button
            camera_button = btn
    
    # 核心参数快速调节（滑块紧凑版）
    param_frame = tk.Frame(top_frame)
    param_frame.pack(fill=tk.X)
    
    # 帧率控制（合并标签和滑块）
    tk.Label(param_frame, text="FPS:").pack(side=tk.LEFT)
    frame_rate_scale = tk.Scale(
        param_frame, 
        from_=1, 
        to=max_recommended_fps if performance_test_done else 360,
        orient=tk.HORIZONTAL,
        length=80,
        showvalue=1
    )
    frame_rate_scale.set(frame_rate)
    frame_rate_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    # 阈值控制
    tk.Label(param_frame, text="阈值:").pack(side=tk.LEFT, padx=(5,0))
    threshold_scale = tk.Scale(
        param_frame,
        from_=1,
        to=100,
        orient=tk.HORIZONTAL,
        length=80,
        showvalue=1
    )
    threshold_scale.set(threshold)
    threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    # ===== 2. 可折叠区域 =====
    # 使用ttk的Labelframe实现更好看的折叠效果
    collapsible_frame = ttk.Labelframe(control_panel, text="⚙️ 高级设置")
    collapsible_frame.pack(fill=tk.X, pady=(0,5))
    
    # 内部使用Notebook实现标签页
    notebook = ttk.Notebook(collapsible_frame)
    notebook.pack(fill=tk.BOTH, expand=True)
    
    # 标签页1：检测设置
    detect_tab = ttk.Frame(notebook)
    notebook.add(detect_tab, text="检测")
    
    # 最小区域设置
    tk.Label(detect_tab, text="最小区域:").grid(row=0, column=0, sticky="e", padx=padx_val)
    min_area_scale = tk.Scale(
        detect_tab,
        from_=1,
        to=1000,
        orient=tk.HORIZONTAL,
        length=120
    )
    min_area_scale.set(min_contour_area)
    min_area_scale.grid(row=0, column=1, sticky="ew", padx=padx_val)
    
    # 动态阈值
    tk.Checkbutton(
        detect_tab,
        text="动态阈值",
        variable=tk.BooleanVar(value=dynamic_threshold_enabled),
        command=lambda: globals().update(dynamic_threshold_enabled=not dynamic_threshold_enabled)
    ).grid(row=1, column=0, columnspan=2, sticky="w", padx=padx_val)
    
    # 渐隐效果
    fade_frame = tk.Frame(detect_tab)
    fade_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
    tk.Checkbutton(
        fade_frame,
        text="渐隐",
        variable=tk.BooleanVar(value=fade_effect_enabled),
        command=lambda: globals().update(fade_effect_enabled=not fade_effect_enabled)
    ).pack(side=tk.LEFT)
    tk.Label(fade_frame, text="速度:").pack(side=tk.LEFT)
    tk.Scale(
        fade_frame,
        from_=1,
        to=30,
        orient=tk.HORIZONTAL,
        length=80
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    # 标签页2：性能设置
    perf_tab = ttk.Frame(notebook)
    notebook.add(perf_tab, text="性能")
    
    # 性能模式
    tk.Label(perf_tab, text="模式:").grid(row=0, column=0, sticky="e")
    ttk.Combobox(
        perf_tab,
        values=["性能", "平衡", "质量"],
        state="readonly",
        width=8
    ).grid(row=0, column=1, sticky="w")
    
    # 自适应帧率
    tk.Checkbutton(
        perf_tab,
        text="自适应FPS",
        variable=tk.BooleanVar(value=adaptive_fps_enabled),
        command=lambda: globals().update(adaptive_fps_enabled=not adaptive_fps_enabled)
    ).grid(row=1, column=0, columnspan=2, sticky="w")
    
    # 跳帧设置
    jump_frame = tk.Frame(perf_tab)
    jump_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
    tk.Checkbutton(
        jump_frame,
        text="跳帧",
        variable=tk.BooleanVar(value=frame_skip_enabled),
        command=lambda: globals().update(frame_skip_enabled=not frame_skip_enabled)
    ).pack(side=tk.LEFT)
    tk.Label(jump_frame, text="最大:").pack(side=tk.LEFT)
    ttk.Combobox(
        jump_frame,
        values=[1, 2, 3],
        width=3
    ).pack(side=tk.LEFT)
    
    # 标签页3：AI优化
    ai_tab = ttk.Frame(notebook)
    notebook.add(ai_tab, text="AI")
    
    # AI开关
    tk.Checkbutton(
        ai_tab,
        text="启用AI优化",
        variable=tk.BooleanVar(value=ai_optimizer_enabled),
        command=lambda: toggle_ai_optimizer()
    ).grid(row=0, column=0, columnspan=2, sticky="w")
    
    # 调参间隔
    tk.Label(ai_tab, text="调参间隔:").grid(row=1, column=0, sticky="e")
    tk.Scale(
        ai_tab,
        from_=10,
        to=120,
        orient=tk.HORIZONTAL,
        length=100
    ).grid(row=1, column=1, sticky="ew")
    
    # ===== 3. 底部状态栏 =====
    status_frame = tk.Frame(control_panel)
    status_frame.pack(fill=tk.X, pady=(0,3))
    
    # 加速后端显示
    backend_var = tk.StringVar(value=f"加速:{acceleration_manager.get_current_backend().name}")
    tk.Label(status_frame, textvariable=backend_var).pack(side=tk.LEFT)
    
    # AI状态
    ai_status_var = tk.StringVar(value="AI:待机")
    tk.Label(status_frame, textvariable=ai_status_var).pack(side=tk.RIGHT)
    
    # 配置列权重
    control_panel.columnconfigure(0, weight=1)
    for tab in [detect_tab, perf_tab, ai_tab]:
        tab.columnconfigure(1, weight=1)

# 创建控制面板
create_control_panel()
# 启动多线程处理器
frame_processor.start_processing(threshold, acceleration_manager.get_current_backend())

# 启动更新帧的函数
update_frame()

# 运行主循环
root.mainloop()

# 释放资源
if monitoring_camera is not None:
    cv2.VideoCapture(monitoring_camera).release()
acceleration_manager.release_all()
frame_processor.stop_processing()