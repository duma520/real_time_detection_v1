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

# ==================== 算法优化模块 ====================
class AlgorithmManager:
    """算法管理器，提供多种处理算法和组合"""
    
    def __init__(self):
        self.current_algorithm = "原始设置"  # 默认使用原始算法
        self.available_algorithms = {
            "原始设置": self.original_algorithm,
            "高斯模糊预处理": self.gaussian_blur_algorithm,
            "背景减除算法": self.background_subtraction_algorithm,
            "形态学处理": self.morphological_algorithm,
            "光流法优化": self.optical_flow_algorithm,
            "帧差分优化": self.frame_diff_enhanced_algorithm,
            "多尺度检测": self.multi_scale_algorithm
        }
        
        # 预定义的算法组合
        self.algorithm_presets = {
            "快速检测组合": ["高斯模糊预处理", "帧差分优化"],
            "精确检测组合": ["背景减除算法", "形态学处理"],
            "运动追踪组合": ["光流法优化", "多尺度检测"],
            "智能组合": []  # 动态选择
        }
        
        # 背景减除器
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        
        # 光流法参数
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # 智能组合缓存
        self.smart_combo_cache = None
        self.last_combo_time = 0
    
    def set_algorithm(self, algorithm_name: str):
        """设置当前使用的算法"""
        if algorithm_name in self.available_algorithms:
            self.current_algorithm = algorithm_name
            return True
        return False
    
    def set_algorithm_combo(self, combo_name: str):
        """设置算法组合"""
        if combo_name in self.algorithm_presets:
            self.current_algorithm = combo_name
            return True
        return False
    
    def smart_select_algorithm(self):
        """智能选择最佳算法组合"""
        current_time = time.time()
        if self.smart_combo_cache and current_time - self.last_combo_time < 10:
            return self.smart_combo_cache
        
        # 根据系统性能动态选择
        cpu_load = psutil.cpu_percent()
        mem_load = psutil.virtual_memory().percent
        
        if cpu_load > 70 or mem_load > 80:
            combo = ["高斯模糊预处理", "帧差分优化"]  # 轻量级组合
        else:
            combo = ["背景减除算法", "形态学处理"]  # 精确组合
        
        self.smart_combo_cache = combo
        self.last_combo_time = current_time
        return combo
    
    def process_frame(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """处理帧的核心方法"""
        if self.current_algorithm == "智能组合":
            combo = self.smart_select_algorithm()
            result = frame1.copy()
            for algo in combo:
                result = self.available_algorithms[algo](result, frame2, threshold)
            return result
        elif self.current_algorithm in self.algorithm_presets:
            combo = self.algorithm_presets[self.current_algorithm]
            result = frame1.copy()
            for algo in combo:
                result = self.available_algorithms[algo](result, frame2, threshold)
            return result
        else:
            return self.available_algorithms[self.current_algorithm](frame1, frame2, threshold)
    
    # ===== 基础算法实现 =====
    def original_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """原始算法"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def gaussian_blur_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """高斯模糊预处理"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
        
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def background_subtraction_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """背景减除算法"""
        # 使用MOG2背景减除器
        fg_mask = self.backSub.apply(frame2)
        _, thresh = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def morphological_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """形态学处理算法"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return thresh
    
    def optical_flow_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """光流法优化"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 计算密集光流
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **self.flow_params)
        
        # 计算光流幅度
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def frame_diff_enhanced_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """增强型帧差分算法"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 加权差分
        diff = cv2.absdiff(gray1, gray2)
        diff = cv2.multiply(diff, 1.5)  # 增强差异
        
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def multi_scale_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        """多尺度检测算法"""
        # 创建多尺度金字塔
        scales = [1.0, 0.75, 0.5]
        results = []
        
        for scale in scales:
            if scale != 1.0:
                resized1 = cv2.resize(frame1, None, fx=scale, fy=scale)
                resized2 = cv2.resize(frame2, None, fx=scale, fy=scale)
            else:
                resized1 = frame1.copy()
                resized2 = frame2.copy()
                
            gray1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            if scale != 1.0:
                thresh = cv2.resize(thresh, (frame1.shape[1], frame1.shape[0]))
            
            results.append(thresh)
        
        # 合并多尺度结果
        final_thresh = np.zeros_like(results[0])
        for thresh in results:
            final_thresh = cv2.bitwise_or(final_thresh, thresh)
        
        return final_thresh

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

# ==================== 主程序 ====================

# 全局变量
version = "v70.4.7"  # 版本
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
last_update_time = None  # 用于计算实际帧率
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
adaptive_fps_enabled = True  # 是否启用自适应帧率
max_fps_limit = 60          # 帧率上限
min_fps_limit = 5           # 帧率下限
performance_mode = "平衡"  # 性能模式：performance/balanced/quality 性能模式：性能/平衡/画质
dynamic_threshold_enabled = False  # 动态阈值调节
current_load_factor = 1.0    # 当前负载系数（用于动态调节）
log_messages = []  # 存储日志消息
max_log_entries = 100  # 最大日志条目数
last_fps_warning_time = 0  # 上次发出帧率警告的时间
fps_warning_interval = 30   # 帧率警告最小间隔(秒)
algorithm_manager = None  # 算法管理器（稍后初始化）
show_algorithm_debug = False  # 是否显示算法调试信息



# 添加日志记录函数
def log_message(message, level="INFO"):
    """线程安全的日志记录函数"""
    global log_messages
    # 生成日志条目
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    
    # 添加到日志列表
    log_messages.append(log_entry)
    if len(log_messages) > max_log_entries:
        log_messages.pop(0)
    
    # 更新日志显示
    if 'log_text' in globals():
        def _log():
            log_text.config(state=tk.NORMAL)
            log_text.insert(tk.END, log_entry + "\n")
            log_text.see(tk.END)  # 自动滚动到底部
            log_text.config(state=tk.DISABLED)

        # 确保在主线程执行UI更新
        if root and root.winfo_exists():
            root.after(0, _log)

    
    # 同时在状态栏显示重要消息
    if level in ["WARNING", "ERROR"]:
        status_bar.config(text=message, fg="red" if level == "ERROR" else "red")

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
    if performance_mode == "性能":
        frame_rate = int(max_fps_limit * (1.8 - current_load_factor))
        if dynamic_threshold_enabled:
            threshold = min(100, int(30 * current_load_factor))
    elif performance_mode == "画质":
        frame_rate = max(min_fps_limit, int(max_fps_limit * (1.2 - current_load_factor/2)))
    else:  # balanced
        frame_rate = int(max_fps_limit * (1.5 - current_load_factor))
    
    # 确保在合理范围内
    frame_rate = max(min_fps_limit, min(max_fps_limit, frame_rate))
    threshold = max(5, min(100, threshold))

# 创建主窗口
root = tk.Tk()

# 初始化算法管理器
algorithm_manager = AlgorithmManager()

root.after(100, lambda: (
    detect_max_fps(),
    log_message(f"系统检测: 推荐最大帧率 {max_recommended_fps}FPS | 加速方式: {acceleration_manager.get_current_backend().name}")
))
root.title(f"智能变化检测系统 ({version} | {author} | {copyrigh})")
root.geometry("500x850")  # 设置窗口大小
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
control_panel_frame.pack(fill=tk.X, pady=5, ipady=2)

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
    backends = [
        cv2.CAP_DSHOW,  # DirectShow
        cv2.CAP_MSMF,   # Microsoft Media Foundation
        cv2.CAP_ANY     # 自动选择
    ]

    for i in range(max_test):
        for backend in backends:
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    cameras.append(i)
                    cap.release()
                    break  # 找到一个可用的后端就停止尝试
                else:
                    cap.release()
            except:
                continue
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
        log_message(f"已开始监控进程: {monitoring_process['name']}")
    
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
        log_message("没有检测到可用的摄像头")
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
            log_message("请先选择一个摄像头")
            return
        
        item_data = tree.item(selected_item)
        global monitoring_camera, monitoring_process
        monitoring_camera = int(item_data['values'][0])
        monitoring_process = None  # 清除进程监控
        
        # 更新按钮状态
        camera_button.config(text=f"停止监控 摄像头{monitoring_camera}")
        process_button.config(text="选择监控进程")
        
        selection_window.destroy()
        log_message(f"已开始监控摄像头: {monitoring_camera}")
    
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
        log_message("已停止监控")
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
        log_message("已停止监控")
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
            log_message(f"已设置监控区域: {monitor_area_roi}")
        else:
            log_message("选择区域太小，请重新选择")
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

def process_frame_roi(frame):
    """处理指定帧区域的核心函数"""
    global last_boxes, fade_boxes
    
    # 获取两帧用于比较
    frame1 = frame.copy()
    time.sleep(1.0/frame_rate)  # 等待适当时间获取下一帧
    frame2 = get_next_frame()   # 获取下一帧
    
    if frame2 is None or frame1.shape != frame2.shape:
        return frame1
    
    # 使用当前加速后端处理帧
    backend = acceleration_manager.get_current_backend()
    thresh = backend.process_frames(frame1, frame2, threshold)
    
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 更新检测框
    current_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            current_boxes.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 渐隐效果处理
    if fade_effect_enabled:
        fade_boxes = [(box, fade_frames) for box in current_boxes] + \
                    [(box, count-1) for box, count in fade_boxes if count > 1]
        
        for box, count in fade_boxes:
            if box not in current_boxes:
                alpha = count / fade_frames
                color = (0, int(255*alpha), int(255*(1-alpha)))
                cv2.rectangle(frame, 
                             (box[0], box[1]), 
                             (box[0]+box[2], box[1]+box[3]), 
                             color, 
                             1 + int(2*alpha))
    
    last_boxes = current_boxes
    return frame

def get_next_frame():
    """获取下一帧"""
    if monitoring_camera is not None:
        cap = cv2.VideoCapture(monitoring_camera)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    else:
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
        frame = np.array(sct.grab(monitor_area))
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# 定义更新帧的函数
def update_frame():
    global last_boxes, fade_boxes,last_update_time,actual_fps,frame_count,skip_counter
    # 帧率过高警告
    global last_fps_warning_time
    current_time = time.time()
    
    if (frame_rate > 120 or abs(actual_fps - frame_rate) > 10) and \
        (current_time - last_fps_warning_time > fps_warning_interval):
    
        status_text = f"⚠️ 帧率过高: 设置 {frame_rate}FPS, 实际 {actual_fps:.1f}FPS"
        status_bar.config(text=status_text, fg="red")
        last_fps_warning_time = current_time



        # 1. 帧跳过逻辑
    if frame_skip_var.get() > 0:
        if not hasattr(update_frame, 'skip_counter'):
            update_frame.skip_counter = 0
        update_frame.skip_counter += 1
        if update_frame.skip_counter <= frame_skip_var.get():
            root.after(max(1, int(1000/frame_rate)), update_frame)
            return
        update_frame.skip_counter = 0
    
    # 2. 分辨率调节
    if monitoring_camera is not None or monitoring_process:
        try:
            # 获取原始帧
            if monitoring_camera is not None:
                cap = cv2.VideoCapture(monitoring_camera)
                ret, original_frame = cap.read()
                cap.release()
                if not ret:
                    root.after(30, update_frame)
                    return
            else:
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
                original_frame = np.array(sct.grab(monitor_area))
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGRA2BGR)
            
            # 应用分辨率缩放
            scale = resolution_scale_var.get()
            if scale < 1.0:
                working_frame = cv2.resize(original_frame, (0,0), fx=scale, fy=scale)
            else:
                working_frame = original_frame.copy()
            
            # 3. ROI自动跟踪
            if roi_tracking_var.get() and len(last_boxes) > 0:
                # 计算ROI区域
                x_min = min(box[0] for box in last_boxes)
                y_min = min(box[1] for box in last_boxes)
                x_max = max(box[0]+box[2] for box in last_boxes)
                y_max = max(box[1]+box[3] for box in last_boxes)
                
                # 扩大ROI区域20%
                margin_x = int((x_max - x_min) * 0.2)
                margin_y = int((y_max - y_min) * 0.2)
                roi = (
                    max(0, x_min - margin_x),
                    max(0, y_min - margin_y),
                    min(working_frame.shape[1], x_max + margin_x) - max(0, x_min - margin_x),
                    min(working_frame.shape[0], y_max + margin_y) - max(0, y_min - margin_y)
                )
                
                # 只处理ROI区域
                roi_frame = working_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
                processed_roi = process_frame_roi(roi_frame)  # 处理ROI区域
                
                # 将处理结果放回原图
                working_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = processed_roi
            else:
                working_frame = process_frame_roi(working_frame)
            
            # 如果缩放过，将结果放大回原始尺寸
            if scale < 1.0:
                working_frame = cv2.resize(working_frame, (original_frame.shape[1], original_frame.shape[0]))
            
            # 更新显示
            update_video_display(working_frame)
            
            # 4. 动态检测间隔
            if dynamic_interval_var.get():
                if len(last_boxes) == 0:  # 没有检测到变化
                    root.after(max(1, int(2000/frame_rate)), update_frame)  # 降低检测频率
                    return
            
        except Exception as e:
            print(f"优化处理出错: {str(e)}")

    # 1. 安全获取时间戳
    try:
        current_time = time.time()
    except:
        current_time = last_update_time or time.time()
    
    # 2. 计算时间差（带多重保护）
    time_diff = 0.033  # 默认30FPS的间隔
    if last_update_time is not None:
        time_diff = max(0.001, current_time - last_update_time)  # 最小1ms间隔
    
    # 3. 更新帧率计算
    actual_fps = 0.9 * actual_fps + 0.1 * (1 / time_diff)
    last_update_time = current_time
    
    # 4. 帧率显示保护
    if not 0 < actual_fps < 1000:  # 合理范围检查
        actual_fps = 30.0


    # 性能监控

    if frame_count % 10 == 0:
        adjust_performance_settings()

    # 统一性能计时起点
    start_time = time.time()
    frame_count += 1
    current_time = time.time()

    # CPU占用监控（每10帧检测一次）
    if cpu_monitor_enabled and monitoring_process and frame_count % 10 == 0:
        cpu_percent = psutil.cpu_percent(interval=0.1)

        if cpu_percent > 80 and frame_rate > 60:
            log_message(
                text=f"⚠️ CPU过载: {cpu_percent}% | 实际FPS: {actual_fps:.1f}/{frame_rate}",
                fg="red"
            )
    
    # 实际帧率计算
    current_time = time.time()

    # 安全处理帧率计算
    if last_update_time is not None and current_time > last_update_time:
        time_diff = current_time - last_update_time
        actual_fps = 0.9 * actual_fps + 0.1 * (1 / time_diff)  # 平滑处理
    last_update_time = current_time


    if monitoring_camera is not None:
        # 摄像头模式
        try:
            # 尝试多种后端
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            cap = None
            
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(monitoring_camera, backend)
                    if cap.isOpened():
                        break
                except:
                    if cap:
                        cap.release()
                    continue
                    
            if not cap or not cap.isOpened():
                log_message(f"无法打开摄像头 {monitoring_camera}", fg="red")
                if cap:
                    cap.release()
                root.after(1000, update_frame)
                return
                
            # 设置摄像头参数以获得更稳定的帧率
            cap.set(cv2.CAP_PROP_FPS, frame_rate)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # 读取第一帧
            ret, current_frame = cap.read()
            if not ret:
                log_message("无法读取视频帧")
                cap.release()
                root.after(30, update_frame)
                return
            
            # 读取第二帧
            ret, next_frame = cap.read()
            if not ret:
                log_message("无法读取视频帧")
                cap.release()
                root.after(30, update_frame)
                return
            
            # 释放摄像头资源
            cap.release()
            
            # 检查帧尺寸是否有效
            if current_frame is None or next_frame is None:
                log_message("获取的帧无效", fg="red")
                root.after(30, update_frame)
                return
                
            # 统一帧尺寸（如果两帧尺寸不一致）
            if current_frame.shape != next_frame.shape:
                next_frame = cv2.resize(next_frame, (current_frame.shape[1], current_frame.shape[0]))
                
        except Exception as e:
            log_message(f"摄像头错误: {str(e)}", "ERROR")
            globals()["monitoring_camera"] = None
            camera_button.config(text="选择镜头监控")
            root.after(1000, update_frame)
            return

    elif monitoring_process:
        # 进程监控模式
        try:
            start_time = time.time()
            # 获取窗口位置
            rect = get_window_rect(monitoring_process['hwnd'])
            monitor_area = {
                "left": rect["left"],
                "top": rect["top"],
                "width": rect["width"],
                "height": rect["height"]
            }
            
            # 如果设置了监控区域ROI，则调整监控区域
            if monitor_area_roi:
                x, y, w, h = monitor_area_roi
                monitor_area["left"] += x
                monitor_area["top"] += y
                monitor_area["width"] = w
                monitor_area["height"] = h

            # 读取两帧用于比较
            current_frame = np.array(sct.grab(monitor_area))
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2BGR)
            
            next_frame = np.array(sct.grab(monitor_area))
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGRA2BGR)

            # 计算实际处理耗时
            process_time = (time.time() - start_time) * 1000  # 毫秒
            if process_time > 10:  # 如果单次处理超过10ms
                status_bar.config(text=f"警告：处理延迟 {process_time:.1f}ms", fg="red")


        except Exception as e:
            log_message(f"监控出错: {str(e)}", "ERROR")
            globals()["monitoring_process"] = None
            process_button.config(text="选择镜头监控")
            root.after(1000, update_frame)  # 1秒后重试
            return
    else:
        # 没有选择监控源
        current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            from PIL import Image, ImageDraw, ImageFont
            img_pil = Image.fromarray(current_frame)
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype(font_path, 30)
            draw.text((150, 200), "请选择监控源", font=font, fill=(255, 255, 255))
            font = ImageFont.truetype(font_path, 20)
            draw.text((100, 250), "1. 点击上方按钮选择监控进程", font=font, fill=(255, 255, 255))
            draw.text((100, 280), "2. 或选择监控摄像头", font=font, fill=(255, 255, 255))
            current_frame = np.array(img_pil)
        except:
            cv2.putText(current_frame, "Please select source", (150, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(current_frame, "1. Click button to monitor process", (50, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(current_frame, "2. Or select camera", (50, 290), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        update_video_display(current_frame)
        root.after(30, update_frame)
        return

    # 使用当前算法处理帧
    backend = acceleration_manager.get_current_backend()
    if use_multithread:
        thresh = frame_processor.process_frames(current_frame, next_frame, threshold, backend)
    else:
        thresh = algorithm_manager.process_frame(current_frame, next_frame, threshold)
    
    # 显示算法调试信息
    if show_algorithm_debug:
        debug_info = f"算法: {algorithm_manager.current_algorithm}"
        if algorithm_manager.current_algorithm in algorithm_manager.algorithm_presets:
            debug_info += f" ({', '.join(algorithm_manager.algorithm_presets[algorithm_manager.current_algorithm])})"
        log_message(debug_info, "DEBUG")    

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建副本帧用于绘制
    display_frame = current_frame.copy()

    # 在原始帧上绘制轮廓
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:  # 过滤掉小的变化
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 绘制带渐隐效果的检测框
    current_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            current_boxes.append((x, y, w, h))
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # 渐隐效果处理
    if fade_effect_enabled:
        # 更新渐隐框列表
        fade_boxes = [(box, fade_frames) for box in current_boxes] + \
                    [(box, count-1) for box, count in fade_boxes if count > 1]
    
        # 绘制渐隐框
        for box, count in fade_boxes:
            if box not in current_boxes:  # 只绘制消失的框
                alpha = count / fade_frames  # 计算透明度
                color = (0, int(255*alpha), int(255*(1-alpha)))  # 从绿渐变到黄
                cv2.rectangle(display_frame, 
                             (box[0], box[1]), 
                             (box[0]+box[2], box[1]+box[3]), 
                             color, 
                             1 + int(2*alpha))  # 线宽也逐渐变细
    else:
        # 不启用渐隐效果时的简单绘制
        for box in current_boxes:
            cv2.rectangle(display_frame, 
                         (box[0], box[1]), 
                         (box[0]+box[2], box[1]+box[3]), 
                         (0, 255, 0), 2)

    # 更新视频显示
    update_video_display(current_frame)

    # 计算实际帧率
    current_time = time.time()
    if last_update_time:
        actual_fps = 0.9 * actual_fps + 0.1 * (1 / (current_time - last_update_time))
    last_update_time = current_time

    #状态栏显示保护
    fps_display = f"{min(999, max(1, actual_fps)):.1f}"  # 限制显示范围1-999

    # 构建状态栏文本
    backend = acceleration_manager.get_current_backend()
    status_text = (
        f"运行中: {fps_display}FPS/{frame_rate}FPS | "
        f"负载: {current_load_factor*100:.0f}% | "
        f"模式: {performance_mode}"
    )
    
    # 帧率过高警告
    if frame_rate > 120 or abs(actual_fps - frame_rate) > 10:
        # 初始化status_text变量
        status_text = ""
    
        # 构建基本状态信息
        backend = acceleration_manager.get_current_backend()
        status_text = (
            f"运行中: {fps_display}FPS/{frame_rate}FPS | "
            f"负载: {current_load_factor*100:.0f}% | "
            f"模式: {performance_mode} (设置: {frame_rate}FPS)"
        )
    
        status_text += f" (设置: {frame_rate}FPS)"
        status_bar.config(text=status_text, fg="red")
    
        # 使用线程池异步记录日志，避免阻塞主线程
        def log_fps_warning():
            # global status_text
            status_text = f"⚠️ 帧率过高: 设置 {frame_rate}FPS, 实际 {actual_fps:.1f}FPS"
            status_bar.config(text=status_text, fg="red")
            # 可选：如果需要同时记录日志可以取消下面注释
            # log_message(status_text, "WARNING")
    
        threading.Thread(target=log_fps_warning, daemon=True).start()
        last_fps_warning_time = current_time
        status_text += f" (设置: {frame_rate}FPS)"
        status_bar.config(fg="red")
        status_text += f" (⚠️ 帧率过高: 设置 {frame_rate}FPS, 实际 {actual_fps:.1f}FPS)"
    
    if 'cpu_percent' in locals() and cpu_percent > 80:
        status_text += f" | CPU: {cpu_percent}% ⚠️ CPU过载"
        log_message(f"CPU过载: {cpu_percent}%", "WARNING")
    else:
        status_bar.config(fg="black")
    
    status_text += f" | 处理方式: {'多线程' if use_multithread else '单线程'}"
    status_bar.config(text=status_text)

    # 帧率控制
    #effective_fps = min(frame_rate, 360)
    #delay = max(1, int(1000 / effective_fps))
    #root.after(delay, update_frame)

    # 在帧率控制部分使用调节后的frame_rate
    #effective_fps = min(frame_rate, max_fps_limit) if adaptive_fps_enabled else frame_rate
    effective_fps = max(1, min(frame_rate, max_fps_limit))
    delay = max(1, int(1000 / effective_fps))
    root.after(delay, update_frame)


# 创建控制面板布局（优化布局）


def create_control_panel():
    global frame_rate_scale, threshold_scale, min_area_scale
    global frame_rate_entry, threshold_entry, min_area_entry
    global log_text

    # 使用紧凑的Notebook布局
    tab_control = ttk.Notebook(control_panel)
    tab_control.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    # 主设置标签页 - 使用网格布局并减少间距
    main_tab = ttk.Frame(tab_control)
    tab_control.add(main_tab, text="主设置")

    # 监控源选择 - 紧凑按钮布局
    source_frame = tk.Frame(main_tab, padx=0, pady=0)
    source_frame.pack(fill=tk.X, padx=2, pady=1)
    
    global process_button, camera_button
    process_button = ttk.Button(source_frame, text="选择进程", command=toggle_process_monitoring, width=8)
    process_button.pack(side=tk.LEFT, padx=1, pady=0)
    camera_button = ttk.Button(source_frame, text="选择镜头", command=toggle_camera_monitoring, width=8)
    camera_button.pack(side=tk.LEFT, padx=1, pady=0)
    ttk.Button(source_frame, text="选择区域", command=select_monitor_area, width=8).pack(side=tk.LEFT, padx=1, pady=0)

    # 检测参数设置 - 紧凑滑块布局
    settings_frame = tk.Frame(main_tab)
    settings_frame.pack(fill=tk.X, padx=2, pady=1)

    # 紧凑的参数滑块
    params = [
        ("帧率:", "frame_rate", 1, 360, 4),
        ("阈值:", "threshold", 1, 100, 3),
        ("最小区域:", "min_contour_area", 1, 1000, 4)
    ]
    
    for label, var, min_val, max_val, width in params:
        frame = tk.Frame(settings_frame)
        frame.pack(fill=tk.X, pady=0)
        tk.Label(frame, text=label, width=5, anchor="e").pack(side=tk.LEFT)
        
        scale = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, 
                        length=60, showvalue=0, highlightthickness=0)
        scale.set(globals()[var])
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        entry = tk.Entry(frame, width=width)
        entry.insert(0, str(globals()[var]))
        entry.pack(side=tk.LEFT, padx=2)
        
        # 保存引用
        if var == "frame_rate":
            frame_rate_scale = scale
            frame_rate_entry = entry
        elif var == "threshold":
            threshold_scale = scale
            threshold_entry = entry
        elif var == "min_contour_area":
            min_area_scale = scale
            min_area_entry = entry

    # 显示设置 - 紧凑复选框
    display_frame = tk.Frame(main_tab)
    display_frame.pack(fill=tk.X, padx=2, pady=1)
    
    # 使用小号复选框
    tk.Checkbutton(display_frame, text="宽高比", variable=tk.BooleanVar(value=lock_aspect_ratio),
                  command=lambda: globals().update(lock_aspect_ratio=not lock_aspect_ratio)).pack(side=tk.LEFT, padx=2)
    tk.Checkbutton(display_frame, text="置顶", variable=tk.BooleanVar(value=root.attributes('-topmost')),
                  command=lambda: root.attributes('-topmost', not root.attributes('-topmost'))).pack(side=tk.LEFT, padx=2)
    
    # 渐隐效果紧凑布局
    fade_var = tk.BooleanVar(value=fade_effect_enabled)
    tk.Checkbutton(display_frame, text="渐隐", variable=fade_var,
                  command=lambda: globals().update(fade_effect_enabled=fade_var.get())).pack(side=tk.LEFT, padx=2)
    tk.Scale(display_frame, from_=1, to=30, orient=tk.HORIZONTAL, 
            length=40, showvalue=0, variable=tk.IntVar(value=fade_frames),
            command=lambda v: globals().update(fade_frames=int(v))).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # 加速设置 - 紧凑下拉框
    accel_frame = tk.Frame(main_tab)
    accel_frame.pack(fill=tk.X, padx=2, pady=1)
    
    tk.Label(accel_frame, text="后端:").pack(side=tk.LEFT)
    backend_var = tk.StringVar()
    backend_menu = ttk.Combobox(accel_frame, textvariable=backend_var, state="readonly", width=8)
    backend_menu['values'] = [b.name for b in acceleration_manager.backends]
    backend_menu.current([b.name for b in acceleration_manager.backends].index(acceleration_manager.get_current_backend().name))
    backend_menu.pack(side=tk.LEFT, padx=2)
    backend_menu.bind("<<ComboboxSelected>>", lambda e: acceleration_manager.set_backend(backend_var.get()))

    # 性能调节 - 紧凑布局
    perf_frame = tk.Frame(main_tab)
    perf_frame.pack(fill=tk.X, padx=2, pady=1)
    
    # 第一行
    tk.Checkbutton(perf_frame, text="自适应", variable=tk.BooleanVar(value=adaptive_fps_enabled),
                  command=lambda: globals().update(adaptive_fps_enabled=not adaptive_fps_enabled)).pack(side=tk.LEFT, padx=2)
    tk.Checkbutton(perf_frame, text="动态阈值", variable=tk.BooleanVar(value=dynamic_threshold_enabled),
                  command=lambda: globals().update(dynamic_threshold_enabled=not dynamic_threshold_enabled)).pack(side=tk.LEFT, padx=2)
    
    # 第二行
    tk.Label(perf_frame, text="模式:").pack(side=tk.LEFT, padx=2)
    mode_var = tk.StringVar(value=performance_mode)
    ttk.Combobox(perf_frame, textvariable=mode_var, values=["性能", "平衡", "画质"], 
                state="readonly", width=6).pack(side=tk.LEFT, padx=2)
    mode_var.trace("w", lambda *_: globals().update(performance_mode=mode_var.get()))
    
    # 第三行
    tk.Checkbutton(perf_frame, text="多线程", variable=tk.BooleanVar(value=use_multithread),
                  command=lambda: globals().update(use_multithread=not use_multithread)).pack(side=tk.LEFT, padx=2)
    tk.Checkbutton(perf_frame, text="CPU监控", variable=tk.BooleanVar(value=cpu_monitor_enabled),
                  command=lambda: globals().update(cpu_monitor_enabled=not cpu_monitor_enabled)).pack(side=tk.LEFT, padx=2)

    # 高级设置标签页
    optimize_tab = ttk.Frame(tab_control)
    tab_control.add(optimize_tab, text="高级")

    # 帧跳过设置
    skip_frame = tk.Frame(optimize_tab)
    skip_frame.pack(fill=tk.X, padx=2, pady=1)
    tk.Label(skip_frame, text="跳过帧数:").pack(side=tk.LEFT)
    global frame_skip_var
    frame_skip_var = tk.IntVar(value=0)
    tk.Scale(skip_frame, from_=0, to=5, orient=tk.HORIZONTAL, variable=frame_skip_var,
            showvalue=0, length=80).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # 分辨率调节
    res_frame = tk.Frame(optimize_tab)
    res_frame.pack(fill=tk.X, padx=2, pady=1)
    tk.Label(res_frame, text="分辨率:").pack(side=tk.LEFT)
    global resolution_scale_var
    resolution_scale_var = tk.DoubleVar(value=1.0)
    tk.Scale(res_frame, from_=0.3, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
            variable=resolution_scale_var, showvalue=0, length=80).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # 处理优化
    process_frame = tk.Frame(optimize_tab)
    process_frame.pack(fill=tk.X, padx=2, pady=1)
    global roi_tracking_var, dynamic_interval_var
    roi_tracking_var = tk.BooleanVar(value=False)
    dynamic_interval_var = tk.BooleanVar(value=False)
    tk.Checkbutton(process_frame, text="ROI跟踪", variable=roi_tracking_var).pack(side=tk.LEFT, padx=2)
    tk.Checkbutton(process_frame, text="动态间隔", variable=dynamic_interval_var).pack(side=tk.LEFT, padx=2)

    # 算法优化
    algo_frame = tk.Frame(optimize_tab)
    algo_frame.pack(fill=tk.X, padx=2, pady=1)
    
    # 算法模式选择
    algo_mode = tk.StringVar(value="single")
    tk.Radiobutton(algo_frame, text="单一", variable=algo_mode, value="single").pack(side=tk.LEFT, padx=2)
    tk.Radiobutton(algo_frame, text="组合", variable=algo_mode, value="combo").pack(side=tk.LEFT, padx=2)
    tk.Radiobutton(algo_frame, text="智能", variable=algo_mode, value="smart").pack(side=tk.LEFT, padx=2)
    
    # 算法选择下拉框
    algo_select = ttk.Combobox(algo_frame, state="readonly", width=12)
    algo_select['values'] = list(algorithm_manager.available_algorithms.keys())
    algo_select.current(0)
    algo_select.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
    
    # 算法模式切换
    def update_algo_mode(*_):
        if algo_mode.get() == "single":
            algorithm_manager.set_algorithm(algo_select.get())
        elif algo_mode.get() == "combo":
            algorithm_manager.set_algorithm_combo(algo_select.get())
        else:
            algorithm_manager.set_algorithm("智能组合")
    
    algo_mode.trace("w", update_algo_mode)
    algo_select.bind("<<ComboboxSelected>>", update_algo_mode)

    # 日志标签页
    log_tab = ttk.Frame(tab_control)
    tab_control.add(log_tab, text="日志")

    # 紧凑的日志显示
    log_text = tk.Text(log_tab, wrap=tk.WORD, height=4)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    v_scroll = ttk.Scrollbar(log_tab, command=log_text.yview)
    v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.config(yscrollcommand=v_scroll.set)
    
    # 日志控制按钮
    log_btn_frame = tk.Frame(log_tab)
    log_btn_frame.pack(fill=tk.X, padx=2, pady=1)
    ttk.Button(log_btn_frame, text="清除", command=lambda: log_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=2)
    ttk.Button(log_btn_frame, text="复制", command=lambda: root.clipboard_append(log_text.get(1.0, tk.END))).pack(side=tk.LEFT, padx=2)

    # 绑定原有的事件处理
    frame_rate_scale.config(command=lambda v: (
        frame_rate_entry.delete(0, tk.END),
        frame_rate_entry.insert(0, str(int(float(v)))),
        globals().update(frame_rate=int(float(v)))
    ))
    
    threshold_scale.config(command=lambda v: (
        threshold_entry.delete(0, tk.END),
        threshold_entry.insert(0, str(int(float(v)))),
        globals().update(threshold=int(float(v)))
    ))
    
    min_area_scale.config(command=lambda v: (
        min_area_entry.delete(0, tk.END),
        min_area_entry.insert(0, str(int(float(v)))),
        globals().update(min_contour_area=int(float(v)))
    ))
    
    for entry, var in [(frame_rate_entry, "frame_rate"), (threshold_entry, "threshold"), (min_area_entry, "min_contour_area")]:
        entry.bind("<Return>", lambda e, v=var: globals().update({v: int(e.widget.get())}))


# 创建控制面板
create_control_panel()

# 启动多线程处理器
frame_processor.start_processing(threshold, acceleration_manager.get_current_backend())

# 启动更新帧的函数
update_frame()

# 初始化算法管理器
algorithm_manager = AlgorithmManager()

# 运行主循环
root.mainloop()

# 释放算法资源
if hasattr(algorithm_manager, 'backSub'):
    algorithm_manager.backSub = None

# 释放资源
if monitoring_camera is not None:
    cv2.VideoCapture(monitoring_camera).release()
acceleration_manager.release_all()
frame_processor.stop_processing()