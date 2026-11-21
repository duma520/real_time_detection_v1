"""
智能变化检测系统 - PySide6版本
基于原始Tkinter代码升级，保留了所有功能并利用Qt的优势
"""

import collections
import cv2
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
from abc import ABC, abstractmethod
import threading
import queue
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# PySide6导入
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtUiTools import QUiLoader

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
            
            if not torch.cuda.is_available():
                print("PyTorch CUDA不可用，将使用CPU模式")
                self.device = torch.device('cpu')
                return False
            
            try:
                self.device = torch.device('cuda')
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
            CPUBackend(),
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
        return self.backends[-1]
    
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
        self.current_algorithm = "原始设置"
        self.available_algorithms = {
            "原始设置": self.original_algorithm,
            "高斯模糊预处理": self.gaussian_blur_algorithm,
            "背景减除算法": self.background_subtraction_algorithm,
            "形态学处理": self.morphological_algorithm,
            "光流法优化": self.optical_flow_algorithm,
            "帧差分优化": self.frame_diff_enhanced_algorithm,
            "多尺度检测": self.multi_scale_algorithm
        }
        
        self.algorithm_presets = {
            "快速检测组合": ["高斯模糊预处理", "帧差分优化"],
            "精确检测组合": ["背景减除算法", "形态学处理"],
            "运动追踪组合": ["光流法优化", "多尺度检测"],
            "智能组合": []
        }
        
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        self.smart_combo_cache = None
        self.last_combo_time = 0
    
    def set_algorithm(self, algorithm_name: str) -> bool:
        """设置当前使用的算法"""
        if algorithm_name in self.available_algorithms:
            self.current_algorithm = algorithm_name
            return True
        return False
    
    def set_algorithm_combo(self, combo_name: str) -> bool:
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
        
        cpu_load = psutil.cpu_percent()
        mem_load = psutil.virtual_memory().percent
        
        if cpu_load > 70 or mem_load > 80:
            combo = ["高斯模糊预处理", "帧差分优化"]
        else:
            combo = ["背景减除算法", "形态学处理"]
        
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
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def gaussian_blur_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
        
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def background_subtraction_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        fg_mask = self.backSub.apply(frame2)
        _, thresh = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def morphological_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return thresh
    
    def optical_flow_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **self.flow_params)
        
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def frame_diff_enhanced_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray1, gray2)
        diff = cv2.multiply(diff, 1.5)
        
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    
    def multi_scale_algorithm(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
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
        
        final_thresh = np.zeros_like(results[0])
        for thresh in results:
            final_thresh = cv2.bitwise_or(final_thresh, thresh)
        
        return final_thresh

# ==================== 多线程处理类 ====================

class FrameProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        self.running = True
        self.use_multithread = True
        
    def set_multithread(self, enabled: bool):
        self.use_multithread = enabled
        
    def process_frame_task(self, frame1, frame2, threshold, backend):
        try:
            thresh = backend.process_frames(frame1, frame2, threshold)
            return thresh
        except Exception as e:
            print(f"线程处理失败: {e}")
            return None
    
    def process_frames(self, frame1, frame2, threshold, backend):
        if not self.use_multithread:
            return backend.process_frames(frame1, frame2, threshold)
            
        try:
            self.frame_queue.put((frame1, frame2))
            if not self.result_queue.empty():
                future = self.result_queue.get()
                new_thresh = future.result()
                if new_thresh is not None:
                    if new_thresh.dtype != np.uint8:
                        new_thresh = new_thresh.astype(np.uint8)
                    return new_thresh
            return backend.process_frames(frame1, frame2, threshold)
        except Exception as e:
            print(f"多线程处理失败: {str(e)}，使用单线程")
            return backend.process_frames(frame1, frame2, threshold)
    
    def start_processing(self, threshold, backend):
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
        self.running = False
        self.executor.shutdown(wait=True)
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join()

# ==================== PySide6主窗口 ====================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 全局变量
        self.version = "v70.4.7"
        self.author = "杜玛"
        self.copyright = "Copyright © 杜玛. All rights reserved."
        self.threshold = 30
        self.min_contour_area = 500
        self.monitoring_process = None
        self.monitoring_camera = None
        self.sct = mss()
        self.frame_rate = 30
        self.lock_aspect_ratio = True
        self.monitor_area_roi = None
        self.font_path = "msyh.ttf"
        self.max_recommended_fps = 30
        self.performance_test_done = False
        self.last_update_time = None
        self.actual_fps = 0
        self.cpu_monitor_enabled = True
        self.frame_count = 0
        self.last_boxes = []
        self.fade_effect_enabled = False
        self.fade_frames = 10
        self.fade_boxes = []
        self.use_multithread = True
        self.adaptive_fps_enabled = True
        self.max_fps_limit = 60
        self.min_fps_limit = 5
        self.performance_mode = "平衡"
        self.dynamic_threshold_enabled = False
        self.current_load_factor = 1.0
        self.log_messages = []
        self.max_log_entries = 100
        self.last_fps_warning_time = 0
        self.fps_warning_interval = 30
        self.show_algorithm_debug = False
        
        self.frame_times = collections.deque(maxlen=60)
        
        # 初始化管理器
        self.acceleration_manager = AccelerationManager()
        self.frame_processor = FrameProcessor()
        self.algorithm_manager = AlgorithmManager()
        
        # 设置UI
        self.setup_ui()
        
        # 启动帧处理器
        self.frame_processor.start_processing(self.threshold, self.acceleration_manager.get_current_backend())
        
        # 启动性能检测
        QTimer.singleShot(100, self.detect_max_fps)
        
        # 启动视频更新
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_frame)
        self.update_timer.start(33)  # 约30fps
        
    def setup_ui(self):
        """设置UI界面"""
        self.setWindowTitle(f"智能变化检测系统 ({self.version} | {self.author} | {self.copyright})")
        self.setGeometry(100, 100, 500, 850)
        self.setMinimumSize(400, 600)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(320, 240)
        main_layout.addWidget(self.video_label, 1)
        
        # 控制面板开关按钮
        self.toggle_btn = QPushButton("▲ 隐藏控制面板 ▲")
        self.toggle_btn.clicked.connect(self.toggle_control_panel)
        main_layout.addWidget(self.toggle_btn)
        
        # 控制面板
        self.control_panel = QWidget()
        self.control_panel.setVisible(True)
        self.control_panel_layout = QVBoxLayout(self.control_panel)
        self.control_panel_layout.setContentsMargins(2, 2, 2, 2)
        self.control_panel_layout.setSpacing(2)
        main_layout.addWidget(self.control_panel)
        
        # 创建标签页
        self.setup_tab_control()
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)
        
    def setup_tab_control(self):
        """设置标签页控件"""
        tab_widget = QTabWidget()
        self.control_panel_layout.addWidget(tab_widget)
        
        # 主设置标签页
        main_tab = QWidget()
        tab_widget.addTab(main_tab, "主设置")
        self.setup_main_tab(main_tab)
        
        # 高级标签页
        advanced_tab = QWidget()
        tab_widget.addTab(advanced_tab, "高级")
        self.setup_advanced_tab(advanced_tab)
        
        # 日志标签页
        log_tab = QWidget()
        tab_widget.addTab(log_tab, "日志")
        self.setup_log_tab(log_tab)
        
    def setup_main_tab(self, parent):
        """设置主标签页"""
        layout = QVBoxLayout(parent)
        layout.setSpacing(2)
        
        # 监控源选择
        source_layout = QHBoxLayout()
        self.process_btn = QPushButton("选择进程")
        self.camera_btn = QPushButton("选择镜头")
        self.area_btn = QPushButton("选择区域")
        
        self.process_btn.clicked.connect(self.toggle_process_monitoring)
        self.camera_btn.clicked.connect(self.toggle_camera_monitoring)
        self.area_btn.clicked.connect(self.select_monitor_area)
        
        source_layout.addWidget(self.process_btn)
        source_layout.addWidget(self.camera_btn)
        source_layout.addWidget(self.area_btn)
        layout.addLayout(source_layout)
        
        # 参数设置
        params = [
            ("帧率:", "frame_rate", 1, 360),
            ("阈值:", "threshold", 1, 100),
            ("最小区域:", "min_contour_area", 1, 1000)
        ]
        
        self.param_scales = {}
        self.param_edits = {}
        
        for label_text, var_name, min_val, max_val in params:
            param_layout = QHBoxLayout()
            
            label = QLabel(label_text)
            label.setFixedWidth(50)
            param_layout.addWidget(label)
            
            scale = QSlider(Qt.Horizontal)
            scale.setRange(min_val, max_val)
            scale.setValue(getattr(self, var_name))
            scale.valueChanged.connect(lambda v, vn=var_name: self.update_param(vn, v))
            param_layout.addWidget(scale, 1)
            
            edit = QLineEdit()
            edit.setFixedWidth(50)
            edit.setText(str(getattr(self, var_name)))
            edit.returnPressed.connect(lambda e=None, vn=var_name, ed=edit: self.update_param_from_edit(vn, ed))
            param_layout.addWidget(edit)
            
            self.param_scales[var_name] = scale
            self.param_edits[var_name] = edit
            
            layout.addLayout(param_layout)
        
        # 显示设置
        display_layout = QHBoxLayout()
        
        self.aspect_check = QCheckBox("宽高比")
        self.aspect_check.setChecked(self.lock_aspect_ratio)
        self.aspect_check.toggled.connect(lambda v: setattr(self, 'lock_aspect_ratio', v))
        
        self.topmost_check = QCheckBox("置顶")
        self.topmost_check.toggled.connect(self.toggle_topmost)
        
        self.fade_check = QCheckBox("渐隐")
        self.fade_check.toggled.connect(lambda v: setattr(self, 'fade_effect_enabled', v))
        
        self.fade_slider = QSlider(Qt.Horizontal)
        self.fade_slider.setRange(1, 30)
        self.fade_slider.setValue(self.fade_frames)
        self.fade_slider.setFixedWidth(60)
        self.fade_slider.valueChanged.connect(lambda v: setattr(self, 'fade_frames', v))
        
        display_layout.addWidget(self.aspect_check)
        display_layout.addWidget(self.topmost_check)
        display_layout.addWidget(self.fade_check)
        display_layout.addWidget(self.fade_slider)
        layout.addLayout(display_layout)
        
        # 加速设置
        accel_layout = QHBoxLayout()
        accel_layout.addWidget(QLabel("后端:"))
        
        self.backend_combo = QComboBox()
        backends = [b.name for b in self.acceleration_manager.backends]
        self.backend_combo.addItems(backends)
        current_backend = self.acceleration_manager.get_current_backend().name
        self.backend_combo.setCurrentText(current_backend)
        self.backend_combo.currentTextChanged.connect(self.change_backend)
        accel_layout.addWidget(self.backend_combo)
        
        layout.addLayout(accel_layout)
        
        # 性能调节
        self.setup_performance_controls(layout)
        
        layout.addStretch()
        
    def setup_performance_controls(self, parent_layout):
        """设置性能控制"""
        # 第一行
        perf_layout1 = QHBoxLayout()
        
        self.adaptive_check = QCheckBox("自适应")
        self.adaptive_check.setChecked(self.adaptive_fps_enabled)
        self.adaptive_check.toggled.connect(lambda v: setattr(self, 'adaptive_fps_enabled', v))
        
        self.dynamic_threshold_check = QCheckBox("动态阈值")
        self.dynamic_threshold_check.setChecked(self.dynamic_threshold_enabled)
        self.dynamic_threshold_check.toggled.connect(lambda v: setattr(self, 'dynamic_threshold_enabled', v))
        
        perf_layout1.addWidget(self.adaptive_check)
        perf_layout1.addWidget(self.dynamic_threshold_check)
        parent_layout.addLayout(perf_layout1)
        
        # 第二行
        perf_layout2 = QHBoxLayout()
        perf_layout2.addWidget(QLabel("模式:"))
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["性能", "平衡", "画质"])
        self.mode_combo.setCurrentText(self.performance_mode)
        self.mode_combo.currentTextChanged.connect(lambda v: setattr(self, 'performance_mode', v))
        perf_layout2.addWidget(self.mode_combo)
        
        # 第三行
        self.multithread_check = QCheckBox("多线程")
        self.multithread_check.setChecked(self.use_multithread)
        self.multithread_check.toggled.connect(lambda v: setattr(self, 'use_multithread', v))
        
        self.cpu_monitor_check = QCheckBox("CPU监控")
        self.cpu_monitor_check.setChecked(self.cpu_monitor_enabled)
        self.cpu_monitor_check.toggled.connect(lambda v: setattr(self, 'cpu_monitor_enabled', v))
        
        perf_layout2.addWidget(self.multithread_check)
        perf_layout2.addWidget(self.cpu_monitor_check)
        parent_layout.addLayout(perf_layout2)
        
    def setup_advanced_tab(self, parent):
        """设置高级标签页"""
        layout = QVBoxLayout(parent)
        layout.setSpacing(2)
        
        # 帧跳过
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("跳过帧数:"))
        
        self.skip_slider = QSlider(Qt.Horizontal)
        self.skip_slider.setRange(0, 5)
        self.skip_slider.setValue(0)
        skip_layout.addWidget(self.skip_slider, 1)
        layout.addLayout(skip_layout)
        
        # 分辨率
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("分辨率:"))
        
        self.res_slider = QSlider(Qt.Horizontal)
        self.res_slider.setRange(3, 10)
        self.res_slider.setValue(10)
        self.res_slider.setTickInterval(1)
        res_layout.addWidget(self.res_slider, 1)
        
        self.res_label = QLabel("1.0")
        res_layout.addWidget(self.res_label)
        
        self.res_slider.valueChanged.connect(lambda v: self.res_label.setText(f"{v/10:.1f}"))
        layout.addLayout(res_layout)
        
        # 处理优化
        process_layout = QHBoxLayout()
        
        self.roi_tracking_check = QCheckBox("ROI跟踪")
        self.roi_tracking_check.setChecked(False)
        
        self.dynamic_interval_check = QCheckBox("动态间隔")
        self.dynamic_interval_check.setChecked(False)
        
        process_layout.addWidget(self.roi_tracking_check)
        process_layout.addWidget(self.dynamic_interval_check)
        layout.addLayout(process_layout)
        
        # 算法优化
        algo_layout = QHBoxLayout()
        
        self.algo_group = QButtonGroup()
        self.single_radio = QRadioButton("单一")
        self.combo_radio = QRadioButton("组合")
        self.smart_radio = QRadioButton("智能")
        
        self.algo_group.addButton(self.single_radio)
        self.algo_group.addButton(self.combo_radio)
        self.algo_group.addButton(self.smart_radio)
        self.single_radio.setChecked(True)
        
        algo_layout.addWidget(self.single_radio)
        algo_layout.addWidget(self.combo_radio)
        algo_layout.addWidget(self.smart_radio)
        
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(list(self.algorithm_manager.available_algorithms.keys()))
        self.algo_combo.currentTextChanged.connect(self.change_algorithm)
        algo_layout.addWidget(self.algo_combo)
        
        layout.addLayout(algo_layout)
        
        # 算法模式切换
        self.single_radio.toggled.connect(self.update_algo_mode)
        self.combo_radio.toggled.connect(self.update_algo_mode)
        self.smart_radio.toggled.connect(self.update_algo_mode)
        
        layout.addStretch()
        
    def setup_log_tab(self, parent):
        """设置日志标签页"""
        layout = QVBoxLayout(parent)
        
        # 将 QTextEdit 改为 QPlainTextEdit
        self.log_text = QPlainTextEdit()  # 修改这里
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(100)  # 这个方法现在可以用了
        layout.addWidget(self.log_text)
        
        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("清除")
        clear_btn.clicked.connect(self.log_text.clear)
        
        copy_btn = QPushButton("复制")
        copy_btn.clicked.connect(self.copy_log)
        
        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(copy_btn)
        layout.addLayout(btn_layout)
        
    # ==================== 辅助方法 ====================
    
    def log_message(self, message, level="INFO"):
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        self.log_messages.append(log_entry)
        if len(self.log_messages) > self.max_log_entries:
            self.log_messages.pop(0)
        
        # 更新日志显示
        self.log_text.appendPlainText(log_entry)
        
        # 状态栏显示重要消息
        if level in ["WARNING", "ERROR"]:
            self.status_label.setText(message)
            if level == "ERROR":
                self.status_label.setStyleSheet("color: red;")
            else:
                self.status_label.setStyleSheet("color: orange;")
    
    def update_param(self, var_name, value):
        """更新参数值"""
        setattr(self, var_name, value)
        if var_name in self.param_edits:
            self.param_edits[var_name].setText(str(value))
    
    def update_param_from_edit(self, var_name, edit):
        """从编辑框更新参数"""
        try:
            value = int(edit.text())
            setattr(self, var_name, value)
            if var_name in self.param_scales:
                self.param_scales[var_name].setValue(value)
        except ValueError:
            pass
    
    def toggle_topmost(self, checked):
        """切换置顶状态"""
        if checked:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()
    
    def toggle_control_panel(self):
        """切换控制面板显示"""
        if self.control_panel.isVisible():
            self.control_panel.hide()
            self.toggle_btn.setText("▼ 显示控制面板 ▼")
        else:
            self.control_panel.show()
            self.toggle_btn.setText("▲ 隐藏控制面板 ▲")
    
    def change_backend(self, backend_name):
        """更改加速后端"""
        self.acceleration_manager.set_backend(backend_name)
        self.log_message(f"切换到加速后端: {backend_name}")
    
    def change_algorithm(self, algorithm_name):
        """更改算法"""
        if self.single_radio.isChecked():
            self.algorithm_manager.set_algorithm(algorithm_name)
        elif self.combo_radio.isChecked():
            self.algorithm_manager.set_algorithm_combo(algorithm_name)
        else:
            self.algorithm_manager.set_algorithm("智能组合")
    
    def update_algo_mode(self):
        """更新算法模式"""
        if self.single_radio.isChecked():
            self.algorithm_manager.set_algorithm(self.algo_combo.currentText())
        elif self.combo_radio.isChecked():
            self.algorithm_manager.set_algorithm_combo(self.algo_combo.currentText())
        else:
            self.algorithm_manager.set_algorithm("智能组合")
    
    def copy_log(self):
        """复制日志到剪贴板"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.log_text.toPlainText())  # 这行不变，toPlainText() 对两者都有效
    
    # ==================== 功能方法 ====================
    
    def detect_max_fps(self):
        """检测最大FPS"""
        test_duration = 2
        start_time = time.time()
        frame_count = 0
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        backend = self.acceleration_manager.get_current_backend()
        
        while time.time() - start_time < test_duration:
            _ = backend.process_frames(test_frame, test_frame, self.threshold)
            frame_count += 1
        
        self.max_recommended_fps = min(360, max(30, frame_count // test_duration))
        self.performance_test_done = True
        self.log_message(f"系统检测: 推荐最大帧率 {self.max_recommended_fps}FPS | 加速方式: {backend.name}")
        return self.max_recommended_fps
    
    def adjust_performance_settings(self):
        """根据当前负载动态调节参数"""
        if not self.adaptive_fps_enabled:
            return
        
        cpu_load = psutil.cpu_percent() / 100
        mem_load = psutil.virtual_memory().percent / 100
        current_load = max(cpu_load, mem_load)
        
        self.current_load_factor = 0.5 + current_load
        
        if self.performance_mode == "性能":
            frame_rate = int(self.max_fps_limit * (1.8 - self.current_load_factor))
            if self.dynamic_threshold_enabled:
                self.threshold = min(100, int(30 * self.current_load_factor))
        elif self.performance_mode == "画质":
            frame_rate = max(self.min_fps_limit, int(self.max_fps_limit * (1.2 - self.current_load_factor/2)))
        else:  # balanced
            frame_rate = int(self.max_fps_limit * (1.5 - self.current_load_factor))
        
        self.frame_rate = max(self.min_fps_limit, min(self.max_fps_limit, frame_rate))
        self.threshold = max(5, min(100, self.threshold))
        
        # 更新UI
        if "frame_rate" in self.param_scales:
            self.param_scales["frame_rate"].setValue(self.frame_rate)
    
    def get_process_windows(self):
        """获取所有可见窗口的进程列表"""
        process_list = []
        
        def callback(hwnd, hwnd_list):
            if win32gui.IsWindowVisible(hwnd):
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                try:
                    process = psutil.Process(pid)
                    window_title = win32gui.GetWindowText(hwnd)
                    if window_title:
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
    
    def get_available_cameras(self, max_test=5):
        """获取可用的摄像头列表"""
        cameras = []
        backends = [
            cv2.CAP_DSHOW,
            cv2.CAP_MSMF,
            cv2.CAP_ANY
        ]

        for i in range(max_test):
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        cameras.append(i)
                        cap.release()
                        break
                except:
                    continue
        return cameras
    
    def get_window_rect(self, hwnd):
        """获取窗口的矩形区域"""
        rect = win32gui.GetWindowRect(hwnd)
        return {
            "left": rect[0],
            "top": rect[1],
            "width": rect[2] - rect[0],
            "height": rect[3] - rect[1]
        }
    
    def get_next_frame(self):
        """获取下一帧"""
        if self.monitoring_camera is not None:
            cap = cv2.VideoCapture(self.monitoring_camera)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        else:
            rect = self.get_window_rect(self.monitoring_process['hwnd'])
            monitor_area = {
                "left": rect["left"],
                "top": rect["top"],
                "width": rect["width"],
                "height": rect["height"]
            }
            if self.monitor_area_roi:
                x, y, w, h = self.monitor_area_roi
                monitor_area["left"] += x
                monitor_area["top"] += y
                monitor_area["width"] = w
                monitor_area["height"] = h
            frame = np.array(self.sct.grab(monitor_area))
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    def process_frame_roi(self, frame):
        """处理指定帧区域的核心函数"""
        # 获取两帧用于比较
        frame1 = frame.copy()
        time.sleep(1.0/self.frame_rate)
        frame2 = self.get_next_frame()
        
        if frame2 is None or frame1.shape != frame2.shape:
            return frame1
        
        backend = self.acceleration_manager.get_current_backend()
        thresh = backend.process_frames(frame1, frame2, self.threshold)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                current_boxes.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if self.fade_effect_enabled:
            self.fade_boxes = [(box, self.fade_frames) for box in current_boxes] + \
                             [(box, count-1) for box, count in self.fade_boxes if count > 1]
            
            for box, count in self.fade_boxes:
                if box not in current_boxes:
                    alpha = count / self.fade_frames
                    color = (0, int(255*alpha), int(255*(1-alpha)))
                    cv2.rectangle(frame, 
                                 (box[0], box[1]), 
                                 (box[0]+box[2], box[1]+box[3]), 
                                 color, 
                                 1 + int(2*alpha))
        
        self.last_boxes = current_boxes
        return frame
    
    def toggle_process_monitoring(self):
        """切换进程监控"""
        if self.monitoring_process:
            self.monitoring_process = None
            self.process_btn.setText("选择进程")
            self.log_message("已停止监控")
        else:
            self.monitoring_camera = None
            self.camera_btn.setText("选择镜头")
            self.show_process_selection_dialog()
    
    def toggle_camera_monitoring(self):
        """切换摄像头监控"""
        if self.monitoring_camera is not None:
            self.monitoring_camera = None
            self.camera_btn.setText("选择镜头")
            self.log_message("已停止监控")
        else:
            self.monitoring_process = None
            self.process_btn.setText("选择进程")
            self.show_camera_selection_dialog()
    
    def select_monitor_area(self):
        """选择监控区域"""
        if not self.monitoring_process:
            QMessageBox.warning(self, "警告", "请先选择要监控的进程")
            return
        
        # 简化的区域选择，实际应用中可能需要更复杂的实现
        hwnd = self.monitoring_process['hwnd']
        rect = self.get_window_rect(hwnd)
        
        # 创建简单的输入对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择监控区域")
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("请输入区域坐标 (x, y, width, height):"))
        
        input_layout = QHBoxLayout()
        x_edit = QLineEdit()
        y_edit = QLineEdit()
        w_edit = QLineEdit()
        h_edit = QLineEdit()
        
        x_edit.setPlaceholderText("X")
        y_edit.setPlaceholderText("Y")
        w_edit.setPlaceholderText("宽度")
        h_edit.setPlaceholderText("高度")
        
        input_layout.addWidget(x_edit)
        input_layout.addWidget(y_edit)
        input_layout.addWidget(w_edit)
        input_layout.addWidget(h_edit)
        layout.addLayout(input_layout)
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QDialog.Accepted:
            try:
                x = int(x_edit.text())
                y = int(y_edit.text())
                w = int(w_edit.text())
                h = int(h_edit.text())
                
                if w > 10 and h > 10:
                    self.monitor_area_roi = (x, y, w, h)
                    self.log_message(f"已设置监控区域: {self.monitor_area_roi}")
                else:
                    self.log_message("选择区域太小，请重新选择", "WARNING")
            except ValueError:
                self.log_message("请输入有效的数字", "WARNING")
    
    def show_process_selection_dialog(self):
        """显示进程选择对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"选择需要监控的程序 ({self.version} | {self.author} | {self.copyright})")
        dialog.resize(750, 500)
        
        layout = QVBoxLayout(dialog)
        
        # 搜索框
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("搜索:"))
        search_edit = QLineEdit()
        search_layout.addWidget(search_edit)
        
        refresh_btn = QPushButton("刷新")
        search_layout.addWidget(refresh_btn)
        layout.addLayout(search_layout)
        
        # 进程列表
        tree = QTreeWidget()
        tree.setHeaderLabels(["PID", "进程名", "窗口标题"])
        tree.setColumnWidth(0, 80)
        tree.setColumnWidth(1, 150)
        tree.setColumnWidth(2, 350)
        layout.addWidget(tree)
        
        def update_process_list():
            tree.clear()
            processes = self.get_process_windows()
            for process in processes:
                item = QTreeWidgetItem(tree)
                item.setText(0, str(process['pid']))
                item.setText(1, process['name'])
                item.setText(2, process['title'])
                item.setData(0, Qt.UserRole, process)
        
        def filter_processes():
            query = search_edit.text().lower()
            tree.clear()
            processes = self.get_process_windows()
            for process in processes:
                if (query in str(process['pid']).lower() or 
                    query in process['name'].lower() or 
                    query in process['title'].lower()):
                    item = QTreeWidgetItem(tree)
                    item.setText(0, str(process['pid']))
                    item.setText(1, process['name'])
                    item.setText(2, process['title'])
                    item.setData(0, Qt.UserRole, process)
        
        refresh_btn.clicked.connect(update_process_list)
        search_edit.textChanged.connect(filter_processes)
        
        update_process_list()
        
        # 按钮
        btn_layout = QHBoxLayout()
        select_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        
        select_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        
        btn_layout.addWidget(select_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QDialog.Accepted:
            selected = tree.currentItem()
            if selected:
                process = selected.data(0, Qt.UserRole)
                self.monitoring_process = process
                self.process_btn.setText(f"停止监控 {process['name']}")
                self.log_message(f"已开始监控进程: {process['name']}")
    
    def show_camera_selection_dialog(self):
        """显示摄像头选择对话框"""
        cameras = self.get_available_cameras()
        if not cameras:
            self.log_message("没有检测到可用的摄像头", "WARNING")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"选择监控镜头 ({self.version} | {self.author} | {self.copyright})")
        dialog.resize(400, 300)
        
        layout = QVBoxLayout(dialog)
        
        tree = QTreeWidget()
        tree.setHeaderLabels(["摄像头ID", "状态"])
        tree.setColumnWidth(0, 100)
        layout.addWidget(tree)
        
        for cam_id in cameras:
            item = QTreeWidgetItem(tree)
            item.setText(0, str(cam_id))
            item.setText(1, "可用")
        
        btn_layout = QHBoxLayout()
        select_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        
        select_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        
        btn_layout.addWidget(select_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QDialog.Accepted:
            selected = tree.currentItem()
            if selected:
                cam_id = int(selected.text(0))
                self.monitoring_camera = cam_id
                self.camera_btn.setText(f"停止监控 摄像头{cam_id}")
                self.log_message(f"已开始监控摄像头: {cam_id}")
    
    def update_video_display(self, frame):
        """更新视频显示"""
        if frame is None:
            return
        
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 获取QLabel尺寸
        label_size = self.video_label.size()
        if label_size.width() <= 1 or label_size.height() <= 1:
            return
        
        # 转换为QImage
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 缩放图像以适应标签
        if self.lock_aspect_ratio:
            pixmap = QPixmap.fromImage(q_img).scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            pixmap = QPixmap.fromImage(q_img).scaled(
                label_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        
        self.video_label.setPixmap(pixmap)
    
    def update_frame(self):
        """更新帧的主函数"""
        try:
            # 性能监控
            if self.frame_count % 10 == 0:
                self.adjust_performance_settings()
            
            self.frame_count += 1
            start_time = time.time()
            
            # 获取帧
            if self.monitoring_camera is not None:
                current_frame, next_frame = self.get_camera_frames()
            elif self.monitoring_process:
                current_frame, next_frame = self.get_process_frames()
            else:
                self.show_no_source_message()
                return
            
            if current_frame is None or next_frame is None:
                return
            
            # 处理帧
            processed_frame = self.process_frames(current_frame, next_frame)
            
            # 更新显示
            self.update_video_display(processed_frame)
            
            # 更新状态栏
            self.update_status_bar()
            
            # 计算实际帧率
            current_time = time.time()
            if self.last_update_time:
                time_diff = current_time - self.last_update_time
                if time_diff > 0:
                    self.actual_fps = 0.9 * self.actual_fps + 0.1 * (1 / time_diff)
            self.last_update_time = current_time
            
        except Exception as e:
            self.log_message(f"更新帧时出错: {str(e)}", "ERROR")
    
    def get_camera_frames(self):
        """获取摄像头帧"""
        try:
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            cap = None
            
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(self.monitoring_camera, backend)
                    if cap.isOpened():
                        break
                except:
                    if cap:
                        cap.release()
                    continue
            
            if not cap or not cap.isOpened():
                self.log_message(f"无法打开摄像头 {self.monitoring_camera}", "ERROR")
                if cap:
                    cap.release()
                return None, None
            
            cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            ret1, frame1 = cap.read()
            ret2, frame2 = cap.read()
            cap.release()
            
            if not ret1 or not ret2:
                return None, None
            
            if frame1.shape != frame2.shape:
                frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
            
            return frame1, frame2
            
        except Exception as e:
            self.log_message(f"摄像头错误: {str(e)}", "ERROR")
            self.monitoring_camera = None
            self.camera_btn.setText("选择镜头")
            return None, None
    
    def get_process_frames(self):
        """获取进程窗口帧"""
        try:
            rect = self.get_window_rect(self.monitoring_process['hwnd'])
            monitor_area = {
                "left": rect["left"],
                "top": rect["top"],
                "width": rect["width"],
                "height": rect["height"]
            }
            
            if self.monitor_area_roi:
                x, y, w, h = self.monitor_area_roi
                monitor_area["left"] += x
                monitor_area["top"] += y
                monitor_area["width"] = w
                monitor_area["height"] = h
            
            frame1 = np.array(self.sct.grab(monitor_area))
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2BGR)
            
            frame2 = np.array(self.sct.grab(monitor_area))
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGRA2BGR)
            
            return frame1, frame2
            
        except Exception as e:
            self.log_message(f"监控出错: {str(e)}", "ERROR")
            self.monitoring_process = None
            self.process_btn.setText("选择进程")
            return None, None
    
    def show_no_source_message(self):
        """显示没有选择监控源的消息"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            # 创建带文本的图像
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "请选择监控源", (150, 220), font, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "1. 点击按钮选择监控进程", (50, 260), font, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "2. 或选择监控摄像头", (50, 290), font, 0.6, (255, 255, 255), 1)
        except:
            pass
        
        self.update_video_display(frame)
    
    def process_frames(self, current_frame, next_frame):
        """处理帧并返回带检测框的图像"""
        # 获取分辨率缩放
        scale = self.res_slider.value() / 10.0
        
        if scale < 1.0:
            working_frame = cv2.resize(current_frame, (0, 0), fx=scale, fy=scale)
        else:
            working_frame = current_frame.copy()
        
        # ROI跟踪
        if self.roi_tracking_check.isChecked() and len(self.last_boxes) > 0:
            x_min = min(box[0] for box in self.last_boxes)
            y_min = min(box[1] for box in self.last_boxes)
            x_max = max(box[0]+box[2] for box in self.last_boxes)
            y_max = max(box[1]+box[3] for box in self.last_boxes)
            
            margin_x = int((x_max - x_min) * 0.2)
            margin_y = int((y_max - y_min) * 0.2)
            
            roi_x = max(0, x_min - margin_x)
            roi_y = max(0, y_min - margin_y)
            roi_w = min(working_frame.shape[1], x_max + margin_x) - roi_x
            roi_h = min(working_frame.shape[0], y_max + margin_y) - roi_y
            
            roi_frame = working_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            processed_roi = self.process_frame_roi(roi_frame)
            working_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = processed_roi
        else:
            # 使用算法管理器处理
            backend = self.acceleration_manager.get_current_backend()
            thresh = self.algorithm_manager.process_frame(current_frame, next_frame, self.threshold)
            
            # 查找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 绘制检测框
            current_boxes = []
            for contour in contours:
                if cv2.contourArea(contour) > self.min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    current_boxes.append((x, y, w, h))
                    
                    # 根据缩放比例调整坐标
                    if scale < 1.0:
                        x = int(x / scale)
                        y = int(y / scale)
                        w = int(w / scale)
                        h = int(h / scale)
                    
                    cv2.rectangle(working_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 渐隐效果
            if self.fade_effect_enabled:
                self.fade_boxes = [(box, self.fade_frames) for box in current_boxes] + \
                                 [(box, count-1) for box, count in self.fade_boxes if count > 1]
                
                for box, count in self.fade_boxes:
                    if box not in current_boxes:
                        alpha = count / self.fade_frames
                        color = (0, int(255*alpha), int(255*(1-alpha)))
                        
                        if scale < 1.0:
                            x = int(box[0] / scale)
                            y = int(box[1] / scale)
                            w = int(box[2] / scale)
                            h = int(box[3] / scale)
                        else:
                            x, y, w, h = box
                        
                        cv2.rectangle(working_frame, (x, y), (x+w, y+h), color, 1 + int(2*alpha))
            
            self.last_boxes = current_boxes
        
        # 如果需要，缩放回原始大小
        if scale < 1.0:
            working_frame = cv2.resize(working_frame, (current_frame.shape[1], current_frame.shape[0]))
        
        return working_frame
    
    def update_status_bar(self):
        """更新状态栏"""
        backend = self.acceleration_manager.get_current_backend()
        
        # 构建状态文本
        status_text = (
            f"运行中: {self.actual_fps:.1f}FPS/{self.frame_rate}FPS | "
            f"负载: {self.current_load_factor*100:.0f}% | "
            f"模式: {self.performance_mode} | "
            f"后端: {backend.name}"
        )
        
        # 帧率警告
        if self.frame_rate > 120 or abs(self.actual_fps - self.frame_rate) > 10:
            current_time = time.time()
            if current_time - self.last_fps_warning_time > self.fps_warning_interval:
                self.log_message(f"⚠️ 帧率过高: 设置 {self.frame_rate}FPS, 实际 {self.actual_fps:.1f}FPS", "WARNING")
                self.last_fps_warning_time = current_time
            self.status_label.setStyleSheet("color: red;")
        else:
            self.status_label.setStyleSheet("")
        
        self.status_label.setText(status_text)
    
    def closeEvent(self, event):
        """关闭事件处理"""
        self.log_message("正在关闭程序...")
        
        # 停止定时器
        self.update_timer.stop()
        
        # 停止处理器
        self.frame_processor.stop_processing()
        
        # 释放资源
        if self.monitoring_camera is not None:
            cv2.VideoCapture(self.monitoring_camera).release()
        
        self.acceleration_manager.release_all()
        
        event.accept()

# ==================== 主程序入口 ====================

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("智能变化检测系统")
    app.setOrganizationName("杜玛")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()