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
            # 降采样处理
            scale_factor = 0.7
            small_frame1 = cv2.resize(frame1, (0,0), fx=scale_factor, fy=scale_factor)
            small_frame2 = cv2.resize(frame2, (0,0), fx=scale_factor, fy=scale_factor)
            
            self.gpu_frame1.upload(small_frame1, self.stream)
            self.gpu_frame2.upload(small_frame2, self.stream)
            
            gpu_gray1 = cv2.cuda.cvtColor(self.gpu_frame1, cv2.COLOR_BGR2GRAY, stream=self.stream)
            gpu_gray2 = cv2.cuda.cvtColor(self.gpu_frame2, cv2.COLOR_BGR2GRAY, stream=self.stream)
            
            gpu_diff = cv2.cuda.absdiff(gpu_gray1, gpu_gray2, stream=self.stream)
            _, gpu_thresh = cv2.cuda.threshold(gpu_diff, threshold, 255, cv2.THRESH_BINARY, stream=self.stream)
            
            thresh = gpu_thresh.download(stream=self.stream)
            self.stream.waitForCompletion()
            return cv2.resize(thresh, (frame1.shape[1], frame1.shape[0]))
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
            # 降采样处理
            scale_factor = 0.7
            small_frame1 = cv2.resize(frame1, (0,0), fx=scale_factor, fy=scale_factor)
            small_frame2 = cv2.resize(frame2, (0,0), fx=scale_factor, fy=scale_factor)
            
            umat1 = cv2.UMat(small_frame1)
            umat2 = cv2.UMat(small_frame2)
            
            gray1 = cv2.cvtColor(umat1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(umat2, cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            result = cv2.UMat.get(thresh) if isinstance(thresh, cv2.UMat) else thresh
            return cv2.resize(result, (frame1.shape[1], frame1.shape[0]))
        except:
            self.initialized = False
            return super().process_frames(frame1, frame2, threshold)
    
    def release(self):
        """释放OpenCL资源"""
        cv2.ocl.setUseOpenCL(False)
        self.initialized = False

class CPUBackend(AccelerationBackend):
    """CPU后备方案"""
    def __init__(self):
        super().__init__()
        self.name = "CPU"
    
    def initialize(self) -> bool:
        self.initialized = True
        return True
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        # 降采样处理
        scale_factor = 0.7
        small_frame1 = cv2.resize(frame1, (0,0), fx=scale_factor, fy=scale_factor)
        small_frame2 = cv2.resize(frame2, (0,0), fx=scale_factor, fy=scale_factor)
        
        gray1 = cv2.cvtColor(small_frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(small_frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        return cv2.resize(thresh, (frame1.shape[1], frame1.shape[0]))
    
    def release(self):
        pass

class AccelerationManager:
    """加速后端管理器"""
    def __init__(self):
        self.backends = [
            CUDABackend(),
            OpenCLBackend(),
            CPUBackend()  # 最后一个是CPU后备
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
        self.executor = ThreadPoolExecutor(max_workers=2)  # 减少线程数
        self.frame_queue = Queue(maxsize=2)  # 更小的队列
        self.result_queue = Queue(maxsize=2)
        self.running = True
        self.use_multithread = True
        self.last_frame_time = 0
        self.frame_interval = 1/30  # 目标30FPS
        
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
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return None  # 跳过太快的帧
            
        self.last_frame_time = current_time
        
        if not self.use_multithread:
            return backend.process_frames(frame1, frame2, threshold)
            
        try:
            future = self.executor.submit(
                self.process_frame_task, 
                frame1.copy(), frame2.copy(), threshold, backend
            )
            return future.result(timeout=0.05)  # 添加超时
        except:
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

# ==================== 主程序框架 ====================

class RealTimeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.setup_globals()
        self.setup_acceleration()
        self.setup_ui()
        self.setup_status_bar()
        self.setup_tooltip()
        # self.setup_bindings()

    def setup_acceleration(self):
        """初始化加速后端管理器"""
        self.acceleration_manager = AccelerationManager()
        self.frame_processor = FrameProcessor()
        self.frame_processor.start_processing(self.threshold, self.acceleration_manager.get_current_backend())
        
    def setup_globals(self):
        """初始化全局变量"""
        self.version = "v70.2.5"
        self.author = "杜玛"
        self.copyright = "Copyright © 杜玛. All rights reserved."
        self.threshold = 30
        self.min_contour_area = 500
        self.monitoring_process = None
        self.monitoring_camera = None
        self.sct = mss()
        self.frame_rate = 60
        self.lock_aspect_ratio = True
        self.monitor_area_roi = None
        self.font_path = "msyh.ttf"
        self.max_recommended_fps = 60
        self.performance_test_done = False
        self.last_update_time = None
        self.actual_fps = 0
        self.cpu_monitor_enabled = True
        self.frame_count = 0
        self.last_boxes = []
        self.fade_frames = 3
        self.use_multithread = True
        self.last_processed_time = 0
        self.last_alert_time = 0
        self.alert_message = ""
        self.current_image = None
        self.control_panel_visible = True
        
    def setup_ui(self):
        """设置用户界面"""
        self.root.title(f"智能变化检测系统 ({self.version} | {self.author} | {self.copyright})")
        self.root.geometry("500x700")
        self.root.minsize(400, 300)

        # 主框架
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 视频显示区域
        self.video_frame = tk.Frame(self.main_frame, bg='black')
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        # 画布
        self.canvas = tk.Canvas(self.video_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 控制面板区域
        self.control_panel_frame = tk.Frame(self.main_frame, bd=2, relief=tk.RAISED)
        self.control_panel_frame.pack(fill=tk.X)

        # 控制面板内容
        self.control_panel = tk.Frame(self.control_panel_frame)
        self.control_panel.pack(fill=tk.X, padx=5, pady=5)

        # 控制面板开关按钮
        self.toggle_button = tk.Button(
            self.main_frame, 
            text="▲ 显示控制面板 ▲", 
            command=self.toggle_control_panel,
            relief=tk.RAISED,
            bd=1
        )
        self.toggle_button.pack(fill=tk.X, side=tk.BOTTOM)
        
        # 创建控制面板内容
        self.create_control_panel()

    def setup_status_bar(self):
        """多标签状态栏实现"""
        self.status_bar = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # 左侧指标区
        left_frame = tk.Frame(self.status_bar)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.fps_label = tk.Label(left_frame, text="FPS: --", width=8, anchor=tk.W)
        self.fps_label.pack(side=tk.LEFT)
        
        self.cpu_label = tk.Label(left_frame, text="CPU: --%", width=8, anchor=tk.W)
        self.cpu_label.pack(side=tk.LEFT)
        
        self.backend_label = tk.Label(left_frame, text="加速: --", width=12, anchor=tk.W)
        self.backend_label.pack(side=tk.LEFT)
        
        # 右侧警报区
        self.alert_label = tk.Label(self.status_bar, text="", fg="black", anchor=tk.W)
        self.alert_label.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def setup_tooltip(self):
        """工具提示实现"""
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.withdraw()
        self.tooltip.overrideredirect(True)
        
        self.tooltip_label = tk.Label(
            self.tooltip, bd=1, relief=tk.SOLID, 
            bg='lightyellow', justify=tk.LEFT
        )
        self.tooltip_label.pack()
        
        self.status_bar.bind("<Enter>", self.show_tooltip)
        self.status_bar.bind("<Leave>", lambda e: self.tooltip.withdraw())
    
    def show_tooltip(self, event):
        """显示完整状态信息"""
        info = f"""系统状态详情:
帧率: {self.actual_fps:.1f}/{self.frame_rate} FPS
CPU: {psutil.cpu_percent()}%
内存: {psutil.virtual_memory().percent}%
加速后端: {self.acceleration_manager.get_current_backend().name}
检测阈值: {self.threshold}
最小区域: {self.min_contour_area}px
线程数: {threading.active_count()}"""
        
        self.tooltip_label.config(text=info)
        self.tooltip.geometry(f"+{event.x_root+10}+{event.y_root+10}")
        self.tooltip.deiconify()
    
    def update_status(self, fps=None, cpu=None, alert=None, backend=None):
        """更新状态栏"""
        if fps is not None:
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            
        if cpu is not None:
            color = "red" if cpu > 80 else ("orange" if cpu > 70 else "black")
            self.cpu_label.config(text=f"CPU: {cpu}%", fg=color)
            
        if backend is not None:
            self.backend_label.config(text=f"加速: {backend}")

        if alert is not None:
            self.alert_message = alert
            self.last_alert_time = time.time()
            self.alert_label.config(text=alert, fg="red")
            self.root.after(5000, 
                lambda: self.alert_label.config(text="", fg="black"))
            
    def create_control_panel(self):
        """创建控制面板"""
        # 监控源选择区域
        source_frame = tk.LabelFrame(self.control_panel, text="监控源选择", padx=5, pady=5)
        source_frame.grid(row=0, column=0, sticky="ew", pady=(0,5))
        
        # 控制按钮
        self.process_button = tk.Button(
            source_frame, 
            text="选择监控进程", 
            command=self.toggle_process_monitoring
        )
        self.process_button.pack(side=tk.LEFT, padx=2, expand=True)
        
        self.camera_button = tk.Button(
            source_frame,
            text="选择镜头监控",
            command=self.toggle_camera_monitoring
        )
        self.camera_button.pack(side=tk.LEFT, padx=2, expand=True)
        
        self.area_button = tk.Button(
            source_frame,
            text="选择监控区域",
            command=self.select_monitor_area
        )
        self.area_button.pack(side=tk.LEFT, padx=2, expand=True)

        # 参数设置区域
        settings_frame = tk.LabelFrame(self.control_panel, text="检测参数设置", padx=5, pady=5)
        settings_frame.grid(row=1, column=0, sticky="ew")
        
        # 帧率设置
        tk.Label(settings_frame, text="帧率(FPS):").grid(row=0, column=0, padx=2, pady=2, sticky="e")
        self.frame_rate_scale = tk.Scale(
            settings_frame, 
            from_=1, 
            to=120,
            orient=tk.HORIZONTAL,
            length=150
        )
        self.frame_rate_scale.set(self.frame_rate)
        self.frame_rate_scale.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        self.frame_rate_entry = tk.Entry(settings_frame, width=4)
        self.frame_rate_entry.insert(0, str(self.frame_rate))
        self.frame_rate_entry.grid(row=0, column=2, padx=2, pady=2)
        
        # 阈值设置
        tk.Label(settings_frame, text="变化阈值:").grid(row=1, column=0, padx=2, pady=2, sticky="e")
        self.threshold_scale = tk.Scale(
            settings_frame, 
            from_=1, 
            to=100,
            orient=tk.HORIZONTAL,
            length=150
        )
        self.threshold_scale.set(self.threshold)
        self.threshold_scale.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        self.threshold_entry = tk.Entry(settings_frame, width=4)
        self.threshold_entry.insert(0, str(self.threshold))
        self.threshold_entry.grid(row=1, column=2, padx=2, pady=2)
        
        # 最小区域设置
        tk.Label(settings_frame, text="最小区域:").grid(row=2, column=0, padx=2, pady=2, sticky="e")
        self.min_area_scale = tk.Scale(
            settings_frame,
            from_=1,
            to=1000,
            orient=tk.HORIZONTAL,
            length=150
        )
        self.min_area_scale.set(self.min_contour_area)
        self.min_area_scale.grid(row=2, column=1, padx=2, pady=2, sticky="ew")
        self.min_area_entry = tk.Entry(settings_frame, width=4)
        self.min_area_entry.insert(0, str(self.min_contour_area))
        self.min_area_entry.grid(row=2, column=2, padx=2, pady=2)
        
        # 绑定事件
        self.frame_rate_scale.config(command=self.on_frame_rate_scale)
        self.threshold_scale.config(command=self.on_threshold_scale)
        self.min_area_scale.config(command=self.on_min_area_scale)
        
        self.frame_rate_entry.bind("<Return>", self.on_frame_rate_entry)
        self.threshold_entry.bind("<Return>", self.on_threshold_entry)
        self.min_area_entry.bind("<Return>", self.on_min_area_entry)
        
        # 加速后端选择
        accel_frame = tk.LabelFrame(self.control_panel, text="加速设置", padx=5, pady=5)
        accel_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        
        tk.Label(accel_frame, text="加速后端:").grid(row=0, column=0, padx=2, pady=2, sticky="e")
        
        self.backend_var = tk.StringVar()
        self.backend_menu = ttk.Combobox(
            accel_frame, 
            textvariable=self.backend_var, 
            state="readonly"
        )
        self.backend_menu['values'] = [b.name for b in self.acceleration_manager.backends]
        self.backend_menu.current(0)
        self.backend_menu.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        self.backend_menu.bind("<<ComboboxSelected>>", self.on_backend_change)
        
        # 多线程开关
        self.multithread_var = tk.BooleanVar(value=self.use_multithread)
        tk.Checkbutton(
            accel_frame, 
            text="启用多线程处理", 
            variable=self.multithread_var,
            command=self.toggle_multithread
        ).grid(row=1, column=0, columnspan=2, padx=2, pady=2, sticky="w")
        
        # 其他选项
        self.lock_aspect_var = tk.BooleanVar(value=self.lock_aspect_ratio)
        tk.Checkbutton(
            settings_frame,
            text="保持比例",
            variable=self.lock_aspect_var,
            command=self.toggle_aspect_ratio
        ).grid(row=3, column=0, padx=2, pady=2, sticky="w")
        
        self.topmost_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            settings_frame,
            text="总是置顶",
            variable=self.topmost_var,
            command=self.toggle_topmost
        ).grid(row=3, column=1, padx=2, pady=2, sticky="w")
        
        # 配置列权重
        settings_frame.columnconfigure(1, weight=1)
        self.control_panel.columnconfigure(0, weight=1)

    def toggle_control_panel(self):
        """切换控制面板可见性"""
        if self.control_panel_visible:
            self.control_panel_frame.pack_forget()
            self.toggle_button.config(text="▼ 显示控制面板 ▼")
            self.control_panel_visible = False
        else:
            self.control_panel_frame.pack(fill=tk.X, before=self.toggle_button)
            self.toggle_button.config(text="▲ 隐藏控制面板 ▲")
            self.control_panel_visible = True

    def on_frame_rate_scale(self, val):
        """帧率滑块回调"""
        self.frame_rate_entry.delete(0, tk.END)
        self.frame_rate_entry.insert(0, str(int(float(val))))
        self.frame_rate = int(float(val))
        self.update_status(alert=f"帧率已设置为: {self.frame_rate} FPS")

    def on_threshold_scale(self, val):
        """阈值滑块回调"""
        self.threshold_entry.delete(0, tk.END)
        self.threshold_entry.insert(0, str(int(float(val))))
        self.threshold = int(float(val))
        self.update_status(alert=f"变化检测阈值已设置为: {self.threshold}")

    def on_min_area_scale(self, val):
        """最小区域滑块回调"""
        self.min_area_entry.delete(0, tk.END)
        self.min_area_entry.insert(0, str(int(float(val))))
        self.min_contour_area = int(float(val))
        self.update_status(alert=f"最小变化区域已设置为: {self.min_contour_area} 像素")

    def on_frame_rate_entry(self, event):
        """帧率输入框回调"""
        try:
            value = int(self.frame_rate_entry.get())
            if 1 <= value <= 120:
                self.frame_rate_scale.set(value)
                self.frame_rate = value
                self.update_status(alert=f"帧率已设置为: {value} FPS")
        except ValueError:
            pass

    def on_threshold_entry(self, event):
        """阈值输入框回调"""
        try:
            value = int(self.threshold_entry.get())
            if 1 <= value <= 100:
                self.threshold_scale.set(value)
                self.threshold = value
                self.update_status(alert=f"变化检测阈值已设置为: {value}")
        except ValueError:
            pass

    def on_min_area_entry(self, event):
        """最小区域输入框回调"""
        try:
            value = int(self.min_area_entry.get())
            if 1 <= value <= 1000:
                self.min_area_scale.set(value)
                self.min_contour_area = value
                self.update_status(alert=f"最小变化区域已设置为: {value} 像素")
        except ValueError:
            pass

    def on_backend_change(self, event):
        """加速后端切换回调"""
        selected_backend = self.backend_var.get()
        if self.acceleration_manager.set_backend(selected_backend):
            self.update_status(
                backend=selected_backend,
                alert=f"已切换到 {selected_backend} 加速"
            )
            self.detect_max_fps()
        else:
            self.update_status(alert=f"无法切换到 {selected_backend}，使用当前后端")
            self.backend_menu.current([
                b.name for b in self.acceleration_manager.backends
            ].index(self.acceleration_manager.get_current_backend().name))

    def toggle_multithread(self):
        """切换多线程处理"""
        self.use_multithread = self.multithread_var.get()
        self.frame_processor.set_multithread(self.use_multithread)
        self.update_status(alert=f"已{'启用' if self.use_multithread else '禁用'}多线程处理")

    def toggle_aspect_ratio(self):
        """切换保持宽高比"""
        self.lock_aspect_ratio = self.lock_aspect_var.get()

    def toggle_topmost(self):
        """切换窗口置顶"""
        self.root.attributes('-topmost', self.topmost_var.get())

    def detect_max_fps(self):
        """检测最大推荐帧率"""
        test_duration = 2  # 测试2秒
        start_time = time.time()
        test_frame_count = 0
        
        # 模拟实际工作负载
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        backend = self.acceleration_manager.get_current_backend()
        
        while time.time() - start_time < test_duration:
            _ = backend.process_frames(test_frame, test_frame, self.threshold)
            test_frame_count += 1
        
        self.max_recommended_fps = min(120, max(30, test_frame_count // test_duration))
        self.performance_test_done = True
        self.update_status(alert=f"系统检测: 推荐最大帧率 {self.max_recommended_fps}FPS")
        return self.max_recommended_fps

    def adjust_processing_params(self):
        """动态调整处理参数"""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 75:  # 高负载时
            self.min_contour_area = min(1000, int(self.min_contour_area * 1.2))
            self.threshold = min(100, self.threshold + 2)
            self.frame_rate = max(15, self.frame_rate - 5)
            self.update_status(alert=f"系统负载高，自动调整参数 (CPU: {cpu_percent}%)")
        elif cpu_percent < 50:  # 低负载时
            self.min_contour_area = max(100, int(self.min_contour_area * 0.9))
            self.threshold = max(10, self.threshold - 1)
            self.frame_rate = min(60, self.frame_rate + 5)

    def update_video_display(self, frame):
        """更新视频显示"""
        if frame.shape[1] > 1280 or frame.shape[0] > 720:
            frame = cv2.resize(frame, (1280, 720))
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        img_width, img_height = img.size
        
        if self.lock_aspect_ratio:
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
        else:
            new_width = canvas_width
            new_height = canvas_height
        
        img = img.resize((new_width, new_height), Image.BILINEAR)
        self.current_image = ImageTk.PhotoImage(image=img)
        
        self.canvas.delete("all")
        self.canvas.create_image(
            (canvas_width - new_width) // 2,
            (canvas_height - new_height) // 2,
            anchor=tk.NW,
            image=self.current_image
        )

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
        """获取可用摄像头列表"""
        cameras = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(i)
                cap.release()
        return cameras

    def get_window_rect(self, hwnd):
        """获取窗口矩形区域"""
        rect = win32gui.GetWindowRect(hwnd)
        return {
            "left": rect[0],
            "top": rect[1],
            "width": rect[2] - rect[0],
            "height": rect[3] - rect[1]
        }

    def create_process_selection_window(self):
        """创建进程选择窗口"""
        selection_window = tk.Toplevel(self.root)
        selection_window.title(f"选择需要监控的程序 ({self.version} | {self.author} | {self.copyright})")
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
            processes = self.get_process_windows()
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
            processes = self.get_process_windows()
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
            self.monitoring_process = {
                'hwnd': next(p['hwnd'] for p in self.get_process_windows() if p['pid'] == int(item_data['values'][0])),
                'pid': int(item_data['values'][0]),
                'name': item_data['values'][1],
                'title': item_data['values'][2]
            }
            self.monitoring_camera = None  # 清除摄像头监控
            
            # 更新按钮状态
            self.process_button.config(text=f"停止监控 {self.monitoring_process['name']}")
            self.camera_button.config(text="选择镜头监控")
            
            selection_window.destroy()
            self.update_status(alert=f"已开始监控进程: {self.monitoring_process['name']}")
        
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

    def create_camera_selection_window(self):
        """创建摄像头选择窗口"""
        selection_window = tk.Toplevel(self.root)
        selection_window.title(f"选择监控镜头 ({self.version} | {self.author} | {self.copyright})")
        selection_window.geometry("400x300")
        
        # 获取可用摄像头
        cameras = self.get_available_cameras()
        if not cameras:
            self.update_status(alert="没有检测到可用的摄像头")
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
                self.update_status(alert="请先选择一个摄像头")
                return
            
            item_data = tree.item(selected_item)
            self.monitoring_camera = int(item_data['values'][0])
            self.monitoring_process = None  # 清除进程监控
            
            # 更新按钮状态
            self.camera_button.config(text=f"停止监控 摄像头{self.monitoring_camera}")
            self.process_button.config(text="选择监控进程")
            
            selection_window.destroy()
            self.update_status(alert=f"已开始监控摄像头: {self.monitoring_camera}")
        
        # 取消按钮
        def on_cancel():
            selection_window.destroy()
        
        button_frame = tk.Frame(selection_window)
        button_frame.pack(pady=10)
        
        select_button = tk.Button(button_frame, text="确定", command=on_select)
        select_button.pack(side=tk.LEFT, padx=10)
        
        cancel_button = tk.Button(button_frame, text="取消", command=on_cancel)
        cancel_button.pack(side=tk.LEFT, padx=10)

    def create_camera_selection_window(self):
        """创建摄像头选择窗口"""
        selection_window = tk.Toplevel(self.root)
        selection_window.title(f"选择监控镜头 ({self.version} | {self.author} | {self.copyright})")
        selection_window.geometry("400x300")
        
        # 获取可用摄像头
        cameras = self.get_available_cameras()
        if not cameras:
            self.update_status(alert="没有检测到可用的摄像头")
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
                self.update_status(alert="请先选择一个摄像头")
                return
            
            item_data = tree.item(selected_item)
            self.monitoring_camera = int(item_data['values'][0])
            self.monitoring_process = None  # 清除进程监控
            
            # 更新按钮状态
            self.camera_button.config(text=f"停止监控 摄像头{self.monitoring_camera}")
            self.process_button.config(text="选择监控进程")
            
            selection_window.destroy()
            self.update_status(alert=f"已开始监控摄像头: {self.monitoring_camera}")
        
        # 取消按钮
        def on_cancel():
            selection_window.destroy()
        
        button_frame = tk.Frame(selection_window)
        button_frame.pack(pady=10)
        
        select_button = tk.Button(button_frame, text="确定", command=on_select)
        select_button.pack(side=tk.LEFT, padx=10)
        
        cancel_button = tk.Button(button_frame, text="取消", command=on_cancel)
        cancel_button.pack(side=tk.LEFT, padx=10)

    def select_monitor_area(self):
        """选择监控区域"""
        if not self.monitoring_process:
            messagebox.showwarning("警告", "请先选择要监控的进程")
            return
        
        # 获取窗口句柄
        hwnd = self.monitoring_process['hwnd']
        
        # 创建临时窗口用于选择区域
        temp_window = tk.Toplevel(self.root)
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
            x1, y1 = min(start_x, end_x), min(start_y, end_y)
            x2, y2 = max(start_x, end_x), max(start_y, end_y)
            
            # 确保选择的区域有效
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # 最小10像素的宽度和高度
                self.monitor_area_roi = (x1, y1, x2-x1, y2-y1)
                self.update_status(alert=f"已设置监控区域: {self.monitor_area_roi}")
            else:
                self.update_status(alert="选择区域太小，请重新选择")
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

    def toggle_process_monitoring(self):
        """切换进程监控"""
        if self.monitoring_process:
            self.monitoring_process = None
            self.process_button.config(text="选择监控进程")
            self.update_status(alert="已停止监控")
        else:
            self.monitoring_camera = None
            self.camera_button.config(text="选择镜头监控")
            self.create_process_selection_window()

    def toggle_camera_monitoring(self):
        """切换摄像头监控"""
        if self.monitoring_camera is not None:
            self.monitoring_camera = None
            self.camera_button.config(text="选择镜头监控")
            self.update_status(alert="已停止监控")
        else:
            self.monitoring_process = None
            self.process_button.config(text="选择监控进程")
            self.create_camera_selection_window()

    def select_monitor_area(self):
        """选择监控区域"""
        if not self.monitoring_process:
            messagebox.showwarning("警告", "请先选择要监控的进程")
            return
        
        # 获取窗口句柄
        hwnd = self.monitoring_process['hwnd']
        
        # 创建临时窗口用于选择区域
        temp_window = tk.Toplevel(self.root)
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
            x1, y1 = min(start_x, end_x), min(start_y, end_y)
            x2, y2 = max(start_x, end_x), max(start_y, end_y)
            
            # 确保选择的区域有效
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # 最小10像素的宽度和高度
                self.monitor_area_roi = (x1, y1, x2-x1, y2-y1)
                self.update_status(alert=f"已设置监控区域: {self.monitor_area_roi}")
            else:
                self.update_status(alert="选择区域太小，请重新选择")
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

    def update_frame(self):
        """更新帧的主循环"""
        start_time = time.perf_counter()
        self.frame_count += 1
        
        # 动态参数调整
        if self.frame_count % 30 == 0:
            self.adjust_processing_params()
        
        # 帧率控制
        current_time = time.time()
        if self.last_update_time is None:
            self.last_update_time = current_time
            
        frame_interval = current_time - self.last_update_time
        self.last_update_time = current_time
        
        target_fps = min(self.frame_rate, self.max_recommended_fps)
        min_frame_time = 1.0 / target_fps
        
        # 如果处理太快，等待一下
        if frame_interval < min_frame_time:
            time.sleep(min_frame_time - frame_interval)
        
        # 获取帧数据
        if self.monitoring_camera is not None:
            # 摄像头模式
            cap = cv2.VideoCapture(self.monitoring_camera)
            if not cap.isOpened():
                self.update_status(alert=f"无法打开摄像头 {self.monitoring_camera}")
                self.root.after(1000, self.update_frame)  # 1秒后重试
                return
            
            ret, current_frame = cap.read()
            if not ret:
                self.update_status(alert="无法读取视频帧")
                cap.release()
                self.root.after(30, self.update_frame)
                return
            
            ret, next_frame = cap.read()
            if not ret:
                self.update_status(alert="无法读取视频帧")
                cap.release()
                self.root.after(30, self.update_frame)
                return
            
            cap.release()
        elif self.monitoring_process:
            # 进程监控模式
            try:
                # 获取窗口位置
                rect = self.get_window_rect(self.monitoring_process['hwnd'])
                monitor_area = {
                    "left": rect["left"],
                    "top": rect["top"],
                    "width": rect["width"],
                    "height": rect["height"]
                }
                
                # 如果设置了监控区域ROI，则调整监控区域
                if self.monitor_area_roi:
                    x, y, w, h = self.monitor_area_roi
                    monitor_area["left"] += x
                    monitor_area["top"] += y
                    monitor_area["width"] = w
                    monitor_area["height"] = h

                # 读取两帧用于比较
                current_frame = np.array(self.sct.grab(monitor_area))
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2BGR)
                
                next_frame = np.array(self.sct.grab(monitor_area))
                next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGRA2BGR)

                # 计算实际处理耗时
                process_time = (time.time() - start_time) * 1000  # 毫秒
                if process_time > 10:  # 如果单次处理超过10ms
                    self.update_status(alert=f"警告：处理延迟 {process_time:.1f}ms")

            except Exception as e:
                self.update_status(alert=f"监控出错: {str(e)}")
                self.root.after(1000, self.update_frame)  # 1秒后重试
                return
        else:
            # 没有选择监控源
            current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            try:
                from PIL import ImageDraw, ImageFont
                img_pil = Image.fromarray(current_frame)
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype(self.font_path, 30)
                draw.text((150, 200), "请选择监控源", font=font, fill=(255, 255, 255))
                font = ImageFont.truetype(self.font_path, 20)
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
            self.update_video_display(current_frame)
            self.root.after(30, self.update_frame)
            return

        # 使用当前加速后端处理帧
        backend = self.acceleration_manager.get_current_backend()
        
        # 根据设置选择处理方式
        if self.use_multithread:
            thresh = self.frame_processor.process_frames(current_frame, next_frame, self.threshold, backend)
        else:
            thresh = backend.process_frames(current_frame, next_frame, self.threshold)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建副本帧用于绘制
        display_frame = current_frame.copy()

        # 在原始帧上绘制轮廓
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:  # 过滤掉小的变化
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 绘制带渐隐效果的检测框
        current_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                current_boxes.append((x, y, w, h))
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 渐隐旧框
        for i, (x, y, w, h) in enumerate(self.last_boxes):
            if (x, y, w, h) not in current_boxes:
                alpha = i / self.fade_frames
                color = (0, int(255*alpha), 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 1)

        self.last_boxes = current_boxes[-10:]

        # 更新视频显示
        self.update_video_display(current_frame)

        # 计算实际帧率
        current_time = time.time()
        if self.last_update_time:
            self.actual_fps = 0.9 * self.actual_fps + 0.1 * (1 / (current_time - self.last_update_time))  # 平滑处理
        self.last_update_time = current_time

        # 更新状态
        self.update_status(
            fps=self.actual_fps,
            cpu=psutil.cpu_percent(),
            backend=self.acceleration_manager.get_current_backend().name
        )
        
        # 检查警报
        if self.alert_message and time.time() - self.last_alert_time < 5:
            self.alert_label.config(text=self.alert_message, fg="red")
        
        # 帧率控制
        effective_fps = min(self.frame_rate, self.max_recommended_fps)
        delay = max(1, int(1000 / effective_fps))
        self.root.after(delay, self.update_frame)

    def cleanup(self):
        """清理资源"""
        if self.monitoring_camera is not None:
            cv2.VideoCapture(self.monitoring_camera).release()
        self.acceleration_manager.release_all()
        self.frame_processor.stop_processing()

# ==================== 程序入口 ====================

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeDetectionApp(root)
    
    # 初始化加速后端
    # app.acceleration_manager = AccelerationManager()
    # app.frame_processor = FrameProcessor()
    # app.frame_processor.start_processing(app.threshold, app.acceleration_manager.get_current_backend())
    
    # 启动主循环
    app.update_frame()
    
    # 设置关闭回调
    root.protocol("WM_DELETE_WINDOW", lambda: [app.cleanup(), root.destroy()])
    
    root.mainloop()
