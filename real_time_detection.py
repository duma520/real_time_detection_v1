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
    
    def initialize(self) -> bool:
        try:
            import torch
            self.backend_info["torch_version"] = torch.__version__
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cpu':
                return False
                
            self.initialized = True
            return True
        except Exception as e:
            print(f"PyTorch初始化失败: {str(e)}")
            return False
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold: int) -> np.ndarray:
        if not self.initialized or self.device is None:
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
            PyTorchBackend(),
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

# ==================== 主程序 ====================

# 全局变量
version = "v61.20.18"  # 版本
author = "杜玛"
copyrigh = "Copyright © 杜玛. All rights reserved."
threshold = 30  # 变化检测阈值
min_contour_area = 500  # 最小变化区域
monitoring_process = None  # 当前监控的进程
monitoring_camera = None  # 当前监控的摄像头
sct = mss()  # 屏幕截图对象
frame_rate = 60  # 帧率
lock_aspect_ratio = True  # 是否锁定宽高比
monitor_area_roi = None  # 监控区域的ROI
font_path = "msyh.ttf"
max_recommended_fps = 60
performance_test_done = False
last_update_time = None  # 用于计算实际帧率
actual_fps = 0          # 记录实际帧率
cpu_monitor_enabled = True  # 是否启用CPU监控
frame_count = 0  # 用于计数帧数
last_boxes = []  # 存储上一帧的检测框
fade_frames = 3  # 渐隐帧数，控制渐隐速度
acceleration_manager = AccelerationManager()

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

# 定义更新帧的函数
def update_frame():
    global last_boxes
    # 性能监控
    global last_update_time, actual_fps
    global frame_count

    # 统一性能计时起点
    start_time = time.time()
    frame_count += 1
    current_time = time.time()

    # CPU占用监控（每10帧检测一次）
    if cpu_monitor_enabled and monitoring_process and frame_count % 10 == 0:
        cpu_percent = psutil.cpu_percent(interval=0.1)

        if cpu_percent > 80 and frame_rate > 60:
            status_bar.config(
                text=f"⚠️ CPU过载: {cpu_percent}% | 实际FPS: {actual_fps:.1f}/{frame_rate}",
                fg="red"
            )
    
    # 实际帧率计算
    current_time = time.time()
    if last_update_time:
        actual_fps = 0.9 * actual_fps + 0.1 * (1 / (current_time - last_update_time))  # 平滑处理
    last_update_time = current_time

    if monitoring_camera is not None:
        # 摄像头模式
        cap = cv2.VideoCapture(monitoring_camera)
        if not cap.isOpened():
            status_bar.config(text=f"无法打开摄像头 {monitoring_camera}")
            root.after(1000, update_frame)  # 1秒后重试
            return
        
        ret, current_frame = cap.read()
        if not ret:
            status_bar.config(text="无法读取视频帧")
            cap.release()
            root.after(30, update_frame)
            return
        
        ret, next_frame = cap.read()
        if not ret:
            status_bar.config(text="无法读取视频帧")
            cap.release()
            root.after(30, update_frame)
            return
        
        cap.release()
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
            status_bar.config(text=f"监控出错: {str(e)}", fg="red")
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

    # 使用当前加速后端处理帧
    backend = acceleration_manager.get_current_backend()
    try:
        thresh = backend.process_frames(current_frame, next_frame, threshold)
        # 确保 thresh 是 uint8 类型
        if thresh.dtype != np.uint8:
            thresh = thresh.astype(np.uint8)
    except Exception as e:
        print(f"{backend.name}处理失败: {str(e)}，回退到CPU")
        acceleration_manager.set_backend("CPU")
        backend = acceleration_manager.get_current_backend()
        thresh = backend.process_frames(current_frame, next_frame, threshold)
        # 确保 thresh 是 uint8 类型
        if thresh.dtype != np.uint8:
            thresh = thresh.astype(np.uint8)

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
    
    # 渐隐旧框
    for i, (x, y, w, h) in enumerate(last_boxes):
        if (x, y, w, h) not in current_boxes:
            alpha = i / fade_frames
            color = (0, int(255*alpha), 255)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 1)

    last_boxes = current_boxes[-10:]

    # 更新视频显示
    update_video_display(current_frame)

    # 计算实际帧率
    current_time = time.time()
    if last_update_time:
        actual_fps = 0.9 * actual_fps + 0.1 * (1 / (current_time - last_update_time))
    last_update_time = current_time

    # 构建状态栏文本
    backend = acceleration_manager.get_current_backend()
    status_text = f"运行中: {actual_fps:.1f}FPS | 加速: {backend.name}"
    if frame_rate > 120 or abs(actual_fps - frame_rate) > 10:
        status_text += f" (设置: {frame_rate}FPS)"
        status_bar.config(fg="red")
    
    if 'cpu_percent' in locals() and cpu_percent > 80:
        status_text += f" | CPU: {cpu_percent}% ⚠️ CPU过载"
        status_bar.config(fg="red")
    else:
        status_bar.config(fg="black")
    
    status_bar.config(text=status_text)

    # 帧率控制
    effective_fps = min(frame_rate, 360)
    delay = max(1, int(1000 / effective_fps))
    root.after(delay, update_frame)

# 创建控制面板（添加加速后端选择）
def create_control_panel():
    global frame_rate_scale, threshold_scale, min_area_scale
    global frame_rate_entry, threshold_entry, min_area_entry

    # 监控源选择区域
    source_frame = tk.LabelFrame(control_panel, text="监控源选择", padx=5, pady=5)
    source_frame.grid(row=0, column=0, sticky="ew", pady=(0,5))
    
    # 使用紧凑的按钮布局
    buttons = [
        ("选择监控进程", toggle_process_monitoring),
        ("选择镜头监控", toggle_camera_monitoring),
        ("选择监控区域", select_monitor_area)
    ]
    
    for col, (text, cmd) in enumerate(buttons):
        btn = tk.Button(source_frame, text=text, command=cmd)
        btn.grid(row=0, column=col, padx=2, pady=2, sticky="ew")
        source_frame.columnconfigure(col, weight=1)
        if text.startswith("选择监控进程"):
            global process_button
            process_button = btn
        elif text.startswith("选择镜头监控"):
            global camera_button
            camera_button = btn

    # 参数设置区域
    settings_frame = tk.LabelFrame(control_panel, text="检测参数设置", padx=5, pady=5)
    settings_frame.grid(row=1, column=0, sticky="ew")
    
    # 参数配置项
    params = [
        ("帧率(FPS):", "frame_rate", 1, 360),
        ("变化阈值:", "threshold", 1, 100),
        ("最小区域:", "min_contour_area", 1, 1000)
    ]
    
    # 先创建所有控件
    for row, (label_text, var_name, min_val, max_val) in enumerate(params):
        tk.Label(settings_frame, text=label_text).grid(row=row, column=0, padx=2, pady=2, sticky="e")
        
        # 创建滑块
        if var_name == "frame_rate":
            frame_rate_scale = tk.Scale(
                settings_frame, 
                from_=1, 
                to=max_recommended_fps if performance_test_done else 360,  # 动态上限
                orient=tk.HORIZONTAL,
                length=150
            )
            frame_rate_scale.set(min(60, max_recommended_fps))
            frame_rate_scale.set(globals()[var_name])
            frame_rate_scale.grid(row=row, column=1, padx=2, pady=2, sticky="ew")
            
            frame_rate_entry = tk.Entry(settings_frame, width=4)
            frame_rate_entry.insert(0, str(globals()[var_name]))
            frame_rate_entry.grid(row=row, column=2, padx=2, pady=2)
            
        elif var_name == "threshold":
            threshold_scale = tk.Scale(settings_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, 
                                     length=150)
            threshold_scale.set(globals()[var_name])
            threshold_scale.grid(row=row, column=1, padx=2, pady=2, sticky="ew")
            
            threshold_entry = tk.Entry(settings_frame, width=4)
            threshold_entry.insert(0, str(globals()[var_name]))
            threshold_entry.grid(row=row, column=2, padx=2, pady=2)
            
        elif var_name == "min_contour_area":
            min_area_scale = tk.Scale(settings_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, 
                                     length=150)
            min_area_scale.set(globals()[var_name])
            min_area_scale.grid(row=row, column=1, padx=2, pady=2, sticky="ew")
            
            min_area_entry = tk.Entry(settings_frame, width=4)
            min_area_entry.insert(0, str(globals()[var_name]))
            min_area_entry.grid(row=row, column=2, padx=2, pady=2)
    
    # 然后绑定事件处理函数
    def on_frame_rate_scale(val):
        frame_rate_entry.delete(0, tk.END)
        frame_rate_entry.insert(0, str(int(float(val))))
        globals()["frame_rate"] = int(float(val))
        status_bar.config(text=f"帧率已设置为: {int(float(val))} FPS")
    
    def on_threshold_scale(val):
        threshold_entry.delete(0, tk.END)
        threshold_entry.insert(0, str(int(float(val))))
        globals()["threshold"] = int(float(val))
        status_bar.config(text=f"变化检测阈值已设置为: {int(float(val))}")
    
    def on_min_area_scale(val):
        min_area_entry.delete(0, tk.END)
        min_area_entry.insert(0, str(int(float(val))))
        globals()["min_contour_area"] = int(float(val))
        status_bar.config(text=f"最小变化区域已设置为: {int(float(val))} 像素")
    
    frame_rate_scale.config(command=on_frame_rate_scale)
    threshold_scale.config(command=on_threshold_scale)
    min_area_scale.config(command=on_min_area_scale)
    
    def on_frame_rate_entry(event):
        try:
            value = int(frame_rate_entry.get())
            if 1 <= value <= 120:
                frame_rate_scale.set(value)
                globals()["frame_rate"] = value
                status_bar.config(text=f"帧率已设置为: {value} FPS")
        except ValueError:
            pass
    
    def on_threshold_entry(event):
        try:
            value = int(threshold_entry.get())
            if 1 <= value <= 100:
                threshold_scale.set(value)
                globals()["threshold"] = value
                status_bar.config(text=f"变化检测阈值已设置为: {value}")
        except ValueError:
            pass
    
    def on_min_area_entry(event):
        try:
            value = int(min_area_entry.get())
            if 1 <= value <= 1000:
                min_area_scale.set(value)
                globals()["min_contour_area"] = value
                status_bar.config(text=f"最小变化区域已设置为: {value} 像素")
        except ValueError:
            pass
    
    frame_rate_entry.bind("<Return>", on_frame_rate_entry)
    threshold_entry.bind("<Return>", on_threshold_entry)
    min_area_entry.bind("<Return>", on_min_area_entry)        

    # 加速后端选择区域
    accel_frame = tk.LabelFrame(control_panel, text="加速设置", padx=5, pady=5)
    accel_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))
    
    tk.Label(accel_frame, text="加速后端:").grid(row=0, column=0, padx=2, pady=2, sticky="e")
    
    backend_var = tk.StringVar()
    backend_menu = ttk.Combobox(accel_frame, textvariable=backend_var, state="readonly")
    backend_menu['values'] = [backend.name for backend in acceleration_manager.backends]
    backend_menu.current([backend.name for backend in acceleration_manager.backends].index(
        acceleration_manager.get_current_backend().name))
    backend_menu.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
    
    def on_backend_change(event):
        selected_backend = backend_var.get()
        if acceleration_manager.set_backend(selected_backend):
            status_bar.config(text=f"已切换到 {selected_backend} 加速")
            # 重新检测最大帧率
            detect_max_fps()
            # 更新帧率滑块上限
            frame_rate_scale.config(to=max_recommended_fps if performance_test_done else 360)
        else:
            status_bar.config(text=f"无法切换到 {selected_backend}，使用当前后端", fg="red")
            backend_menu.current([backend.name for backend in acceleration_manager.backends].index(
                acceleration_manager.get_current_backend().name))
    
    backend_menu.bind("<<ComboboxSelected>>", on_backend_change)
    
    # 复选框设置
    lock_aspect_var = tk.BooleanVar(value=lock_aspect_ratio)
    tk.Checkbutton(
        settings_frame, 
        text="保持比例", 
        variable=lock_aspect_var,
        command=lambda: globals().update(lock_aspect_ratio=lock_aspect_var.get())
    ).grid(row=len(params), column=0, padx=2, pady=2, sticky="w")
    
    tk.Checkbutton(
        settings_frame,
        text="总是置顶",
        command=lambda: root.attributes('-topmost', not root.attributes('-topmost'))
    ).grid(row=len(params), column=1, padx=2, pady=2, sticky="w")
    
    # 配置列权重
    settings_frame.columnconfigure(1, weight=1)
    control_panel.columnconfigure(0, weight=1)

# 创建控制面板
create_control_panel()

# 启动更新帧的函数
update_frame()

# 运行主循环
root.mainloop()

# 释放资源
if monitoring_camera is not None:
    cv2.VideoCapture(monitoring_camera).release()
acceleration_manager.release_all()