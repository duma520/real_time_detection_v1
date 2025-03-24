# Version 22

import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import argparse
from mss import mss
import win32gui
import win32process
import psutil
import re

# 全局变量
transparency = 0.5  # 监控窗口透明度
threshold = 30      # 变化检测阈值
min_contour_area = 500  # 最小变化区域
always_on_top = False  # 是否置顶监控面板窗口
monitoring_process = None  # 当前监控的进程
monitoring_camera = None  # 当前监控的摄像头
monitor_window = None  # 监控窗口
sct = mss()  # 屏幕截图对象
result_window_roi = None  # 结果窗口的ROI区域
selecting_roi = False    # 是否正在选择ROI
roi_start_point = None   # ROI选择的起点
roi_end_point = None     # ROI选择的终点
frame_rate = 30  # 添加全局变量定义

# 创建主窗口
root = tk.Tk()
root.title("控制面板窗口")
root.geometry("500x450")  # 设置主窗口大小

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
    selection_window.title("选择需要监控的程序")
    selection_window.geometry("600x400")
    
    # 创建进程列表
    tree = ttk.Treeview(selection_window, columns=('pid', 'name', 'title'), show='headings')
    tree.heading('pid', text='PID')
    tree.heading('name', text='进程名')
    tree.heading('title', text='窗口标题')
    tree.column('pid', width=80)
    tree.column('name', width=120)
    tree.column('title', width=400)
    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 添加滚动条
    scrollbar = ttk.Scrollbar(selection_window, orient="vertical", command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # 填充进程列表
    processes = get_process_windows()
    for process in processes:
        tree.insert('', 'end', values=(process['pid'], process['name'], process['title']))
    
    # 确定按钮
    def on_select():
        selected_item = tree.focus()
        if not selected_item:
            messagebox.showwarning("警告", "请先选择一个进程")
            return
        
        item_data = tree.item(selected_item)
        global monitoring_process, monitoring_camera
        monitoring_process = {
            'hwnd': next(p['hwnd'] for p in processes if p['pid'] == int(item_data['values'][0])),
            'pid': int(item_data['values'][0]),
            'name': item_data['values'][1],
            'title': item_data['values'][2]
        }
        monitoring_camera = None  # 清除摄像头监控
        
        # 更新按钮状态
        process_button.config(text=f"停止监控 {monitoring_process['name']}")
        camera_button.config(text="选择镜头监控")
        
        # 如果存在监控窗口，则关闭
        global monitor_window
        if monitor_window:
            monitor_window.destroy()
            monitor_window = None
        
        selection_window.destroy()
        messagebox.showinfo("提示", f"已开始监控进程: {monitoring_process['name']}")
    
    # 取消按钮
    def on_cancel():
        selection_window.destroy()
    
    button_frame = tk.Frame(selection_window)
    button_frame.pack(pady=10)
    
    select_button = tk.Button(button_frame, text="确定", command=on_select)
    select_button.pack(side=tk.LEFT, padx=10)
    
    cancel_button = tk.Button(button_frame, text="取消", command=on_cancel)
    cancel_button.pack(side=tk.LEFT, padx=10)

# 创建摄像头选择窗口
def create_camera_selection_window():
    selection_window = tk.Toplevel(root)
    selection_window.title("选择监控镜头")
    selection_window.geometry("400x300")
    
    # 获取可用摄像头
    cameras = get_available_cameras()
    if not cameras:
        messagebox.showwarning("警告", "没有检测到可用的摄像头")
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
            messagebox.showwarning("警告", "请先选择一个摄像头")
            return
        
        item_data = tree.item(selected_item)
        global monitoring_camera, monitoring_process
        monitoring_camera = int(item_data['values'][0])
        monitoring_process = None  # 清除进程监控
        
        # 更新按钮状态
        camera_button.config(text=f"停止监控 摄像头{monitoring_camera}")
        process_button.config(text="选择监控进程")
        
        # 如果存在监控窗口，则关闭
        global monitor_window
        if monitor_window:
            monitor_window.destroy()
            monitor_window = None
        
        selection_window.destroy()
        messagebox.showinfo("提示", f"已开始监控摄像头: {monitoring_camera}")
    
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
        messagebox.showinfo("提示", "已停止监控")
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
        messagebox.showinfo("提示", "已停止监控")
    else:
        # 开始监控 - 打开摄像头选择窗口
        monitoring_process = None  # 清除进程监控
        process_button.config(text="选择监控进程")
        create_camera_selection_window()

# 创建监控窗口
def create_monitor_window():
    global monitor_window
    
    if monitor_window:
        monitor_window.destroy()
    
    monitor_window = tk.Toplevel(root)
    monitor_window.title("监控面板窗口")
    monitor_window.geometry("800x600")  # 设置监控窗口大小
    monitor_window.attributes("-alpha", transparency)  # 设置初始透明度
    
    # 创建画布用于显示视频帧
    canvas = tk.Canvas(monitor_window)
    canvas.pack(fill=tk.BOTH, expand=True)  # 画布填充整个监控窗口

# 鼠标事件处理函数
def on_mouse_event(event, x, y, flags, param):
    global result_window_roi, selecting_roi, roi_start_point, roi_end_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 开始选择ROI
        selecting_roi = True
        roi_start_point = (x, y)
        roi_end_point = (x, y)
        
    elif event == cv2.EVENT_MOUSEMOVE and selecting_roi:
        # 更新ROI选择框
        roi_end_point = (x, y)
        
    elif event == cv2.EVENT_LBUTTONUP and selecting_roi:
        # 结束选择ROI
        selecting_roi = False
        x1, y1 = roi_start_point
        x2, y2 = roi_end_point
        
        # 确保x1,y1是左上角，x2,y2是右下角
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 设置ROI区域
        result_window_roi = (x1, y1, x2-x1, y2-y1)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右键点击重置ROI
        result_window_roi = None

# 重置ROI区域
def reset_roi():
    global result_window_roi
    result_window_roi = None

# 调整结果窗口大小
def resize_result_window():
    cv2.namedWindow("Change Detection Result Panel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Change Detection Result Panel", 800, 600)

# 定义更新帧的函数
def update_frame():
    global selecting_roi, roi_start_point, roi_end_point
    
    if monitoring_camera is not None:
        # 摄像头模式
        cap = cv2.VideoCapture(monitoring_camera)
        if not cap.isOpened():
            print(f"无法打开摄像头 {monitoring_camera}")
            root.after(1000, update_frame)  # 1秒后重试
            return
        
        ret, current_frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            cap.release()
            root.after(30, update_frame)
            return
        
        ret, next_frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            cap.release()
            root.after(30, update_frame)
            return
        
        cap.release()
    elif monitoring_process:
        # 进程监控模式
        try:
            # 获取窗口位置
            rect = get_window_rect(monitoring_process['hwnd'])
            monitor_area = {
                "left": rect["left"],
                "top": rect["top"],
                "width": rect["width"],
                "height": rect["height"]
            }
            
            # 读取两帧用于比较
            current_frame = np.array(sct.grab(monitor_area))
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2BGR)
            
            next_frame = np.array(sct.grab(monitor_area))
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"监控窗口出错: {e}")
            root.after(1000, update_frame)  # 1秒后重试
            return
    else:
        # 透明窗口模式
        if not monitor_window:
            root.after(30, update_frame)
            return
        
        x = monitor_window.winfo_x()
        y = monitor_window.winfo_y()
        width = monitor_window.winfo_width()
        height = monitor_window.winfo_height()
        monitor_area = {"top": y, "left": x, "width": width, "height": height}
        
        current_frame = np.array(sct.grab(monitor_area))
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2BGR)
        
        next_frame = np.array(sct.grab(monitor_area))
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGRA2BGR)

    # 转换为灰度图
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # 计算帧间差异
    frame_diff = cv2.absdiff(current_gray, next_gray)

    # 二值化差异图像
    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始帧上绘制轮廓
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:  # 过滤掉小的变化
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 应用ROI裁剪（如果设置了ROI）
    if result_window_roi:
        x, y, w, h = result_window_roi
        # 确保ROI在图像范围内
        h, w_total = current_frame.shape[:2]
        x = max(0, min(x, w_total-1))
        y = max(0, min(y, h-1))
        w = max(1, min(w, w_total-x))
        h = max(1, min(h, h-y))
        current_frame = current_frame[y:y+h, x:x+w]

    # 显示图像 变化区域结果面板窗口
    cv2.imshow("Change Detection Result Panel", current_frame)
    
    # 设置鼠标回调函数用于选择ROI
    cv2.setMouseCallback("Change Detection Result Panel", on_mouse_event)
    
    # 如果正在选择ROI，绘制选择框
    if selecting_roi and roi_start_point and roi_end_point:
        temp_frame = current_frame.copy()
        cv2.rectangle(temp_frame, roi_start_point, roi_end_point, (255, 0, 0), 2)
        cv2.imshow("Change Detection Result Panel", temp_frame)

    cv2.setWindowProperty("Change Detection Result Panel", cv2.WND_PROP_TOPMOST, int(always_on_top_result_var.get()))

    # 将焦点返回到控制面板窗口
    root.focus_force()

    # 递归调用，持续更新帧
    root.after(30, update_frame)

# 创建控制面板
control_frame = tk.Frame(root)
control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

# 监控进程按钮
process_button = tk.Button(control_frame, text="选择监控进程", command=toggle_process_monitoring)
process_button.pack(pady=5)

# 监控摄像头按钮
camera_button = tk.Button(control_frame, text="选择镜头监控", command=toggle_camera_monitoring)
camera_button.pack(pady=5)

# ROI控制按钮
roi_control_frame = tk.Frame(control_frame)
roi_control_frame.pack(fill=tk.X, pady=5)

reset_roi_button = tk.Button(roi_control_frame, text="重置显示区域", command=reset_roi)
reset_roi_button.pack(side=tk.LEFT, padx=5)

resize_button = tk.Button(roi_control_frame, text="调整结果窗口大小", command=resize_result_window)
resize_button.pack(side=tk.LEFT, padx=5)

# 透明度调整
transparency_frame = tk.Frame(control_frame)
transparency_frame.pack(fill=tk.X, pady=5)

transparency_label = tk.Label(transparency_frame, text="监控窗口透明度:")
transparency_label.pack(side=tk.LEFT)

transparency_scale = tk.Scale(transparency_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                            length=300, command=lambda val: monitor_window.attributes("-alpha", float(val)) if monitor_window else None)
transparency_scale.set(transparency)
transparency_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

transparency_entry = tk.Entry(transparency_frame, width=5)
transparency_entry.insert(0, str(transparency))
transparency_entry.pack(side=tk.LEFT, padx=5)

def update_transparency_from_entry():
    try:
        value = float(transparency_entry.get())
        if 0.1 <= value <= 1.0:
            transparency_scale.set(value)
            if monitor_window:
                monitor_window.attributes("-alpha", value)
            globals().update(transparency=value)
    except ValueError:
        pass

def update_transparency_from_scale(val):
    value = float(val)
    transparency_entry.delete(0, tk.END)
    transparency_entry.insert(0, str(value))
    if monitor_window:
        monitor_window.attributes("-alpha", value)
    globals().update(transparency=value)

transparency_scale.config(command=update_transparency_from_scale)
transparency_entry.bind("<Return>", lambda event: update_transparency_from_entry())

transparency_entry.bind("<Return>", lambda event: update_transparency_from_entry())

# 帧的读取频率调整
frame_rate_frame = tk.Frame(control_frame)
frame_rate_frame.pack(fill=tk.X, pady=5)

frame_rate_label = tk.Label(frame_rate_frame, text="帧的读取频率:")
frame_rate_label.pack(side=tk.LEFT)

frame_rate_scale = tk.Scale(frame_rate_frame, from_=1, to=120, orient=tk.HORIZONTAL, length=300,
                            command=lambda val: globals().update(frame_rate=int(val)))
frame_rate_scale.set(30)  # 默认值为30
frame_rate_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

frame_rate_entry = tk.Entry(frame_rate_frame, width=5)
frame_rate_entry.insert(0, "30")  # 默认值为30
frame_rate_entry.pack(side=tk.LEFT, padx=5)

def update_frame_rate_from_entry():
    try:
        value = int(frame_rate_entry.get())
        if 1 <= value <= 120:
            frame_rate_scale.set(value)
            globals().update(frame_rate=value)
    except ValueError:
        pass

def update_frame_rate_from_scale(val):
    frame_rate_entry.delete(0, tk.END)
    frame_rate_entry.insert(0, str(val))
    globals().update(frame_rate=int(val))

frame_rate_scale.config(command=update_frame_rate_from_scale)
frame_rate_entry.bind("<Return>", lambda event: update_frame_rate_from_entry())

# 变化检测阈值调整
threshold_frame = tk.Frame(control_frame)
threshold_frame.pack(fill=tk.X, pady=5)

threshold_label = tk.Label(threshold_frame, text="变化检测阈值:")
threshold_label.pack(side=tk.LEFT)

threshold_scale = tk.Scale(threshold_frame, from_=1, to=100, orient=tk.HORIZONTAL, length=300,
                           command=lambda val: globals().update(threshold=int(val)))
threshold_scale.set(threshold)
threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

threshold_entry = tk.Entry(threshold_frame, width=5)
threshold_entry.insert(0, str(threshold))
threshold_entry.pack(side=tk.LEFT, padx=5)

def update_threshold_from_entry():
    try:
        value = int(threshold_entry.get())
        if 1 <= value <= 100:
            threshold_scale.set(value)
            globals().update(threshold=value)
    except ValueError:
        pass

def update_threshold_from_scale(val):
    threshold_entry.delete(0, tk.END)
    threshold_entry.insert(0, str(val))
    globals().update(threshold=int(val))

threshold_scale.config(command=update_threshold_from_scale)
threshold_entry.bind("<Return>", lambda event: update_threshold_from_entry())

# 最小变化区域调整
min_area_frame = tk.Frame(control_frame)
min_area_frame.pack(fill=tk.X, pady=5)

min_area_label = tk.Label(min_area_frame, text="最小变化区域:")
min_area_label.pack(side=tk.LEFT)

min_area_scale = tk.Scale(min_area_frame, from_=1, to=1000, orient=tk.HORIZONTAL, length=300,
                          command=lambda val: globals().update(min_contour_area=int(val)))
min_area_scale.set(min_contour_area)
min_area_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

min_area_entry = tk.Entry(min_area_frame, width=5)
min_area_entry.insert(0, str(min_contour_area))
min_area_entry.pack(side=tk.LEFT, padx=5)

def update_min_area_from_entry():
    try:
        value = int(min_area_entry.get())
        if 1 <= value <= 5000:
            min_area_scale.set(value)
            globals().update(min_contour_area=value)
    except ValueError:
        pass

def update_min_area_from_scale(val):
    min_area_entry.delete(0, tk.END)
    min_area_entry.insert(0, str(val))
    globals().update(min_contour_area=int(val))

min_area_scale.config(command=update_min_area_from_scale)
min_area_entry.bind("<Return>", lambda event: update_min_area_from_entry())

# 是否置顶复选框
always_on_top_var = tk.BooleanVar(value=always_on_top)
always_on_top_check = tk.Checkbutton(control_frame, text="总是置顶监控面板", variable=always_on_top_var,
                                    command=lambda: monitor_window.attributes("-topmost", always_on_top_var.get()) if monitor_window else None)
always_on_top_check.pack(pady=5)

# 是否置顶变化区域结果面板复选框
always_on_top_result_var = tk.BooleanVar(value=False)
always_on_top_result_check = tk.Checkbutton(control_frame, text="总是置顶变化结果面板", variable=always_on_top_result_var,
                                           command=lambda: cv2.setWindowProperty("Change Detection Result Panel", cv2.WND_PROP_TOPMOST, int(always_on_top_result_var.get())))
always_on_top_result_check.pack(pady=5)

# 启动更新帧的函数
update_frame()

# 运行主循环
root.mainloop()

# 释放资源
if monitoring_camera is not None:
    cv2.VideoCapture(monitoring_camera).release()
cv2.destroyAllWindows()
