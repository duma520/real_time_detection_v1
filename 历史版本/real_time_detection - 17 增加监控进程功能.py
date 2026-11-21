# Version 17

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
monitor_window = None  # 监控窗口
sct = mss()  # 屏幕截图对象

# 解析命令行参数
parser = argparse.ArgumentParser(description="实时变化检测")
parser.add_argument("--camera", action="store_true", help="使用摄像头作为输入源")
args = parser.parse_args()

# 创建主窗口
root = tk.Tk()
root.title("控制面板窗口")
root.geometry("500x300")  # 设置主窗口大小

# 打开摄像头
if args.camera:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
else:
    cap = None

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
        global monitoring_process
        monitoring_process = {
            'hwnd': next(p['hwnd'] for p in processes if p['pid'] == int(item_data['values'][0])),
            'pid': int(item_data['values'][0]),
            'name': item_data['values'][1],
            'title': item_data['values'][2]
        }
        
        # 更新监控按钮状态
        monitor_button.config(text=f"停止监控 {monitoring_process['name']}")
        
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

# 开始/停止监控进程
def toggle_process_monitoring():
    global monitoring_process, monitor_window
    
    if monitoring_process:
        # 停止监控
        monitoring_process = None
        monitor_button.config(text="选择监控进程")
        
        # 关闭监控窗口
        if monitor_window:
            monitor_window.destroy()
            monitor_window = None
        
        messagebox.showinfo("提示", "已停止监控")
    else:
        # 开始监控 - 打开进程选择窗口
        create_process_selection_window()

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

# 定义更新帧的函数
def update_frame():
    if args.camera:
        # 摄像头模式
        ret, current_frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            return
        
        ret, next_frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            return
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

    # 显示图像 变化区域结果面板窗口
    cv2.imshow("Change Detection Result Panel", current_frame)
    cv2.setWindowProperty("Change Detection Result Panel", cv2.WND_PROP_TOPMOST, int(always_on_top_result_var.get()))

    # 将焦点返回到控制面板窗口
    root.focus_force()

    # 递归调用，持续更新帧
    root.after(30, update_frame)

# 创建控制面板
control_frame = tk.Frame(root)
control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

# 监控进程按钮
monitor_button = tk.Button(control_frame, text="选择监控进程" if not monitoring_process else f"停止监控 {monitoring_process['name']}", 
                          command=toggle_process_monitoring)
monitor_button.pack(pady=5)

# 透明度调整 (仅在非进程监控模式下显示)
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
    except ValueError:
        pass

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
        if 10 <= value <= 1000:
            frame_rate_scale.set(value)
            globals().update(frame_rate=value)
    except ValueError:
        pass

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
if args.camera:
    cap.release()
cv2.destroyAllWindows()
