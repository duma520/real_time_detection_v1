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
import os


# 全局变量
version = "v61.0.14"  # 版本
threshold = 30  # 变化检测阈值
min_contour_area = 500  # 最小变化区域
monitoring_process = None  # 当前监控的进程
monitoring_camera = None  # 当前监控的摄像头
monitor_window = None  # 监控窗口
sct = mss()  # 屏幕截图对象
frame_rate = 30  # 帧率
result_window_size = (800, 600)  # 结果窗口初始大小
lock_aspect_ratio = True  # 是否锁定宽高比
icon_cache = {}  # 进程图标缓存
monitor_area_roi = None  # 监控区域的ROI
selecting_monitor_area = False  # 是否正在选择监控区域


# 创建主窗口
root = tk.Tk()
root.title(f"智能变化检测系统 - 控制面板 ({version})")
root.geometry("500x520")  # 设置主窗口大小
root.resizable(False, False)  # 禁止改变窗口大小


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
    selection_window.title(f"选择需要监控的程序 ({version})")
    selection_window.geometry("750x500")
    
    # 创建搜索框
    search_frame = tk.Frame(selection_window)
    search_frame.pack(fill=tk.X, padx=10, pady=5)
    
    search_label = tk.Label(search_frame, text="搜索:")
    search_label.pack(side=tk.LEFT)
    
    search_var = tk.StringVar()
    search_entry = tk.Entry(search_frame, textvariable=search_var)
    search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # 图标缓存
    icon_cache = {}
    
    # 填充进程列表的函数
    def update_process_list(treeview):
        processes = get_process_windows()
        treeview.delete(*treeview.get_children())
        
        for process in processes:
            treeview.insert('', 'end', values=(process['pid'], process['name'], process['title']))
    
    refresh_button = tk.Button(search_frame, text="刷新", command=lambda: update_process_list(tree))
    refresh_button.pack(side=tk.RIGHT, padx=5)
    
    # 创建进程列表 - 注意列名要与values中的顺序一致
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
        
        # 如果存在监控窗口，则关闭
        global monitor_window
        if monitor_window:
            monitor_window.destroy()
            monitor_window = None
        
        selection_window.destroy()
        status_label.config(text=f"已开始监控进程: {monitoring_process['name']}")
    
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
    election_window.title(f"选择监控镜头 ({version})")
    selection_window.geometry("400x300")
    
    # 获取可用摄像头
    cameras = get_available_cameras()
    if not cameras:
        status_label.config(text="没有检测到可用的摄像头")
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
            status_label.config(text="请先选择一个摄像头")
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
        status_label.config(text=f"已开始监控摄像头: {monitoring_camera}")
    
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
        status_label.config(text="已停止监控")
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
        status_label.config(text="已停止监控")
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
    monitor_window.title(f"监控面板窗口 ({version})")
    monitor_window.geometry("800x600")  # 设置监控窗口大小
    
    # 创建画布用于显示视频帧
    canvas = tk.Canvas(monitor_window)
    canvas.pack(fill=tk.BOTH, expand=True)  # 画布填充整个监控窗口

# 鼠标事件处理函数
def on_mouse_event(event, x, y, flags, current_frame):
    global result_window_roi, selecting_roi, roi_start_point, roi_end_point
    
    try:
        # 获取窗口位置和大小（包括边框和标题栏）
        window_x, window_y, window_w, window_h = cv2.getWindowImageRect("Change Detection Result Panel")
        if window_w <= 0 or window_h <= 0:
            return
    except:
        return
    
    # 获取原始图像尺寸
    img_h, img_w = current_frame.shape[:2]
    
    # 计算显示区域的宽高（保持宽高比）
    display_ratio = img_w / img_h if img_h > 0 else 1.0
    window_ratio = window_w / window_h if window_h > 0 else 1.0
    if display_ratio > window_ratio:
        # 宽度受限
        display_w = window_w
        display_h = int(window_w / display_ratio)
        offset_x = 0
        offset_y = (window_h - display_h) // 2
    else:
        # 高度受限
        display_h = window_h
        display_w = int(window_h * display_ratio)
        offset_x = (window_w - display_w) // 2
        offset_y = 0
    
    # 计算缩放比例
    scale_x = display_w / img_w if img_w > 0 else 1
    scale_y = display_h / img_h if img_h > 0 else 1
    
    # 调整鼠标坐标（减去偏移并考虑缩放）
    adj_x = (x - offset_x) / scale_x if scale_x > 0 else 0
    adj_y = (y - offset_y) / scale_y if scale_y > 0 else 0
    
    # 将调整后的鼠标坐标映射回原始图像坐标
    orig_x = int(adj_x / scale_x) if scale_x > 0 else 0
    orig_y = int(adj_y / scale_y) if scale_y > 0 else 0
    
    # 确保坐标在图像范围内
    orig_x = max(0, min(orig_x, img_w - 1))
    orig_y = max(0, min(orig_y, img_h - 1))
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 开始选择ROI
        selecting_roi = True
        roi_start_point = (orig_x, orig_y)
        roi_end_point = (orig_x, orig_y)
        
    elif event == cv2.EVENT_MOUSEMOVE and selecting_roi:
        # 更新ROI选择框
        roi_end_point = (orig_x, orig_y)
        
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
    
        # 计算ROI区域相对于原图的比例
        img_h, img_w = current_frame.shape[:2]
        if img_w > 0 and img_h > 0:
            roi_ratio = (x2-x1) / (y2-y1) if (y2-y1) > 0 else 1.0
        
            # 获取当前窗口大小
            try:
                _, _, current_w, current_h = cv2.getWindowImageRect("Change Detection Result Panel")
                if lock_aspect_ratio:
                    # 保持ROI的宽高比
                    roi_ratio = (x2-x1) / (y2-y1) if (y2-y1) > 0 else 1.0
                    cv2.setWindowProperty("Change Detection Result Panel", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    
                    # 计算新窗口大小，保持ROI的宽高比
                    if result_window_size[0]/result_window_size[1] > roi_ratio:
                        # 宽度受限
                        new_w = int(result_window_size[0] * (x2-x1)/img_w)
                        new_h = int(new_w / roi_ratio)
                    else:
                        # 高度受限
                        new_h = int(result_window_size[1] * (y2-y1)/img_h)
                        new_w = int(new_h * roi_ratio)
                else:
                    # 不保持宽高比
                    cv2.setWindowProperty("Change Detection Result Panel", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
                    # 直接按比例缩放
                    new_w = int(result_window_size[0] * (x2-x1)/img_w)
                    new_h = int(result_window_size[1] * (y2-y1)/img_h)

                
                    cv2.resizeWindow("Change Detection Result Panel", new_w, new_h)
            except:
                pass
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右键点击重置ROI
        result_window_roi = None



# 重置ROI区域
def reset_roi():
    global result_window_roi
    result_window_roi = None
    status_label.config(text="已重置显示区域")
    # 重置后重新应用宽高比
    if lock_aspect_ratio:
        cv2.setWindowProperty("Change Detection Result Panel", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    # 恢复窗口原始大小
    cv2.resizeWindow("Change Detection Result Panel", result_window_size[0], result_window_size[1])

def select_monitor_area():
    global selecting_monitor_area, monitor_area_roi
    
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
            status_label.config(text=f"已设置监控区域: {monitor_area_roi}")
        else:
            status_label.config(text="选择区域太小，请重新选择")
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



# 调整结果窗口大小
def resize_result_window():
    global result_window_roi
    cv2.namedWindow("Change Detection Result Panel", cv2.WINDOW_NORMAL)
    if lock_aspect_ratio:
        if result_window_roi:
            # 如果有ROI，按ROI比例调整
            aspect_ratio = result_window_roi[2] / result_window_roi[3]
            new_h = int(result_window_size[0] / aspect_ratio)
            cv2.resizeWindow("Change Detection Result Panel", result_window_size[0], new_h)
        else:
            # 如果没有ROI，保持图像原始宽高比
            img = cv2.imread("temp.jpg") if os.path.exists("temp.jpg") else np.zeros((600,800,3), np.uint8)
            aspect_ratio = img.shape[1] / img.shape[0]
            new_h = int(result_window_size[0] / aspect_ratio)
            cv2.resizeWindow("Change Detection Result Panel", result_window_size[0], new_h)
        cv2.setWindowProperty("Change Detection Result Panel", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    else:
        cv2.setWindowProperty("Change Detection Result Panel", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        cv2.resizeWindow("Change Detection Result Panel", result_window_size[0], result_window_size[1])
    status_label.config(text="已调整结果窗口大小")



# 定义更新帧的函数
def update_frame():
    if monitoring_camera is not None:
        # 摄像头模式
        cap = cv2.VideoCapture(monitoring_camera)
        if not cap.isOpened():
            status_label.config(text=f"无法打开摄像头 {monitoring_camera}")
            root.after(1000, update_frame)  # 1秒后重试
            return
        
        ret, current_frame = cap.read()
        if not ret:
            status_label.config(text="无法读取视频帧")
            cap.release()
            root.after(30, update_frame)
            return
        
        ret, next_frame = cap.read()
        if not ret:
            status_label.config(text="无法读取视频帧")
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
        except Exception as e:
            status_label.config(text=f"监控窗口出错: {e}")
            root.after(1000, update_frame)  # 1秒后重试
            return
    else:
        # 没有选择监控源
        root.after(30, update_frame)
        return

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

    # 显示图像
    cv2.namedWindow("Change Detection Result Panel", cv2.WINDOW_NORMAL)
    
    # 设置窗口属性
    cv2.setWindowProperty("Change Detection Result Panel", cv2.WND_PROP_TOPMOST, int(always_on_top_result_var.get()))
    
    if lock_aspect_ratio:
        # 强制设置宽高比
        img_h, img_w = current_frame.shape[:2]
        aspect_ratio = img_w / img_h
        current_w, current_h = cv2.getWindowImageRect("Change Detection Result Panel")[2:]
        if current_w > 0 and current_h > 0:
            if abs(current_w/current_h - aspect_ratio) > 0.01:  # 检查当前比例是否与图像比例不一致
                new_h = int(current_w / aspect_ratio)
                cv2.resizeWindow("Change Detection Result Panel", current_w, new_h)
    else:
        cv2.setWindowProperty("Change Detection Result Panel", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
    cv2.imshow("Change Detection Result Panel", current_frame)


    # 将焦点返回到控制面板窗口
    root.focus_force()

    # 递归调用，持续更新帧
    root.after(30, update_frame)

# 创建控制面板
control_frame = tk.Frame(root)
control_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

# 创建监控源选择区域
source_frame = tk.LabelFrame(control_frame, text="监控源选择", padx=10, pady=10)
source_frame.pack(fill=tk.X, pady=5)

# 监控进程按钮
process_button = tk.Button(source_frame, text="选择监控进程", command=toggle_process_monitoring)
process_button.pack(side=tk.LEFT, padx=5, pady=5)

# 监控摄像头按钮
camera_button = tk.Button(source_frame, text="选择镜头监控", command=toggle_camera_monitoring)
camera_button.pack(side=tk.LEFT, padx=5, pady=5)

# 创建ROI控制区域
roi_frame = tk.LabelFrame(control_frame, text="监控区域控制", padx=10, pady=10)
roi_frame.pack(fill=tk.X, pady=5)

resize_button = tk.Button(roi_frame, text="调整结果窗口大小", command=resize_result_window)
resize_button.pack(side=tk.LEFT, padx=5, pady=5)

select_monitor_button = tk.Button(roi_frame, text="选择目标监控区域", command=select_monitor_area)
select_monitor_button.pack(side=tk.LEFT, padx=5, pady=5)

# 创建参数设置区域
settings_frame = tk.LabelFrame(control_frame, text="检测参数设置", padx=10, pady=10)
settings_frame.pack(fill=tk.X, pady=5)

# 帧的读取频率调整
frame_rate_frame = tk.Frame(settings_frame)
frame_rate_frame.pack(fill=tk.X, pady=5)

frame_rate_label = tk.Label(frame_rate_frame, text="帧率(FPS):")
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
            status_label.config(text=f"帧率已设置为: {value} FPS")
    except ValueError:
        pass

def update_frame_rate_from_scale(val):
    frame_rate_entry.delete(0, tk.END)
    frame_rate_entry.insert(0, str(val))
    globals().update(frame_rate=int(val))
    status_label.config(text=f"帧率已设置为: {val} FPS")

frame_rate_scale.config(command=update_frame_rate_from_scale)
frame_rate_entry.bind("<Return>", lambda event: update_frame_rate_from_entry())

# 变化检测阈值调整
threshold_frame = tk.Frame(settings_frame)
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
            status_label.config(text=f"变化检测阈值已设置为: {value}")
    except ValueError:
        pass

def update_threshold_from_scale(val):
    threshold_entry.delete(0, tk.END)
    threshold_entry.insert(0, str(val))
    globals().update(threshold=int(val))
    status_label.config(text=f"变化检测阈值已设置为: {val}")

threshold_scale.config(command=update_threshold_from_scale)
threshold_entry.bind("<Return>", lambda event: update_threshold_from_entry())

# 最小变化区域调整
min_area_frame = tk.Frame(settings_frame)
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
            status_label.config(text=f"最小变化区域已设置为: {value} 像素")
    except ValueError:
        pass

def update_min_area_from_scale(val):
    min_area_entry.delete(0, tk.END)
    min_area_entry.insert(0, str(val))
    globals().update(min_contour_area=int(val))
    status_label.config(text=f"最小变化区域已设置为: {val} 像素")

min_area_scale.config(command=update_min_area_from_scale)
min_area_entry.bind("<Return>", lambda event: update_min_area_from_entry())

# 是否置顶变化区域结果面板复选框
always_on_top_result_var = tk.BooleanVar(value=False)
always_on_top_result_check = tk.Checkbutton(settings_frame, text="总是置顶变化结果面板", variable=always_on_top_result_var,
                                           command=lambda: cv2.setWindowProperty("Change Detection Result Panel", cv2.WND_PROP_TOPMOST, int(always_on_top_result_var.get())))
always_on_top_result_check.pack(pady=5)

lock_aspect_var = tk.BooleanVar(value=lock_aspect_ratio)
lock_aspect_check = tk.Checkbutton(settings_frame, text="锁定结果窗口宽高比", variable=lock_aspect_var,
                                  command=lambda: [
                                      globals().update(lock_aspect_ratio=lock_aspect_var.get()),
                                      cv2.setWindowProperty("Change Detection Result Panel", cv2.WND_PROP_ASPECT_RATIO, 
                                                          cv2.WINDOW_KEEPRATIO if lock_aspect_var.get() else cv2.WINDOW_FREERATIO)
                                  ])

lock_aspect_check.pack(pady=5)

# 状态显示标签
status_frame = tk.Frame(root)
status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

status_label = tk.Label(status_frame, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(fill=tk.X)

# 启动更新帧的函数
update_frame()

# 运行主循环
root.mainloop()

# 释放资源
if monitoring_camera is not None:
    cv2.VideoCapture(monitoring_camera).release()
cv2.destroyAllWindows()
