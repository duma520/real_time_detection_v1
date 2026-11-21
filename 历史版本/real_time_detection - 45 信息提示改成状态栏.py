# Version 45 - 添加状态信息显示系统

import cv2
import tkinter as tk
from tkinter import ttk
import numpy as np
from mss import mss
import win32gui
import win32process
import psutil
import queue
import threading
import time

# 全局变量
version = "v45"
threshold = 30
min_contour_area = 500
monitoring_process = None
monitoring_camera = None
result_window_roi = None
selecting_roi = False
roi_start_point = None
roi_end_point = None
frame_rate = 30
running = True
result_window_created = False
control_panel_resizable = False
monitor_window = None
last_window_topmost_state = False
task_lock = threading.Lock()
last_frame_time = 0
window_operation_lock = threading.Lock()
last_window_state_check = time.time()
window_initialized = False
status_clear_timer = None  # 状态清除定时器

# 状态显示函数
def show_status(message, level="info"):
    """显示状态信息
    level: info, warning, error, success, system
    """
    global status_clear_timer
    
    # 取消之前的定时器
    if status_clear_timer:
        root.after_cancel(status_clear_timer)
    
    # 定义不同级别的颜色
    colors = {
        "info": "#333333",      # 默认灰色
        "warning": "orange",    # 橙色警告
        "error": "red",         # 红色错误
        "success": "green",     # 绿色成功
        "system": "blue"        # 蓝色系统消息
    }
    
    # 设置状态文本和颜色
    status_label.config(text=message, fg=colors.get(level, "#333333"))
    
    # 如果不是info消息，3秒后自动清除
    if level != "info":
        status_clear_timer = root.after(3000, lambda: status_label.config(
            text="就绪", 
            fg="#333333"
        ))

# 线程安全的停止函数
def stop_monitoring():
    global running, result_window_created, window_initialized, last_window_state_check
    
    running = False
    last_window_state_check = time.time()
    show_status("系统: 正在停止监控...", "system")
    
    # 发送停止信号
    for _ in range(3):
        try:
            task_queue.put("stop", timeout=0.1)
        except queue.Full:
            pass
    
    # 清理窗口资源
    try:
        if result_window_created:
            cv2.destroyWindow("Change Detection Result Panel")
    except:
        pass
    
    with window_operation_lock:
        result_window_created = False
        window_initialized = False
    
    time.sleep(0.2)  # 给线程时间响应
    cleanup_resources()
    show_status("监控已停止", "info")

# 创建任务队列
task_queue = queue.Queue(maxsize=20)
frame_queue = queue.Queue(maxsize=5)

# 更新阈值函数
def update_threshold_from_scale(val):
    threshold_entry.delete(0, tk.END)
    threshold_entry.insert(0, str(val))
    globals().update(threshold=int(val))
    show_status(f"变化阈值已更新: {val}", "info")

def update_threshold_from_entry():
    try:
        value = int(threshold_entry.get())
        if 1 <= value <= 100:
            threshold_scale.set(value)
            globals().update(threshold=value)
            show_status(f"变化阈值已更新: {value}", "info")
        else:
            show_status("阈值必须在1-100之间", "warning")
    except ValueError:
        show_status("请输入有效的数字", "error")

# 更新最小区域函数
def update_min_area_from_scale(val):
    min_area_entry.delete(0, tk.END)
    min_area_entry.insert(0, str(val))
    globals().update(min_contour_area=int(val))
    show_status(f"最小区域已更新: {val}", "info")

def update_min_area_from_entry():
    try:
        value = int(min_area_entry.get())
        if 1 <= value <= 5000:
            min_area_scale.set(value)
            globals().update(min_contour_area=value)
            show_status(f"最小区域已更新: {value}", "info")
        else:
            show_status("最小区域必须在1-5000之间", "warning")
    except ValueError:
        show_status("请输入有效的数字", "error")

# 更新帧率函数
def update_frame_rate_from_scale(val):
    frame_rate_entry.delete(0, tk.END)
    frame_rate_entry.insert(0, str(val))
    globals().update(frame_rate=int(val))
    show_status(f"帧率已更新: {val} FPS", "info")

def update_frame_rate_from_entry():
    try:
        value = int(frame_rate_entry.get())
        if 1 <= value <= 120:
            frame_rate_scale.set(value)
            globals().update(frame_rate=value)
            show_status(f"帧率已更新: {value} FPS", "info")
        else:
            show_status("帧率必须在1-120之间", "warning")
    except ValueError:
        show_status("请输入有效的数字", "error")

# 获取所有可见窗口的进程列表
def get_process_windows():
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
    
    tree = ttk.Treeview(selection_window, columns=('pid', 'name', 'title'), show='headings')
    tree.heading('pid', text='PID')
    tree.heading('name', text='进程名')
    tree.heading('title', text='窗口标题')
    tree.column('pid', width=80)
    tree.column('name', width=120)
    tree.column('title', width=400)
    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    scrollbar = ttk.Scrollbar(selection_window, orient="vertical", command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)
    
    processes = get_process_windows()
    for process in processes:
        tree.insert('', 'end', values=(process['pid'], process['name'], process['title']))
    
    def on_select():
        selected_item = tree.focus()
        if not selected_item:
            show_status("请先选择一个进程", "warning")
            return
        
        item_data = tree.item(selected_item)
        global monitoring_process, monitoring_camera
        monitoring_process = {
            'hwnd': next(p['hwnd'] for p in processes if p['pid'] == int(item_data['values'][0])),
            'pid': int(item_data['values'][0]),
            'name': item_data['values'][1],
            'title': item_data['values'][2]
        }
        monitoring_camera = None
        
        process_button.config(text=f"停止监控 {monitoring_process['name']}")
        camera_button.config(text="选择镜头监控")
        
        selection_window.destroy()
        show_status(f"已开始监控进程: {monitoring_process['name']}", "success")
    
    def on_cancel():
        selection_window.destroy()
        show_status("已取消进程选择", "info")
    
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
    
    cameras = get_available_cameras()
    if not cameras:
        show_status("没有检测到可用的摄像头", "error")
        selection_window.destroy()
        return
    
    tree = ttk.Treeview(selection_window, columns=('id', 'status'), show='headings')
    tree.heading('id', text='摄像头ID')
    tree.heading('status', text='状态')
    tree.column('id', width=100)
    tree.column('status', width=200)
    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    for cam_id in cameras:
        tree.insert('', 'end', values=(cam_id, "可用"))
    
    def on_select():
        selected_item = tree.focus()
        if not selected_item:
            show_status("请先选择一个摄像头", "warning")
            return
        
        item_data = tree.item(selected_item)
        global monitoring_camera, monitoring_process
        monitoring_camera = int(item_data['values'][0])
        monitoring_process = None
        
        camera_button.config(text=f"停止监控 摄像头{monitoring_camera}")
        process_button.config(text="选择监控进程")
        
        global monitor_window
        if monitor_window:
            monitor_window.destroy()
            monitor_window = None
        
        selection_window.destroy()
        show_status(f"已开始监控摄像头: {monitoring_camera}", "success")
    
    def on_cancel():
        selection_window.destroy()
        show_status("已取消摄像头选择", "info")
    
    button_frame = tk.Frame(selection_window)
    button_frame.pack(pady=10)
    
    select_button = tk.Button(button_frame, text="确定", command=on_select)
    select_button.pack(side=tk.LEFT, padx=10)
    
    cancel_button = tk.Button(button_frame, text="取消", command=on_cancel)
    cancel_button.pack(side=tk.LEFT, padx=10)

# 开始/停止监控进程
def toggle_process_monitoring():
    global monitoring_process, monitoring_camera, running, result_window_created
    
    if monitoring_process:
        # 先停止监控
        monitoring_process = None
        process_button.config(text="选择监控进程")
        show_status("已停止进程监控", "info")
        stop_monitoring()
    else:
        monitoring_camera = None
        camera_button.config(text="选择镜头监控")
        running = True
        with window_operation_lock:
            result_window_created = False
        create_process_selection_window()

# 开始/停止监控摄像头
def toggle_camera_monitoring():
    global monitoring_camera, monitoring_process, result_window_created
    
    if monitoring_camera is not None:
        # 先停止监控
        monitoring_camera = None
        camera_button.config(text="选择镜头监控")
        show_status("已停止摄像头监控", "info")
        stop_monitoring()
    else:
        monitoring_process = None
        process_button.config(text="选择监控进程")
        with window_operation_lock:
            result_window_created = False
        create_camera_selection_window()

# 清理资源函数
def cleanup_resources():
    global result_window_created
    with window_operation_lock:
        if not result_window_created:
            try:
                cv2.destroyWindow("Change Detection Result Panel")
            except:
                pass
        result_window_created = False
    
    time.sleep(0.2)

    # 清空队列
    while not task_queue.empty():
        try:
            task_queue.get_nowait()
        except queue.Empty:
            break
    
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break

# 创建监控窗口
def create_monitor_window():
    global monitor_window
    
    if monitor_window:
        monitor_window.destroy()
    
    monitor_window = tk.Toplevel(root)
    monitor_window.title("监控面板窗口")
    monitor_window.geometry("800x600")
    
    canvas = tk.Canvas(monitor_window)
    canvas.pack(fill=tk.BOTH, expand=True)

# 鼠标事件处理函数
def on_mouse_event(event, x, y, flags, param):
    global result_window_roi, selecting_roi, roi_start_point, roi_end_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting_roi = True
        roi_start_point = (x, y)
        roi_end_point = (x, y)
        show_status("正在选择ROI区域...", "info")
        
    elif event == cv2.EVENT_MOUSEMOVE and selecting_roi:
        roi_end_point = (x, y)
        
    elif event == cv2.EVENT_LBUTTONUP and selecting_roi:
        selecting_roi = False
        x1, y1 = roi_start_point
        x2, y2 = roi_end_point
        
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        result_window_roi = (x1, y1, x2-x1, y2-y1)
        show_status(f"ROI区域已设置: {result_window_roi}", "success")
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        result_window_roi = None
        show_status("ROI区域已重置", "info")

# 重置ROI区域
def reset_roi():
    global result_window_roi
    result_window_roi = None
    show_status("ROI区域已重置", "info")

# 调整结果窗口大小
def resize_result_window():
    if result_window_created:
        cv2.namedWindow("Change Detection Result Panel", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Change Detection Result Panel", 800, 600)
        show_status("结果窗口大小已调整", "info")

# 创建结果窗口
def create_result_window():
    global result_window_created, window_initialized
    try:
        if not result_window_created:
            try:
                cv2.destroyWindow("Change Detection Result Panel")
            except:
                pass
            
            cv2.namedWindow("Change Detection Result Panel", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Change Detection Result Panel", 800, 600)
            cv2.setMouseCallback("Change Detection Result Panel", on_mouse_event)
            
            try:
                visible = cv2.getWindowProperty("Change Detection Result Panel", cv2.WND_PROP_VISIBLE)
                if visible >= 0:
                    result_window_created = True
                    window_initialized = True
                    show_status("结果窗口已创建", "success")
                else:
                    raise RuntimeError("窗口创建后不可见")
            except:
                raise RuntimeError("无法验证窗口状态")
    except Exception as e:
        show_status(f"创建结果窗口失败: {e}", "error")
        result_window_created = False
        window_initialized = False
        time.sleep(0.1)

# 控制面板可调整大小
def toggle_control_panel_resizable():
    global control_panel_resizable
    control_panel_resizable = control_panel_resizable_var.get()
    if control_panel_resizable:
        root.resizable(True, True)
        show_status("控制面板已设为可调整大小", "info")
    else:
        root.resizable(False, False)
        show_status("控制面板已设为固定大小", "info")

# 帧处理线程
def frame_processing_thread():
    global running, last_frame_time
    
    sct = mss()
    cap = None
    
    while running:
        try:
            if not task_queue.empty():
                task = task_queue.get(timeout=0.1)
            else:
                task = "process_frame"

            if not running:
                break
                
            if task == "process_frame":
                current_time = time.time()
                if current_time - last_frame_time < (1.0 / frame_rate):
                    continue
                last_frame_time = current_time

                if not (monitoring_process or monitoring_camera is not None):
                    continue
                    
                current_frame = None
                
                if monitoring_camera is not None:
                    try:
                        cap = cv2.VideoCapture(monitoring_camera)
                        if cap.isOpened():
                            ret, current_frame = cap.read()
                            if ret:
                                ret, next_frame = cap.read()
                                if ret:
                                    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                                    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
                                    frame_diff = cv2.absdiff(current_gray, next_gray)
                                    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
                                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    
                                    for contour in contours:
                                        if cv2.contourArea(contour) > min_contour_area:
                                            x, y, w, h = cv2.boundingRect(contour)
                                            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    except Exception as e:
                        show_status(f"摄像头处理出错: {e}", "error")
                    finally:
                        if cap is not None:
                            cap.release()
                            cap = None
                        
                elif monitoring_process:
                    try:
                        rect = get_window_rect(monitoring_process['hwnd'])
                        monitor_area = {
                            "left": rect["left"],
                            "top": rect["top"],
                            "width": rect["width"],
                            "height": rect["height"]
                        }
                        
                        current_frame = np.array(sct.grab(monitor_area))
                        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2BGR)
                        
                        next_frame = np.array(sct.grab(monitor_area))
                        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGRA2BGR)
                        
                        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
                        frame_diff = cv2.absdiff(current_gray, next_gray)
                        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            if cv2.contourArea(contour) > min_contour_area:
                                x, y, w, h = cv2.boundingRect(contour)
                                cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    except Exception as e:
                        show_status(f"窗口监控出错: {e}", "error")
                
                if current_frame is not None and result_window_roi:
                    x, y, w, h = result_window_roi
                    h_total, w_total = current_frame.shape[:2]
                    x = max(0, min(x, w_total-1))
                    y = max(0, min(y, h_total-1))
                    w = max(1, min(w, w_total-x))
                    h = max(1, min(h, h_total-y))
                    current_frame = current_frame[y:y+h, x:x+w]
                
                if current_frame is not None:
                    try:
                        frame_queue.put_nowait(current_frame)
                    except queue.Full:
                        pass
            
            elif task == "stop":
                show_status("正在停止处理线程...", "system")
                break
                
        except queue.Empty:
            continue
        except Exception as e:
            show_status(f"处理线程出错: {e}", "error")
            break
    
    # 清理资源
    if cap is not None:
        cap.release()
    sct.close()
    show_status("帧处理线程已停止", "info")

# 显示线程（完整实现）
def display_thread():
    global running, last_window_topmost_state, result_window_created, window_initialized, last_window_state_check
    
    last_window_state_check = time.time()
    
    while running:
        try:
            # 检查窗口状态
            current_time = time.time()
            if current_time - last_window_state_check > 0.1:
                last_window_state_check = current_time
                
                if result_window_created:
                    try:
                        visible = cv2.getWindowProperty("Change Detection Result Panel", cv2.WND_PROP_VISIBLE)
                        if visible < 1:
                            with window_operation_lock:
                                result_window_created = False
                                window_initialized = False
                            continue
                    except:
                        with window_operation_lock:
                            result_window_created = False
                        window_initialized = False
                        continue
            
            # 如果有监控源但窗口未初始化，则创建窗口
            if (monitoring_process or monitoring_camera is not None) and not window_initialized:
                with window_operation_lock:
                    create_result_window()
                    if not result_window_created:
                        time.sleep(0.5)
                        continue
            
            # 尝试获取帧数据
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                if not window_initialized and (monitoring_process or monitoring_camera is not None):
                    with window_operation_lock:
                        create_result_window()
                continue
            
            # 确保结果窗口存在
            if not result_window_created or not window_initialized:
                with window_operation_lock:
                    create_result_window()
                    if not result_window_created:
                        continue
            
            # 显示帧
            try:
                if selecting_roi and roi_start_point and roi_end_point:
                    temp_frame = frame.copy()
                    cv2.rectangle(temp_frame, roi_start_point, roi_end_point, (255, 0, 0), 2)
                    cv2.imshow("Change Detection Result Panel", temp_frame)
                else:
                    cv2.imshow("Change Detection Result Panel", frame)
                
                # 更新窗口置顶状态
                current_topmost = always_on_top_result_var.get()
                if current_topmost != last_window_topmost_state and result_window_created:
                    try:
                        with window_operation_lock:
                            cv2.setWindowProperty("Change Detection Result Panel", 
                                               cv2.WND_PROP_TOPMOST, 
                                               int(current_topmost))
                            time.sleep(0.02)
                        last_window_topmost_state = current_topmost
                    except:
                        pass
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键退出
                    running = False
                    break
                    
            except Exception as e:
                show_status(f"显示帧失败: {e}", "error")
                with window_operation_lock:
                    result_window_created = False
                window_initialized = False
                
        except Exception as e:
            show_status(f"显示线程异常: {e}", "error")
            if not running:
                break
    
    # 线程结束时的清理工作
    try:
        if result_window_created:
            cv2.destroyWindow("Change Detection Result Panel")
    except:
        pass
    
    with window_operation_lock:
        result_window_created = False
        window_initialized = False
    
    time.sleep(0.2)
    cleanup_resources()
    show_status("显示线程已停止", "info")

# 主窗口创建和布局
root = tk.Tk()
root.title("智能变化检测系统 - 控制面板")
root.geometry("500x500")
root.resizable(False, False)

# 创建控制面板
main_frame = tk.Frame(root, padx=10, pady=10)
main_frame.pack(fill=tk.BOTH, expand=True)

# 顶部标题
header_frame = tk.Frame(main_frame)
header_frame.pack(fill=tk.X, pady=(0, 10))

title_label = tk.Label(header_frame, 
                      text="智能变化检测系统", 
                      font=("Arial", 14, "bold"))
title_label.pack(side=tk.LEFT)

version_label = tk.Label(header_frame, 
                        text=version, 
                        font=("Arial", 10),
                        fg="gray")
version_label.pack(side=tk.RIGHT)

# 分隔线
ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

# 监控源选择区域
source_frame = tk.LabelFrame(main_frame, 
                           text=" 监控源选择 ", 
                           padx=10, 
                           pady=5,
                           font=("Arial", 9, "bold"))
source_frame.pack(fill=tk.X, pady=5)

button_frame = tk.Frame(source_frame)
button_frame.pack(fill=tk.X, pady=5)

process_button = tk.Button(button_frame, 
                         text="选择监控进程", 
                         command=toggle_process_monitoring,
                         width=15)
process_button.pack(side=tk.LEFT, padx=5, expand=True)

camera_button = tk.Button(button_frame, 
                         text="选择镜头监控", 
                         command=toggle_camera_monitoring,
                         width=15)
camera_button.pack(side=tk.LEFT, padx=5, expand=True)

# 分隔线
ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

# 显示控制区域
display_frame = tk.LabelFrame(main_frame, 
                            text=" 显示控制 ", 
                            padx=10, 
                            pady=5,
                            font=("Arial", 9, "bold"))
display_frame.pack(fill=tk.X, pady=5)

roi_control_frame = tk.Frame(display_frame)
roi_control_frame.pack(fill=tk.X, pady=5)

reset_roi_button = tk.Button(roi_control_frame, 
                           text="重置显示区域", 
                           command=reset_roi,
                           width=15)
reset_roi_button.pack(side=tk.LEFT, padx=5, expand=True)

resize_button = tk.Button(roi_control_frame, 
                        text="调整窗口大小", 
                        command=resize_result_window,
                        width=15)
resize_button.pack(side=tk.LEFT, padx=5, expand=True)

# 分隔线
ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

# 参数设置区域
settings_frame = tk.LabelFrame(main_frame, 
                             text=" 检测参数设置 ", 
                             padx=10, 
                             pady=5,
                             font=("Arial", 9, "bold"))
settings_frame.pack(fill=tk.X, pady=5)

# 帧率调整
frame_rate_frame = tk.Frame(settings_frame)
frame_rate_frame.pack(fill=tk.X, pady=2)

tk.Label(frame_rate_frame, text="帧率(FPS):").pack(side=tk.LEFT, padx=(0, 5))

frame_rate_scale = tk.Scale(frame_rate_frame, 
                          from_=1, 
                          to=120, 
                          orient=tk.HORIZONTAL,
                          length=200,
                          command=update_frame_rate_from_scale)
frame_rate_scale.set(30)
frame_rate_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

frame_rate_entry = tk.Entry(frame_rate_frame, 
                          width=5,
                          justify=tk.CENTER)
frame_rate_entry.insert(0, "30")
frame_rate_entry.pack(side=tk.LEFT, padx=5)
frame_rate_entry.bind("<Return>", lambda event: update_frame_rate_from_entry())

# 变化检测阈值调整
threshold_frame = tk.Frame(settings_frame)
threshold_frame.pack(fill=tk.X, pady=2)

tk.Label(threshold_frame, text="变化阈值:").pack(side=tk.LEFT, padx=(0, 5))

threshold_scale = tk.Scale(threshold_frame, 
                         from_=1, 
                         to=100, 
                         orient=tk.HORIZONTAL,
                         length=200,
                         command=update_threshold_from_scale)
threshold_scale.set(threshold)
threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

threshold_entry = tk.Entry(threshold_frame, 
                         width=5,
                         justify=tk.CENTER)
threshold_entry.insert(0, str(threshold))
threshold_entry.pack(side=tk.LEFT, padx=5)
threshold_entry.bind("<Return>", lambda event: update_threshold_from_entry())

# 最小变化区域调整
min_area_frame = tk.Frame(settings_frame)
min_area_frame.pack(fill=tk.X, pady=2)

tk.Label(min_area_frame, text="最小区域:").pack(side=tk.LEFT, padx=(0, 5))

min_area_scale = tk.Scale(min_area_frame, 
                        from_=1, 
                        to=1000, 
                        orient=tk.HORIZONTAL,
                        length=200,
                        command=update_min_area_from_scale)
min_area_scale.set(min_contour_area)
min_area_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

min_area_entry = tk.Entry(min_area_frame, 
                        width=5,
                        justify=tk.CENTER)
min_area_entry.insert(0, str(min_contour_area))
min_area_entry.pack(side=tk.LEFT, padx=5)
min_area_entry.bind("<Return>", lambda event: update_min_area_from_entry())

# 复选框区域
checkbutton_frame = tk.Frame(settings_frame)
checkbutton_frame.pack(fill=tk.X, pady=(5, 0))

always_on_top_result_var = tk.BooleanVar(value=False)
always_on_top_result_check = tk.Checkbutton(
    checkbutton_frame, 
    text="置顶结果面板",
    variable=always_on_top_result_var,
    command=lambda: (
        cv2.setWindowProperty(
            "Change Detection Result Panel", 
            cv2.WND_PROP_TOPMOST, 
            int(always_on_top_result_var.get())
        ) if result_window_created else None
    )
)
always_on_top_result_check.pack(side=tk.LEFT, padx=5)

control_panel_resizable_var = tk.BooleanVar(value=False)
control_panel_check = tk.Checkbutton(
    checkbutton_frame, 
    text="可调整面板大小",
    variable=control_panel_resizable_var,
    command=toggle_control_panel_resizable
)
control_panel_check.pack(side=tk.LEFT, padx=5)

# 状态栏
status_frame = tk.Frame(main_frame, height=20, bg="#f0f0f0")
status_frame.pack(fill=tk.X, pady=(10, 0))

status_label = tk.Label(status_frame, 
                      text="就绪", 
                      bg="#f0f0f0",
                      fg="#333333",
                      anchor=tk.W)
status_label.pack(fill=tk.X, padx=5)

# 设置窗口关闭协议
root.protocol("WM_DELETE_WINDOW", on_closing)

# 启动线程
start_threads()

# 运行主循环
root.mainloop()
