import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from mss import mss
import win32gui
import win32process
import psutil
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# 全局配置
class Config:
    VERSION = "v61.10.1"
    AUTHOR = "杜玛"
    COPYRIGHT = "Copyright © 杜玛. All rights reserved."
    DEFAULT_THRESHOLD = 30
    DEFAULT_MIN_CONTOUR_AREA = 500
    FRAME_RATE = 30
    MAX_WORKERS = 4  # 线程池最大工作线程数
    PAGE_COLORS = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
    ]

# 全局变量
sct = mss()
frame_queue = queue.Queue()
executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
processing_lock = threading.Lock()
current_image = None
control_panel_visible = True
active_pages = set()
pages = []
current_page_id = 0
lock_aspect_ratio = True

# 主窗口初始化
root = tk.Tk()
root.title(f"智能变化检测系统 ({Config.VERSION} | {Config.AUTHOR} | {Config.COPYRIGHT})")
root.geometry("800x900")
root.minsize(800, 600)

# UI组件
main_frame = tk.Frame(root)
video_frame = tk.Frame(main_frame, bg='black')
canvas = tk.Canvas(video_frame, bg='black')
control_panel_frame = tk.Frame(main_frame, bd=2, relief=tk.RAISED)
toggle_button = tk.Button(main_frame, text="▲ 显示控制面板 ▲", command=lambda: toggle_control_panel(), relief=tk.RAISED, bd=1)
status_bar = tk.Label(root, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)

# 布局
main_frame.pack(fill=tk.BOTH, expand=True)
video_frame.pack(fill=tk.BOTH, expand=True)
canvas.pack(fill=tk.BOTH, expand=True)
control_panel_frame.pack(fill=tk.BOTH, expand=True)
toggle_button.pack(fill=tk.X, side=tk.BOTTOM)
status_bar.pack(fill=tk.X, side=tk.BOTTOM)

class Page:
    def __init__(self, page_id):
        self.id = page_id
        self.threshold = Config.DEFAULT_THRESHOLD
        self.min_contour_area = Config.DEFAULT_MIN_CONTOUR_AREA
        self.monitoring_process = None
        self.monitoring_camera = None
        self.monitor_area_roi = None
        self.process_button = None
        self.camera_button = None
        self.active = False

def toggle_control_panel():
    global control_panel_visible
    if control_panel_visible:
        control_panel_frame.pack_forget()
        toggle_button.config(text="▼ 显示控制面板 ▼")
        control_panel_visible = False
    else:
        control_panel_frame.pack(fill=tk.BOTH, expand=True, before=toggle_button)
        toggle_button.config(text="▲ 隐藏控制面板 ▲")
        control_panel_visible = True

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

def get_available_cameras(max_test=5):
    cameras = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras

def get_window_rect(hwnd):
    rect = win32gui.GetWindowRect(hwnd)
    return {
        "left": rect[0],
        "top": rect[1],
        "width": rect[2] - rect[0],
        "height": rect[3] - rect[1]
    }

def create_process_selection_window():
    selection_window = tk.Toplevel(root)
    selection_window.title(f"选择需要监控的程序 ({Config.VERSION} | {Config.AUTHOR} | {Config.COPYRIGHT})")
    selection_window.geometry("750x500")
    
    # UI组件
    search_frame = tk.Frame(selection_window)
    search_label = tk.Label(search_frame, text="搜索:")
    search_var = tk.StringVar()
    search_entry = tk.Entry(search_frame, textvariable=search_var)
    refresh_button = tk.Button(search_frame, text="刷新", command=lambda: update_process_list(tree))
    tree = ttk.Treeview(selection_window, columns=('pid', 'name', 'title'), show='headings')
    scrollbar = ttk.Scrollbar(selection_window, orient="vertical", command=tree.yview)
    button_frame = tk.Frame(selection_window)
    select_button = tk.Button(button_frame, text="确定", command=lambda: on_select())
    cancel_button = tk.Button(button_frame, text="取消", command=lambda: selection_window.destroy())
    
    # 布局
    search_frame.pack(fill=tk.X, padx=10, pady=5)
    search_label.pack(side=tk.LEFT)
    search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    refresh_button.pack(side=tk.RIGHT, padx=5)
    tree.heading('pid', text='PID', command=lambda: sort_treeview(tree, 'pid', False))
    tree.heading('name', text='进程名', command=lambda: sort_treeview(tree, 'name', False))
    tree.heading('title', text='窗口标题', command=lambda: sort_treeview(tree, 'title', False))
    tree.column('pid', width=80, anchor='center')
    tree.column('name', width=150, anchor='center')
    tree.column('title', width=350, anchor='w')
    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)
    button_frame.pack(pady=10)
    select_button.pack(side=tk.LEFT, padx=10)
    cancel_button.pack(side=tk.LEFT, padx=10)
    
    def update_process_list(treeview):
        processes = get_process_windows()
        treeview.delete(*treeview.get_children())
        for process in processes:
            treeview.insert('', 'end', values=(process['pid'], process['name'], process['title']))
    
    def on_search(*args):
        query = search_var.get().lower()
        processes = get_process_windows()
        tree.delete(*tree.get_children())
        for process in processes:
            if (query in str(process['pid']).lower() or 
                query in process['name'].lower() or 
                query in process['title'].lower()):
                tree.insert('', 'end', values=(process['pid'], process['name'], process['title']))
    
    def sort_treeview(treeview, col, reverse):
        data = [(treeview.set(child, col), child) for child in treeview.get_children('')]
        try:
            data.sort(key=lambda x: int(x[0]), reverse=reverse)
        except ValueError:
            data.sort(reverse=reverse)
        for index, (val, child) in enumerate(data):
            treeview.move(child, '', index)
        treeview.heading(col, command=lambda: sort_treeview(treeview, col, not reverse))
    
    def on_select():
        selected_item = tree.focus()
        if not selected_item:
            messagebox.showwarning("警告", "请先选择一个进程")
            return
        
        item_data = tree.item(selected_item)
        global current_page_id
        
        pages[current_page_id].monitoring_process = {
            'hwnd': next(p['hwnd'] for p in get_process_windows() if p['pid'] == int(item_data['values'][0])),
            'pid': int(item_data['values'][0]),
            'name': item_data['values'][1],
            'title': item_data['values'][2]
        }
        pages[current_page_id].monitoring_camera = None
        
        pages[current_page_id].process_button.config(text=f"停止监控 {pages[current_page_id].monitoring_process['name']}")
        pages[current_page_id].camera_button.config(text="选择镜头监控")
        
        selection_window.destroy()
        status_bar.config(text=f"页面 {current_page_id+1}: 已开始监控进程: {pages[current_page_id].monitoring_process['name']}")
    
    search_var.trace("w", on_search)
    tree.bind("<Double-1>", lambda event: on_select())
    update_process_list(tree)

def create_camera_selection_window():
    selection_window = tk.Toplevel(root)
    selection_window.title(f"选择监控镜头 ({Config.VERSION} | {Config.AUTHOR} | {Config.COPYRIGHT})")
    selection_window.geometry("400x300")
    
    cameras = get_available_cameras()
    if not cameras:
        status_bar.config(text="没有检测到可用的摄像头")
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
    
    button_frame = tk.Frame(selection_window)
    button_frame.pack(pady=10)
    
    def on_select():
        selected_item = tree.focus()
        if not selected_item:
            status_bar.config(text="请先选择一个摄像头")
            return
        
        item_data = tree.item(selected_item)
        global current_page_id
        
        pages[current_page_id].monitoring_camera = int(item_data['values'][0])
        pages[current_page_id].monitoring_process = None
        
        pages[current_page_id].camera_button.config(text=f"停止监控 摄像头{pages[current_page_id].monitoring_camera}")
        pages[current_page_id].process_button.config(text="选择监控进程")
        
        selection_window.destroy()
        status_bar.config(text=f"页面 {current_page_id+1}: 已开始监控摄像头: {pages[current_page_id].monitoring_camera}")
    
    select_button = tk.Button(button_frame, text="确定", command=on_select)
    select_button.pack(side=tk.LEFT, padx=10)
    
    cancel_button = tk.Button(button_frame, text="取消", command=lambda: selection_window.destroy())
    cancel_button.pack(side=tk.LEFT, padx=10)

def toggle_process_monitoring(page_id):
    global current_page_id
    current_page_id = page_id
    
    if pages[page_id].monitoring_process:
        pages[page_id].monitoring_process = None
        pages[page_id].process_button.config(text="选择监控进程")
        status_bar.config(text=f"页面 {page_id+1}: 已停止监控")
    else:
        pages[page_id].monitoring_camera = None
        pages[page_id].camera_button.config(text="选择镜头监控")
        status_bar.config(text=f"页面 {page_id+1}: 正在选择进程...")
        create_process_selection_window()

def toggle_camera_monitoring(page_id):
    global current_page_id
    current_page_id = page_id
    
    if pages[page_id].monitoring_camera is not None:
        pages[page_id].monitoring_camera = None
        pages[page_id].camera_button.config(text="选择镜头监控")
        status_bar.config(text=f"页面 {page_id+1}: 已停止监控")
    else:
        pages[page_id].monitoring_process = None
        pages[page_id].process_button.config(text="选择监控进程")
        create_camera_selection_window()

def select_monitor_area(page_id):
    global current_page_id
    current_page_id = page_id
    
    if not pages[page_id].monitoring_process:
        messagebox.showwarning("警告", "请先选择要监控的进程")
        return
    
    hwnd = pages[page_id].monitoring_process['hwnd']
    temp_window = tk.Toplevel(root)
    temp_window.title("请在目标窗口上选择监控区域")
    temp_window.geometry("300x100")
    
    label = tk.Label(temp_window, text="请在目标窗口上拖动鼠标选择监控区域\n释放鼠标左键自动确认")
    label.pack(pady=20)
    temp_window.attributes('-topmost', True)
    
    rect = win32gui.GetWindowRect(hwnd)
    selection_window = tk.Toplevel(temp_window)
    selection_window.overrideredirect(True)
    selection_window.geometry(f"{rect[2]-rect[0]}x{rect[3]-rect[1]}+{rect[0]}+{rect[1]}")
    selection_window.attributes('-alpha', 0.3)
    selection_window.attributes('-topmost', True)
    
    start_x, start_y = 0, 0
    end_x, end_y = 0, 0
    rect_id = None
    
    def on_mouse_down(event):
        nonlocal start_x, start_y, rect_id
        start_x, start_y = event.x, event.y
        if rect_id:
            selection_window.canvas.delete(rect_id)
        rect_id = selection_window.canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2)
    
    def on_mouse_move(event):
        nonlocal end_x, end_y, rect_id
        end_x, end_y = event.x, event.y
        if rect_id:
            selection_window.canvas.coords(rect_id, start_x, start_y, end_x, end_y)
    
    def on_mouse_up(event):
        nonlocal end_x, end_y
        end_x, end_y = event.x, event.y
    
        x1, y1 = min(start_x, end_x), min(start_y, end_y)
        x2, y2 = max(start_x, end_x), max(start_y, end_y)
        
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            pages[page_id].monitor_area_roi = (x1, y1, x2-x1, y2-y1)
            status_bar.config(text=f"页面 {page_id+1}: 已设置监控区域: {pages[page_id].monitor_area_roi}")
        else:
            status_bar.config(text="选择区域太小，请重新选择")
            return
        
        temp_window.destroy()
        selection_window.destroy()
    
    selection_window.canvas = tk.Canvas(selection_window, cursor="cross")
    selection_window.canvas.pack(fill=tk.BOTH, expand=True)
    selection_window.canvas.bind("<ButtonPress-1>", on_mouse_down)
    selection_window.canvas.bind("<B1-Motion>", on_mouse_move)
    selection_window.canvas.bind("<ButtonRelease-1>", on_mouse_up)
    selection_window.canvas.focus_set()

def update_video_display(frame):
    global current_image
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    
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
    
    img = img.resize((new_width, new_height), Image.LANCZOS)
    current_image = ImageTk.PhotoImage(image=img)
    
    canvas.delete("all")
    canvas.create_image(
        (canvas_width - new_width) // 2,
        (canvas_height - new_height) // 2,
        anchor=tk.NW,
        image=current_image
    )

def get_frame_for_page_threadsafe(page):
    try:
        if page.monitoring_camera is not None:
            with processing_lock:
                cap = cv2.VideoCapture(page.monitoring_camera)
                if not cap.isOpened():
                    return None, f"页面 {page.id+1}: 无法打开摄像头 {page.monitoring_camera}"
                
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return None, f"页面 {page.id+1}: 无法读取视频帧"
                
                return frame, None
        
        elif page.monitoring_process:
            with processing_lock:
                try:
                    rect = get_window_rect(page.monitoring_process['hwnd'])
                    monitor_area = {
                        "left": rect["left"],
                        "top": rect["top"],
                        "width": rect["width"],
                        "height": rect["height"]
                    }
                    
                    if page.monitor_area_roi:
                        x, y, w, h = page.monitor_area_roi
                        monitor_area["left"] += x
                        monitor_area["top"] += y
                        monitor_area["width"] = w
                        monitor_area["height"] = h

                    frame = np.array(sct.grab(monitor_area))
                    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), None
                except Exception as e:
                    return None, f"捕获窗口出错: {str(e)}"
        
        return None, None
    except Exception as e:
        return None, f"页面 {page.id+1}: 获取帧出错: {str(e)}"

def process_frame_differences_threadsafe(page, current_frame, next_frame):
    try:
        if current_frame is None or next_frame is None:
            return None, "无有效帧数据"
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(current_gray, next_gray)
        _, thresh = cv2.threshold(frame_diff, page.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_frame = current_frame.copy()

        for contour in contours:
            if cv2.contourArea(contour) > page.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return result_frame, None
    except Exception as e:
        return None, f"页面 {page.id+1}: 处理帧差异出错: {str(e)}"

def process_page_task(page_id):
    page = pages[page_id]
    current_frame, error_msg = get_frame_for_page_threadsafe(page)
    if error_msg:
        frame_queue.put((page_id, None, error_msg))
        return
    
    next_frame, error_msg = get_frame_for_page_threadsafe(page)
    if error_msg:
        frame_queue.put((page_id, None, error_msg))
        return
    
    processed_frame, error_msg = process_frame_differences_threadsafe(page, current_frame, next_frame)
    frame_queue.put((page_id, processed_frame, error_msg))

def update_frame():
    if not active_pages:
        current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(current_frame, "没有激活的监控页面", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        update_video_display(current_frame)
        root.after(30, update_frame)
        return
    
    for page_id in active_pages:
        executor.submit(process_page_task, page_id)
    
    combined_frame = None
    results_received = 0
    
    while results_received < len(active_pages) and not frame_queue.empty():
        page_id, processed_frame, error_msg = frame_queue.get()
        results_received += 1
        
        if error_msg:
            status_bar.config(text=error_msg)
            continue
        
        if processed_frame is not None:
            if combined_frame is None:
                combined_frame = processed_frame
            else:
                color = Config.PAGE_COLORS[page_id % len(Config.PAGE_COLORS)]
                mask = cv2.inRange(processed_frame, (0, 254, 0), (0, 256, 0))
                combined_frame[mask > 0] = color
    
    if combined_frame is not None:
        update_video_display(combined_frame)
    else:
        current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(current_frame, "无有效监控数据", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        update_video_display(current_frame)
    
    root.after(30, update_frame)

def create_control_panel():
    notebook = ttk.Notebook(control_panel_frame)
    notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    for i in range(4):
        page = Page(i)
        pages.append(page)
        
        frame = tk.Frame(notebook)
        notebook.add(frame, text=f"页面 {i+1}")
        
        source_frame = tk.LabelFrame(frame, text="监控源选择", padx=10, pady=10)
        source_frame.pack(fill=tk.X, pady=5)
        
        page.process_button = tk.Button(
            source_frame, 
            text="选择监控进程", 
            command=lambda idx=i: toggle_process_monitoring(idx)
        )
        page.process_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        page.camera_button = tk.Button(
            source_frame, 
            text="选择镜头监控", 
            command=lambda idx=i: toggle_camera_monitoring(idx)
        )
        page.camera_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        activate_var = tk.BooleanVar(value=(i == 0))
        activate_check = tk.Checkbutton(
            source_frame,
            text="激活此页面",
            variable=activate_var,
            command=lambda idx=i: toggle_page_activation(idx)
        )
        activate_check.pack(side=tk.RIGHT, padx=5)
        
        if i == 0:
            active_pages.add(i)
            activate_var.set(True)
        
        roi_frame = tk.LabelFrame(frame, text="监控区域控制", padx=10, pady=10)
        roi_frame.pack(fill=tk.X, pady=5)
        
        select_monitor_button = tk.Button(
            roi_frame, 
            text="选择目标监控区域", 
            command=lambda idx=i: select_monitor_area(idx)
        )
        select_monitor_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        settings_frame = tk.LabelFrame(frame, text="检测参数设置", padx=10, pady=10)
        settings_frame.pack(fill=tk.X, pady=5)
        
        threshold_frame = tk.Frame(settings_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        threshold_label = tk.Label(threshold_frame, text="变化检测阈值:")
        threshold_label.pack(side=tk.LEFT)
        
        threshold_scale = tk.Scale(
            threshold_frame, 
            from_=1, 
            to=100, 
            orient=tk.HORIZONTAL, 
            length=200,
            command=lambda val, idx=i: update_page_threshold(idx, int(val))
        )
        threshold_scale.set(page.threshold)
        threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        threshold_entry = tk.Entry(threshold_frame, width=5)
        threshold_entry.insert(0, str(page.threshold))
        threshold_entry.pack(side=tk.LEFT, padx=5)
        
        def update_threshold_from_entry(idx):
            try:
                value = int(threshold_entry.get())
                if 1 <= value <= 100:
                    threshold_scale.set(value)
                    update_page_threshold(idx, value)
                    status_bar.config(text=f"页面 {idx+1}: 变化检测阈值已设置为: {value}")
            except ValueError:
                pass
        
        threshold_scale.config(command=lambda val: update_page_threshold(i, int(val)))
        threshold_entry.bind("<Return>", lambda event: update_threshold_from_entry(i))
        
        min_area_frame = tk.Frame(settings_frame)
        min_area_frame.pack(fill=tk.X, pady=5)
        
        min_area_label = tk.Label(min_area_frame, text="最小变化区域:")
        min_area_label.pack(side=tk.LEFT)
        
        min_area_scale = tk.Scale(
            min_area_frame, 
            from_=1, 
            to=1000, 
            orient=tk.HORIZONTAL, 
            length=200,
            command=lambda val, idx=i: update_page_min_area(idx, int(val))
        )
        min_area_scale.set(page.min_contour_area)
        min_area_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        min_area_entry = tk.Entry(min_area_frame, width=5)
        min_area_entry.insert(0, str(page.min_contour_area))
        min_area_entry.pack(side=tk.LEFT, padx=5)
        
        def update_min_area_from_entry(idx):
            try:
                value = int(min_area_entry.get())
                if 1 <= value <= 5000:
                    min_area_scale.set(value)
                    update_page_min_area(idx, value)
                    status_bar.config(text=f"页面 {idx+1}: 最小变化区域已设置为: {value} 像素")
            except ValueError:
                pass
        
        min_area_scale.config(command=lambda val: update_page_min_area(i, int(val)))
        min_area_entry.bind("<Return>", lambda event: update_min_area_from_entry(i))
    
    global_frame = tk.Frame(control_panel_frame)
    global_frame.pack(fill=tk.X, pady=5)
    
    lock_aspect_var = tk.BooleanVar(value=lock_aspect_ratio)
    lock_aspect_check = tk.Checkbutton(
        global_frame, 
        text="保持图像比例", 
        variable=lock_aspect_var,
        command=lambda: globals().update(lock_aspect_ratio=not lock_aspect_ratio)
    )
    lock_aspect_check.pack(side=tk.LEFT, padx=5)
    
    always_on_top_var = tk.BooleanVar()
    always_on_top_check = tk.Checkbutton(
        global_frame,
        text="总是置顶",
        variable=always_on_top_var,
        command=lambda: root.attributes('-topmost', not root.attributes('-topmost'))
    )
    always_on_top_check.pack(side=tk.LEFT, padx=5)

def toggle_page_activation(page_id):
    if page_id in active_pages:
        active_pages.remove(page_id)
        status_bar.config(text=f"页面 {page_id+1}: 已停用")
    else:
        active_pages.add(page_id)
        status_bar.config(text=f"页面 {page_id+1}: 已激活")

def update_page_threshold(page_id, value):
    pages[page_id].threshold = value

def update_page_min_area(page_id, value):
    pages[page_id].min_contour_area = value

def on_closing():
    executor.shutdown(wait=False)
    for page in pages:
        if page.monitoring_camera is not None:
            cv2.VideoCapture(page.monitoring_camera).release()
    root.destroy()

# 初始化
create_control_panel()
root.protocol("WM_DELETE_WINDOW", on_closing)
update_frame()
root.mainloop()