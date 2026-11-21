import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from mss import mss
import win32gui
import win32process
import psutil

# 全局变量
version = "v61.10.1"  # 版本
author = "杜玛"
copyrigh = "Copyright © 杜玛. All rights reserved."
default_threshold = 30  # 默认变化检测阈值
default_min_contour_area = 500  # 默认最小变化区域
sct = mss()  # 屏幕截图对象
frame_rate = 30  # 帧率
lock_aspect_ratio = True  # 是否锁定宽高比

# 页面相关变量
pages = []  # 存储所有页面配置
active_pages = set()  # 存储激活的页面ID
page_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), 
               (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]  # 不同页面的颜色
current_page_id = 0  # 当前操作的页面ID

# 创建主窗口
root = tk.Tk()
root.title(f"智能变化检测系统 ({version} | {author} | {copyrigh})")
root.geometry("800x900")  # 设置更大的窗口大小以适应多页面
root.minsize(800, 600)  # 设置最小窗口大小

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
control_panel_frame.pack(fill=tk.BOTH, expand=True)

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
        control_panel_frame.pack(fill=tk.BOTH, expand=True, before=toggle_button)
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
        global current_page_id
        
        # 更新当前页面的监控进程
        pages[current_page_id]['monitoring_process'] = {
            'hwnd': next(p['hwnd'] for p in get_process_windows() if p['pid'] == int(item_data['values'][0])),
            'pid': int(item_data['values'][0]),
            'name': item_data['values'][1],
            'title': item_data['values'][2]
        }
        pages[current_page_id]['monitoring_camera'] = None  # 清除摄像头监控
        
        # 更新按钮状态
        process_buttons[current_page_id].config(text=f"停止监控 {pages[current_page_id]['monitoring_process']['name']}")
        camera_buttons[current_page_id].config(text="选择镜头监控")
        
        selection_window.destroy()
        status_bar.config(text=f"页面 {current_page_id+1}: 已开始监控进程: {pages[current_page_id]['monitoring_process']['name']}")
    
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
        global current_page_id
        
        # 更新当前页面的监控摄像头
        pages[current_page_id]['monitoring_camera'] = int(item_data['values'][0])
        pages[current_page_id]['monitoring_process'] = None  # 清除进程监控
        
        # 更新按钮状态
        camera_buttons[current_page_id].config(text=f"停止监控 摄像头{pages[current_page_id]['monitoring_camera']}")
        process_buttons[current_page_id].config(text="选择监控进程")
        
        selection_window.destroy()
        status_bar.config(text=f"页面 {current_page_id+1}: 已开始监控摄像头: {pages[current_page_id]['monitoring_camera']}")
    
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
def toggle_process_monitoring(page_id):
    global current_page_id
    current_page_id = page_id
    
    if pages[page_id]['monitoring_process']:
        # 停止监控
        pages[page_id]['monitoring_process'] = None
        process_buttons[page_id].config(text="选择监控进程")
        status_bar.config(text=f"页面 {page_id+1}: 已停止监控")
    else:
        # 开始监控 - 打开进程选择窗口
        pages[page_id]['monitoring_camera'] = None  # 清除摄像头监控
        camera_buttons[page_id].config(text="选择镜头监控")
        create_process_selection_window()

# 开始/停止监控摄像头
def toggle_camera_monitoring(page_id):
    global current_page_id
    current_page_id = page_id
    
    if pages[page_id]['monitoring_camera'] is not None:
        # 停止监控
        pages[page_id]['monitoring_camera'] = None
        camera_buttons[page_id].config(text="选择镜头监控")
        status_bar.config(text=f"页面 {page_id+1}: 已停止监控")
    else:
        # 开始监控 - 打开摄像头选择窗口
        pages[page_id]['monitoring_process'] = None  # 清除进程监控
        process_buttons[page_id].config(text="选择监控进程")
        create_camera_selection_window()

def select_monitor_area(page_id):
    global current_page_id, monitor_area_roi
    current_page_id = page_id
    
    if not pages[page_id]['monitoring_process']:
        messagebox.showwarning("警告", "请先选择要监控的进程")
        return
    
    # 获取窗口句柄
    hwnd = pages[page_id]['monitoring_process']['hwnd']
    
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
        x1, y1 = min(start_x, end_x), min(start_y, end_y)
        x2, y2 = max(start_x, end_x), max(start_y, end_y)
        
        # 确保选择的区域有效
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # 最小10像素的宽度和高度
            pages[page_id]['monitor_area_roi'] = (x1, y1, x2-x1, y2-y1)
            status_bar.config(text=f"页面 {page_id+1}: 已设置监控区域: {pages[page_id]['monitor_area_roi']}")
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

# 获取指定页面的帧
def get_frame_for_page(page):
    if page['monitoring_camera'] is not None:
        # 摄像头模式
        cap = cv2.VideoCapture(page['monitoring_camera'])
        if not cap.isOpened():
            status_bar.config(text=f"页面 {page['id']+1}: 无法打开摄像头 {page['monitoring_camera']}")
            return None
        
        ret, frame = cap.read()
        cap.release()
        if not ret:
            status_bar.config(text=f"页面 {page['id']+1}: 无法读取视频帧")
            return None
        
        return frame
    
    elif page['monitoring_process']:
        # 进程监控模式
        try:
            # 获取窗口位置
            rect = get_window_rect(page['monitoring_process']['hwnd'])
            monitor_area = {
                "left": rect["left"],
                "top": rect["top"],
                "width": rect["width"],
                "height": rect["height"]
            }
            
            # 如果设置了监控区域ROI，则调整监控区域
            if page['monitor_area_roi']:
                x, y, w, h = page['monitor_area_roi']
                monitor_area["left"] += x
                monitor_area["top"] += y
                monitor_area["width"] = w
                monitor_area["height"] = h

            # 读取帧
            frame = np.array(sct.grab(monitor_area))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            status_bar.config(text=f"页面 {page['id']+1}: 监控窗口出错: {e}")
            return None
    
    return None

# 处理帧差异
def process_frame_differences(page, current_frame, next_frame):
    if current_frame is None or next_frame is None:
        return None
    
    # 转换为灰度图
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # 计算帧间差异
    frame_diff = cv2.absdiff(current_gray, next_gray)

    # 二值化差异图像
    _, thresh = cv2.threshold(frame_diff, page['threshold'], 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建结果帧
    result_frame = current_frame.copy()

    # 在原始帧上绘制轮廓
    for contour in contours:
        if cv2.contourArea(contour) > page['min_contour_area']:  # 过滤掉小的变化
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return result_frame

# 定义更新帧的函数
def update_frame():
    if not active_pages:
        # 没有激活的页面，显示黑色图像
        current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(current_frame, "没有激活的监控页面", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        update_video_display(current_frame)
        root.after(30, update_frame)
        return
    
    combined_frame = None
    
    for page_id in active_pages:
        page = pages[page_id]
        
        # 获取当前页面的两帧用于比较
        current_frame = get_frame_for_page(page)
        next_frame = get_frame_for_page(page)
        
        if current_frame is None or next_frame is None:
            continue
        
        # 处理帧差异
        processed_frame = process_frame_differences(page, current_frame, next_frame)
        
        if processed_frame is None:
            continue
        
        # 合并结果
        if combined_frame is None:
            combined_frame = processed_frame
        else:
            # 叠加不同颜色的结果
            color = page_colors[page_id % len(page_colors)]
            # 创建一个掩码，找到所有绿色矩形(0,255,0)
            mask = cv2.inRange(processed_frame, (0, 254, 0), (0, 256, 0))
            # 将检测到的变化区域设置为当前页面的颜色
            combined_frame[mask > 0] = color
    
    # 更新显示
    if combined_frame is not None:
        update_video_display(combined_frame)
    else:
        # 显示黑色图像
        current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(current_frame, "无有效监控数据", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        update_video_display(current_frame)
    
    # 递归调用，持续更新帧
    root.after(30, update_frame)

# 创建控制面板内容
def create_control_panel():
    global process_buttons, camera_buttons
    
    # 创建主笔记本(多页面容器)
    notebook = ttk.Notebook(control_panel_frame)
    notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # 存储按钮引用
    process_buttons = []
    camera_buttons = []
    
    # 创建4个默认页面
    for i in range(4):
        frame = tk.Frame(notebook)
        notebook.add(frame, text=f"页面 {i+1}")
        
        # 初始化页面配置
        page_config = {
            'id': i,
            'threshold': default_threshold,
            'min_contour_area': default_min_contour_area,
            'monitoring_process': None,
            'monitoring_camera': None,
            'monitor_area_roi': None
        }
        pages.append(page_config)
        
        # 创建页面内容
        # 监控源选择区域
        source_frame = tk.LabelFrame(frame, text="监控源选择", padx=10, pady=10)
        source_frame.pack(fill=tk.X, pady=5)
        
        # 监控进程按钮
        process_button = tk.Button(
            source_frame, 
            text="选择监控进程", 
            command=lambda idx=i: toggle_process_monitoring(idx)
        )
        process_button.pack(side=tk.LEFT, padx=5, pady=5)
        process_buttons.append(process_button)
        
        # 监控摄像头按钮
        camera_button = tk.Button(
            source_frame, 
            text="选择镜头监控", 
            command=lambda idx=i: toggle_camera_monitoring(idx)
        )
        camera_button.pack(side=tk.LEFT, padx=5, pady=5)
        camera_buttons.append(camera_button)
        
        # 页面激活复选框
        activate_var = tk.BooleanVar(value=(i == 0))  # 默认激活第一个页面
        activate_check = tk.Checkbutton(
            source_frame,
            text="激活此页面",
            variable=activate_var,
            command=lambda idx=i: toggle_page_activation(idx)
        )
        activate_check.pack(side=tk.RIGHT, padx=5)
        
        if i == 0:
            active_pages.add(i)  # 默认激活第一个页面
            activate_var.set(True)
        
        # 监控区域控制
        roi_frame = tk.LabelFrame(frame, text="监控区域控制", padx=10, pady=10)
        roi_frame.pack(fill=tk.X, pady=5)
        
        select_monitor_button = tk.Button(
            roi_frame, 
            text="选择目标监控区域", 
            command=lambda idx=i: select_monitor_area(idx)
        )
        select_monitor_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 参数设置区域
        settings_frame = tk.LabelFrame(frame, text="检测参数设置", padx=10, pady=10)
        settings_frame.pack(fill=tk.X, pady=5)
        
        # 变化检测阈值调整
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
        threshold_scale.set(default_threshold)
        threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        threshold_entry = tk.Entry(threshold_frame, width=5)
        threshold_entry.insert(0, str(default_threshold))
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
        
        def update_threshold_from_scale(val, idx):
            threshold_entry.delete(0, tk.END)
            threshold_entry.insert(0, str(val))
            update_page_threshold(idx, int(val))
            status_bar.config(text=f"页面 {idx+1}: 变化检测阈值已设置为: {val}")
        
        threshold_scale.config(command=lambda val: update_threshold_from_scale(val, i))
        threshold_entry.bind("<Return>", lambda event: update_threshold_from_entry(i))
        
        # 最小变化区域调整
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
        min_area_scale.set(default_min_contour_area)
        min_area_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        min_area_entry = tk.Entry(min_area_frame, width=5)
        min_area_entry.insert(0, str(default_min_contour_area))
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
        
        def update_min_area_from_scale(val, idx):
            min_area_entry.delete(0, tk.END)
            min_area_entry.insert(0, str(val))
            update_page_min_area(idx, int(val))
            status_bar.config(text=f"页面 {idx+1}: 最小变化区域已设置为: {val} 像素")
        
        min_area_scale.config(command=lambda val: update_min_area_from_scale(val, i))
        min_area_entry.bind("<Return>", lambda event: update_min_area_from_entry(i))
    
    # 创建复选框框架
    global_frame = tk.Frame(control_panel_frame)
    global_frame.pack(fill=tk.X, pady=5)
    
    # 锁定宽高比复选框
    lock_aspect_var = tk.BooleanVar(value=lock_aspect_ratio)
    lock_aspect_check = tk.Checkbutton(
        global_frame, 
        text="保持图像比例", 
        variable=lock_aspect_var,
        command=lambda: globals().update(lock_aspect_ratio=not lock_aspect_ratio)
    )
    lock_aspect_check.pack(side=tk.LEFT, padx=5)
    
    # 总是置顶复选框
    always_on_top_var = tk.BooleanVar()
    always_on_top_check = tk.Checkbutton(
        global_frame,
        text="总是置顶",
        variable=always_on_top_var,
        command=lambda: root.attributes('-topmost', not root.attributes('-topmost'))
    )
    always_on_top_check.pack(side=tk.LEFT, padx=5)

# 辅助函数
def toggle_page_activation(page_id):
    if page_id in active_pages:
        active_pages.remove(page_id)
        status_bar.config(text=f"页面 {page_id+1}: 已停用")
    else:
        active_pages.add(page_id)
        status_bar.config(text=f"页面 {page_id+1}: 已激活")

def update_page_threshold(page_id, value):
    pages[page_id]['threshold'] = value

def update_page_min_area(page_id, value):
    pages[page_id]['min_contour_area'] = value

# 创建控制面板
create_control_panel()

# 启动更新帧的函数
update_frame()

# 运行主循环
root.mainloop()

# 释放资源
for page in pages:
    if page['monitoring_camera'] is not None:
        cv2.VideoCapture(page['monitoring_camera']).release()