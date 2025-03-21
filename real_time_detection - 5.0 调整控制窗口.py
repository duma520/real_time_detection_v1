import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import argparse
from mss import mss

# 全局变量
transparency = 0.5  # 监控窗口透明度
threshold = 30      # 变化检测阈值
min_contour_area = 500  # 最小变化区域
always_on_top = False  # 是否置顶

# 解析命令行参数
parser = argparse.ArgumentParser(description="实时变化检测")
parser.add_argument("--camera", action="store_true", help="使用摄像头作为输入源")
args = parser.parse_args()

# 创建主窗口
root = tk.Tk()
root.title("实时变化检测控制面板")
root.geometry("400x300")  # 设置主窗口大小

# 打开摄像头或透明窗口
if args.camera:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
else:
    # 透明窗口模式下，直接使用屏幕截图
    cap = None
    sct = mss()

# 创建监控窗口
monitor_window = tk.Toplevel(root)
monitor_window.title("监控窗口")
monitor_window.geometry("800x600")  # 设置监控窗口大小
monitor_window.attributes("-alpha", transparency)  # 设置初始透明度

# 创建画布用于显示视频帧
canvas = tk.Canvas(monitor_window)
canvas.pack(fill=tk.BOTH, expand=True)  # 画布填充整个监控窗口

# 定义更新帧的函数
def update_frame():
    # 获取监控窗口的位置和大小
    x = monitor_window.winfo_x()
    y = monitor_window.winfo_y()
    width = monitor_window.winfo_width()
    height = monitor_window.winfo_height()

    # 定义监控区域
    monitor_area = {"top": y, "left": x, "width": width, "height": height}

    # 读取当前帧（截取监控窗口所在的下层内容）
    if args.camera:
        ret, current_frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            return
    else:
        # 透明窗口模式下，截取监控窗口所在的下层内容
        current_frame = np.array(sct.grab(monitor_area))
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2BGR)

    # 读取下一帧
    if args.camera:
        ret, next_frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            return
    else:
        # 透明窗口模式下，截取监控窗口所在的下层内容
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

    # 将 OpenCV 图像转换为 PIL 图像
    current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(current_frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    # 更新画布
    canvas.delete("all")  # 清空画布
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk  # 防止图像被垃圾回收

    # 递归调用，持续更新帧
    monitor_window.after(30, update_frame)

# 创建控制面板
control_frame = tk.Frame(root)
control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

# 透明度调整
transparency_frame = tk.Frame(control_frame)
transparency_frame.pack(fill=tk.X, pady=5)

transparency_label = tk.Label(transparency_frame, text="监控窗口透明度:")
transparency_label.pack(side=tk.LEFT)

transparency_scale = tk.Scale(transparency_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                              length=300, command=lambda val: monitor_window.attributes("-alpha", float(val)))
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
            monitor_window.attributes("-alpha", value)
    except ValueError:
        pass

transparency_entry.bind("<Return>", lambda event: update_transparency_from_entry())

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

min_area_scale = tk.Scale(min_area_frame, from_=1, to=5000, orient=tk.HORIZONTAL, length=300,
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
always_on_top_check = tk.Checkbutton(control_frame, text="总是置顶", variable=always_on_top_var,
                                    command=lambda: monitor_window.attributes("-topmost", always_on_top_var.get()))
always_on_top_check.pack(pady=5)

# 启动更新帧的函数
update_frame()

# 运行主循环
root.mainloop()

# 释放资源
if args.camera:
    cap.release()
cv2.destroyAllWindows()
