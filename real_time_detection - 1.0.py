import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import argparse

# 全局变量
transparency = 0.5  # 窗口透明度
threshold = 30      # 变化检测阈值
min_contour_area = 500  # 最小轮廓面积

# 解析命令行参数
parser = argparse.ArgumentParser(description="实时变化检测")
parser.add_argument("--camera", action="store_true", help="使用摄像头作为输入源")
args = parser.parse_args()

# 创建主窗口
root = tk.Tk()
root.title("实时变化检测")
root.geometry("800x600")  # 设置窗口大小

# 设置窗口透明度
root.attributes("-alpha", transparency)

# 创建画布用于显示视频帧
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()

# 打开摄像头或透明窗口
if args.camera:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
else:
    # 透明窗口模式下，直接使用屏幕截图
    cap = None

# 读取第一帧
if args.camera:
    ret, prev_frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        exit()
else:
    # 透明窗口模式下，截取屏幕作为第一帧
    from mss import mss
    sct = mss()
    monitor = sct.monitors[1]  # 获取主显示器
    prev_frame = np.array(sct.grab(monitor))
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGRA2BGR)

# 转换为灰度图
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 定义更新帧的函数
def update_frame():
    global prev_gray

    # 读取下一帧
    if args.camera:
        ret, next_frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            return
    else:
        # 透明窗口模式下，截取屏幕作为下一帧
        next_frame = np.array(sct.grab(monitor))
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGRA2BGR)

    # 转换为灰度图
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # 计算帧间差异
    frame_diff = cv2.absdiff(prev_gray, next_gray)

    # 二值化差异图像
    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始帧上绘制轮廓
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:  # 过滤掉小的变化
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(next_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 将 OpenCV 图像转换为 PIL 图像
    next_frame_rgb = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(next_frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    # 更新画布
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk  # 防止图像被垃圾回收

    # 更新前一帧
    prev_gray = next_gray

    # 递归调用，持续更新帧
    root.after(30, update_frame)

# 启动更新帧的函数
update_frame()

# 运行主循环
root.mainloop()

# 释放资源
if args.camera:
    cap.release()
cv2.destroyAllWindows()
