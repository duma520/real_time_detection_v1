### 🤖 Assistant

# 实时变化检测软件说明书

## 1. 简介
本软件是一个实时变化检测工具，能够通过摄像头或屏幕截图检测画面中的变化区域，并在监控窗口中实时显示。软件提供了丰富的控制选项，用户可以根据需求调整透明度、帧率、变化检测阈值、最小变化区域等参数。

## 2. 适用人群
- **普通用户**：用于监控家庭环境、办公室等场景的变化。
- **安全人员**：用于监控特定区域，检测异常活动。
- **开发者**：用于调试和测试图像处理算法。
- **教育工作者**：用于教学演示，展示图像处理的基本原理。

## 3. 功能概述
- **实时变化检测**：通过摄像头或屏幕截图实时检测画面中的变化区域。
- **透明度调整**：可以调整监控窗口的透明度，方便用户同时进行其他操作。
- **帧率调整**：可以调整帧的读取频率，适应不同的硬件性能。
- **变化检测阈值调整**：可以调整变化检测的敏感度，适应不同的场景。
- **最小变化区域调整**：可以过滤掉小的变化区域，只显示较大的变化。
- **窗口置顶**：可以将监控窗口或变化结果面板置顶，方便用户随时查看。

## 4. 安装与运行
### 4.1 环境要求
- Python 3.x
- OpenCV (`cv2`)
- Tkinter (`tkinter`)
- Pillow (`PIL`)
- NumPy (`numpy`)
- MSS (`mss`)

### 4.2 安装依赖
在命令行中运行以下命令安装所需的Python库：
```bash
pip install opencv-python tkinter pillow numpy mss
```

### 4.3 运行程序
在命令行中运行以下命令启动程序：
```bash
python real_time_detection.py
```
如果需要使用摄像头作为输入源，可以添加 `--camera` 参数：
```bash
python real_time_detection.py --camera
```

## 5. 使用说明
### 5.1 主界面
程序启动后，会显示两个窗口：
- **控制面板窗口**：用于调整各项参数。
- **监控面板窗口**：用于显示实时变化检测的结果。

### 5.2 参数调整
#### 5.2.1 监控窗口透明度
- **功能**：调整监控窗口的透明度。
- **操作**：在控制面板窗口中，拖动“监控窗口透明度”滑块或直接在输入框中输入数值（0.1到1.0之间），然后按回车键。

#### 5.2.2 帧的读取频率
- **功能**：调整帧的读取频率，单位为帧/秒。
- **操作**：在控制面板窗口中，拖动“帧的读取频率”滑块或直接在输入框中输入数值（1到120之间），然后按回车键。

#### 5.2.3 变化检测阈值
- **功能**：调整变化检测的敏感度，数值越小越敏感。
- **操作**：在控制面板窗口中，拖动“变化检测阈值”滑块或直接在输入框中输入数值（1到100之间），然后按回车键。

#### 5.2.4 最小变化区域
- **功能**：过滤掉小的变化区域，只显示较大的变化。
- **操作**：在控制面板窗口中，拖动“最小变化区域”滑块或直接在输入框中输入数值（1到5000之间），然后按回车键。

#### 5.2.5 窗口置顶
- **功能**：将监控窗口或变化结果面板置顶。
- **操作**：在控制面板窗口中，勾选“总是置顶监控面板”或“总是置顶变化结果面板”复选框。

### 5.3 实时变化检测
- **功能**：实时检测画面中的变化区域，并在监控窗口中显示。
- **操作**：程序启动后，监控窗口会自动开始检测变化区域。用户可以通过调整参数来优化检测效果。

## 6. 常见问题
### 6.1 无法打开摄像头
- **原因**：摄像头可能被其他程序占用或未正确连接。
- **解决方法**：检查摄像头连接，确保没有其他程序占用摄像头。

### 6.2 监控窗口不显示
- **原因**：可能是透明度设置过低或窗口被其他窗口遮挡。
- **解决方法**：调整透明度设置，或勾选“总是置顶监控面板”复选框。

### 6.3 变化检测不准确
- **原因**：可能是阈值或最小变化区域设置不当。
- **解决方法**：调整“变化检测阈值”和“最小变化区域”参数，优化检测效果。

## 7. 技术支持
如有任何问题或建议，请联系开发者：
- 邮箱：无
- 电话：无

## 8. 版本信息
- 版本号：15.0.0
- 发布日期：2025-03-23

---

**注意**：本软件仅供学习和研究使用，请勿用于非法用途。
