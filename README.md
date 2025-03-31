# 智能变化检测系统说明书 v71.4.8（终结版）新版V2 已经上线

## 1. 产品概述
智能变化检测系统是一款基于计算机视觉技术的实时监控软件，能够检测画面中的变化区域并进行标记。适用于多种应用场景，如安防监控、游戏录制、自动化测试等。

## 2. 用户指南

### 2.1 初级用户快速入门（小白用户）

#### 2.1.1 基本使用步骤
1. **启动程序**：双击程序图标打开软件
2. **选择监控源**：
   - 点击"选择进程"监控某个窗口
   - 点击"选择镜头"使用摄像头监控
3. **开始监控**：系统会自动检测画面变化并标记绿框
4. **调整灵敏度**：
   - 拖动"阈值"滑块（数值越小越敏感）
   - 拖动"最小区域"滑块（数值越大忽略越小变化）

#### 2.1.2 简单功能介绍
- **渐隐效果**：打开后，消失的物体会有黄色渐隐标记
- **锁定比例**：保持视频画面原始比例不变形
- **置顶窗口**：让监控窗口始终显示在最前面

### 2.2 中级用户进阶功能

#### 2.2.1 区域选择功能
1. 先选择一个要监控的进程窗口
2. 点击"选择区域"按钮
3. 在目标窗口上拖动鼠标框选特定区域
4. 松开鼠标自动确认（区域需大于10x10像素）

#### 2.2.2 性能优化建议
- 帧率设置：普通使用30FPS足够，高性能电脑可尝试60FPS
- 分辨率缩放：对性能要求高时可降低到0.7-0.8
- 开启"自适应"模式让系统自动调节性能

#### 2.2.3 算法选择指导
- **快速检测**：适合运动明显的场景（如行人检测）
- **精确检测**：适合精细变化检测（如工业检测）
- **运动追踪**：适合连续运动的物体跟踪

### 2.3 高级用户专业配置

#### 2.3.1 硬件加速配置
系统支持多种加速后端：
- **CUDA**：NVIDIA显卡用户首选
- **OpenCL**：AMD/Intel显卡用户选择
- **CPU**：无合适显卡时的后备方案

可通过下拉菜单手动选择，或让系统自动检测最佳方案

#### 2.3.2 多线程处理
- 优点：提高多核CPU利用率，提升处理速度
- 缺点：可能增加延迟，老旧CPU不建议开启

#### 2.3.3 动态调节机制
- 动态阈值：根据系统负载自动调整检测灵敏度
- 自适应FPS：在15-60FPS之间自动调节帧率
- 智能算法组合：根据场景复杂度自动选择算法

## 3. 技术说明

### 3.1 核心算法

#### 3.1.1 帧差分算法（基础）
```python
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
frame_diff = cv2.absdiff(gray1, gray2)
_, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
```

#### 3.1.2 背景减除算法
使用OpenCV的MOG2背景建模器适应光照变化

#### 3.1.3 光流法
基于Farneback算法计算稠密光流场，适合运动追踪

### 3.2 性能优化技术

#### 3.2.1 多尺度处理
```python
scales = [1.0, 0.75, 0.5]  # 多尺度金字塔
for scale in scales:
    resized = cv2.resize(frame, None, fx=scale, fy=scale)
    # 处理并合并结果
```

#### 3.2.2 ROI跟踪
只处理检测到变化的区域，减少计算量：

```python
if roi_tracking and last_boxes:
    # 计算ROI区域并扩大20%边界
    x_min = min(box[0] for box in last_boxes)
    y_min = min(box[1] for box in last_boxes)
    roi = (x_min-margin, y_min-margin, width+2*margin, height+2*margin)
```

#### 3.2.3 帧跳过机制
允许跳过指定数量的中间帧，降低处理负荷：

```python
if frame_skip > 0:
    skip_counter += 1
    if skip_counter <= frame_skip:
        return  # 跳过处理
    skip_counter = 0
```

### 3.3 硬件加速实现

#### 3.3.1 CUDA加速
关键代码修改：
```python
# 修改前（CPU）
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# 修改后（CUDA）
gpu_frame1.upload(frame1, self.stream)
gpu_gray1 = cv2.cuda.cvtColor(self.gpu_frame1, cv2.COLOR_BGR2GRAY, stream=self.stream)
```

#### 3.3.2 OpenCL加速
使用UMat实现：
```python
umat1 = cv2.UMat(frame1)  # 上传到OpenCL设备
gray1 = cv2.cvtColor(umat1, cv2.COLOR_BGR2GRAY)  # 在设备上执行
```

## 4. 使用场景举例

### 4.1 居家安防监控
- **配置建议**：使用"精确检测"算法组合
- **参数设置**：阈值25-35，最小区域300-500
- **特点**：能检测微小变化，减少误报

### 4.2 游戏精彩片段录制
- **配置建议**：选择"快速检测"组合
- **参数设置**：高帧率(60FPS)，开启渐隐效果
- **特点**：流畅捕捉快速动作场面

### 4.3 工业自动化检测
- **配置建议**：使用形态学处理算法
- **参数设置**：固定ROI区域，高阈值(40-50)
- **特点**：精确识别产品缺陷

## 5. 常见问题解答

### Q1: 为什么标记框会闪烁？
A: 尝试调高阈值或增大最小区域值，也可开启形态学处理减少噪声

### Q2: 如何降低CPU占用？
A: 1) 降低帧率 2) 开启分辨率缩放 3) 选择性能模式 4) 开启帧跳过

### Q3: 为什么实际帧率低于设置值？
A: 可能是：1) 硬件性能不足 2) 监控窗口太大 3) 算法太过复杂

## 6. 版本更新说明
当前版本：v71.4.8
更新内容：
- 新增智能算法组合功能
- 优化CUDA内存管理
- 修复了帧率过高警告逻辑问题
- 改进ROI跟踪稳定性

---

这份说明书综合考虑了不同用户群体的需求：
1. 初级用户：使用简单步骤和直观指导
2. 中级用户：提供实用技巧和参数建议
3. 高级用户：包含技术实现细节和代码示例
4. 专业开发者：算法原理和性能优化说明

每个部分都配有实际应用案例，便于用户理解如何在不同场景下配置系统。同时保持版本号递增（v71.4.8），专业部分给出具体的代码修改示例，既保证了易用性又满足了专业需求。

软件截图：


![image](https://github.com/user-attachments/assets/fbff66d1-7934-4b2a-bb0b-4e4a2b80797d)
![image](https://github.com/user-attachments/assets/ba4b8602-6906-425d-aeac-41fda4e1d57f)
![image](https://github.com/user-attachments/assets/de46507d-8b69-495e-ab9c-0057748107e1)
![image](https://github.com/user-attachments/assets/b1d01ddd-37d7-4af1-9537-ceabbe15b6e4)
![image](https://github.com/user-attachments/assets/8e5878f2-cdf1-494f-bcd0-1b927e90c386)



