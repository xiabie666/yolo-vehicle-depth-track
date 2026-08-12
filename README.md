# YOLO Vehicle Depth Track

基于 **YOLO26** 的车辆智能感知 Demo：**检测 + 多目标跟踪 + 单目测距 + 测速** 一条龙。  
自动锁定画面中的车 / 公交 / 卡车，稳定跟踪，并实时给出「相机到车辆」的距离与估算速度，开箱即跑、效果直观。

> **定位**：交通演示 / AI 教学 / 作品集 / 二次开发脚手架。  
> **说明**：单目深度与速度均为估算，不是激光雷达、双目测距仪或交警测速仪；请勿用于精确测距测速、执法或安全关键场景。

![AI 封面](assets/10_ai_cover_split.jpg)

## Demo 场景

本仓库演示视频拍摄场景假设为：

**河南省郑州市 · 嵩山路天桥视角（俯拍道路）**

用于展示城市主干道天桥监控视角下的车辆跟踪、距离与速度演示效果。

## 效果预览

### 测速效果（NEW）

实拍抽帧（距离 + 速度 + 靠近/远离）：

![测速实拍帧](assets/11_speed_real_frame.jpg)

测速概念效果图：

![测速海报](assets/12_ai_speed_poster.jpg)

标签示例：`ID8 car 4.5m 18km/h 靠近`  
含义：跟踪到的车辆距相机约 4.5m，估算径向速度约 18km/h，正在靠近相机。

### 实拍视频抽帧（距离效果）

| 帧 1 | 帧 2 | 帧 3 |
|:---:|:---:|:---:|
| ![frame1](assets/06_video_frame_1.jpg) | ![frame2](assets/07_video_frame_2.jpg) | ![frame3](assets/08_video_frame_3.jpg) |

### 模块效果

**1）单目深度（只有深度）**

![深度对比](assets/01_depth_compare.jpg)

**2）车辆检测 + 跟踪（只有 ID）**

![检测跟踪](assets/03_detect_track_preview.jpg)

**3）最终效果（跟踪 + 距离）**

![最终效果](assets/04_final_vehicle_distance.jpg)

### 概念效果图

![场景海报](assets/09_ai_poster_scene.jpg)

## 技术组成

| 模块 | 技术 | 代码目录 |
|------|------|----------|
| 深度 | YOLO26 Monocular Depth | `depth/` |
| 检测跟踪 | YOLO Detect + ByteTrack | `detect_track/` |
| 测距 | 检测跟踪 + 深度采样 | `final/` |
| 测速 | 距离变化 ÷ 时间（Δd/Δt） | `speed/` |
| 水印工具 | OpenCV / 中文绘制 | `tools/` `common/` |

一句话：**YOLO 检测跟踪车辆 → 单目深度估距离 → 用距离变化估速度。**

### 测速怎么理解

```text
速度 ≈ Δ距离 / Δ时间
```

- 估的是相对相机的**径向速度**（靠近 / 远离），不是路面绝对车速  
- 深度有噪声，脚本内做了滑动平滑，结果仅供 Demo 展示

## 目录说明

```text
yolo-vehicle-depth-track/
├── depth/              # 【深度】只有单目深度估计
│   ├── demo_image.py
│   └── demo_video.py
├── detect_track/       # 【检测跟踪】只有车辆检测+跟踪
│   └── demo_video.py
├── final/              # 【测距】检测跟踪 + 距离显示
│   └── demo_video.py
├── speed/              # 【测速】检测跟踪 + 距离 + 速度（推荐）
│   └── demo_video.py
├── tools/              # 给原视频打水印
│   └── watermark_video.py
├── common/             # 公共：路径、车辆类别、水印、中文文字
├── data/               # 演示数据（嵩山路天桥 Demo 视频，带水印）
├── assets/             # README 效果图
├── weights/            # 可选：本地放置 .pt 权重
└── outputs/            # 运行输出
```

## 快速开始

```bash
# 建议使用 conda / venv
pip install -r requirements.txt

# 1. 只看深度（图片）
python depth/demo_image.py

# 2. 只看车辆检测跟踪
python detect_track/demo_video.py

# 3. 测距：跟踪 + 距离
python final/demo_video.py

# 4. 测速：跟踪 + 距离 + 速度（最完整）
python speed/demo_video.py
```

首次运行会自动下载 `yolo26n.pt` / `yolo26n-depth.pt`（也可事先放到 `weights/`）。

演示视频：`data/video.mp4`（郑州嵩山路天桥视角 Demo，已加水印，可公开）。  
本地无水印原片请放 `data/video_raw.mp4`（已 gitignore，勿上传）。

## 距离 / 速度含义

| 项目 | 含义 |
|------|------|
| 距离起点 | 拍摄该画面的相机 |
| 距离终点 | 被跟踪车辆表面（框内中下部采样） |
| 速度 | 上述距离随时间的变化率（Δd/Δt） |
| 不是 | 两车之间的距离、路面绝对车速、执法级测速 |

## 环境

- Python 3.10+
- Windows / Linux
- 有 GPU 会更快（CPU 也可跑通）
- 中文标签依赖系统字体（Windows 下自动使用微软雅黑等）

## 许可

MIT License。演示视频与水印归作者所有；YOLO / Ultralytics 权重请遵循其各自许可。

## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- YOLO26 Monocular Depth
