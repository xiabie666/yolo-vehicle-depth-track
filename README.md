# YOLO Vehicle Depth Track

基于 **YOLO26** 的车辆智能感知 Demo：检测 + 多目标跟踪 + 单目深度测距一条龙。自动锁定画面中的车/公交/卡车，稳定跟踪并实时给出「相机到车辆」的距离读数，开箱即跑、效果直观。

> **定位**：交通演示 / AI 教学 / 作品集 / 二次开发脚手架。  
> **说明**：单目深度为估算，不是激光雷达或双目测距仪；请勿用于精确测距、执法或安全关键场景。

![AI 封面](assets/10_ai_cover_split.jpg)

## 效果预览

### 实拍视频抽帧（最终效果）

| 帧 1 | 帧 2 | 帧 3 |
|:---:|:---:|:---:|
| ![frame1](assets/06_video_frame_1.jpg) | ![frame2](assets/07_video_frame_2.jpg) | ![frame3](assets/08_video_frame_3.jpg) |

标签形如：`ID8 car 4.8m` → 该跟踪车辆到相机约 4.8 米（估算）。

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
| 最终效果 | 检测跟踪 + 深度测距 | `final/` |
| 水印工具 | OpenCV 叠加 | `tools/` |

一句话：**YOLO 检测跟踪车辆，再用单目深度给每辆车估到相机的距离。**

## 目录说明

```text
yolo-vehicle-depth-track/
├── depth/              # 【深度】只有单目深度估计
│   ├── demo_image.py   # 图片 Demo
│   └── demo_video.py   # 视频 Demo
├── detect_track/       # 【检测跟踪】只有车辆检测+跟踪
│   └── demo_video.py
├── final/              # 【最终】检测跟踪 + 距离显示（推荐看这个）
│   └── demo_video.py
├── tools/              # 工具：给原视频打水印
│   └── watermark_video.py
├── common/             # 公共：路径、车辆类别、水印
├── data/               # 演示数据（video.mp4 为带水印发布版）
├── assets/             # README 效果图
├── weights/            # 可选：本地放置 .pt 权重
└── outputs/            # 运行输出（默认不提交大视频）
```

## 快速开始

```bash
# 建议使用 conda / venv
pip install -r requirements.txt

# 1. 只看深度（图片）
python depth/demo_image.py

# 2. 只看车辆检测跟踪
python detect_track/demo_video.py

# 3. 最终效果：跟踪 + 距离（最完整）
python final/demo_video.py
```

首次运行会自动下载 `yolo26n.pt` / `yolo26n-depth.pt`（也可事先放到 `weights/`）。

演示视频：`data/video.mp4`（已加水印，可公开）。  
本地无水印原片请放 `data/video_raw.mp4`（已 gitignore，勿上传）。

## 距离含义

- **起点**：拍摄该画面的相机  
- **终点**：被跟踪车辆表面（框内中下部采样）  
- **不是**：两车之间的距离  

## 环境

- Python 3.10+
- Windows / Linux
- 有 GPU 会更快（CPU 也可跑通）

## 许可

MIT License。演示视频与水印归作者所有；YOLO / Ultralytics 权重请遵循其各自许可。

## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- YOLO26 Monocular Depth
