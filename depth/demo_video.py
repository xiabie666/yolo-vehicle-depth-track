"""
【深度】单目深度估计 - 视频 Demo
只输出深度伪彩对比，不含车辆检测/跟踪。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import OUTPUTS_DIR, resolve_video, weight_path  # noqa: E402
from common.watermark import apply_watermark, build_watermark_layer  # noqa: E402


def depth_range(depth_m: np.ndarray) -> tuple[float, float, np.ndarray]:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return 0.1, 10.0, valid
    lo, hi = np.percentile(depth_m[valid], [2, 98])
    return float(lo), float(max(hi, lo + 1e-6)), valid


def colorize(depth_m: np.ndarray, lo: float, hi: float, valid: np.ndarray) -> np.ndarray:
    norm = np.clip((depth_m - lo) / (hi - lo), 0, 1)
    norm[~valid] = 0
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def draw_colorbar(img: np.ndarray, lo: float, hi: float, width: int = 28) -> np.ndarray:
    h, w = img.shape[:2]
    bar = np.linspace(1, 0, h, dtype=np.float32).reshape(h, 1)
    bar = np.repeat(bar, width, axis=1)
    bar_color = cv2.applyColorMap((bar * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    pad = 70
    canvas = np.full((h, w + width + pad, 3), 255, dtype=np.uint8)
    canvas[:, :w] = img
    canvas[:, w : w + width] = bar_color
    for t in np.linspace(0, 1, 5):
        y = int(t * (h - 1))
        meters = hi - t * (hi - lo)
        cv2.putText(canvas, f"{meters:.1f}m", (w + width + 6, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    source = resolve_video()
    model = YOLO(str(weight_path("yolo26n-depth.pt")))
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(source)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = OUTPUTS_DIR / "depth" / "video_depth.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    wm = None
    idx = 0

    print(f"[depth] input: {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        depth_m = model(frame, verbose=False)[0].depth.data.cpu().numpy()
        if depth_m.shape[:2] != frame.shape[:2]:
            depth_m = cv2.resize(depth_m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        lo, hi, valid = depth_range(depth_m)
        depth_vis = cv2.resize(colorize(depth_m, lo, hi, valid), (frame.shape[1], frame.shape[0]))
        out = np.hstack([frame, draw_colorbar(depth_vis, lo, hi)])
        if wm is None:
            wm = build_watermark_layer(out.shape[0], out.shape[1])
        out = apply_watermark(out, wm)
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0]))
        writer.write(out)
        if idx % 50 == 0 or idx == total:
            print(f"[depth] {idx}/{total}")

    cap.release()
    if writer:
        writer.release()
    print(f"[depth] saved: {out_path}")


if __name__ == "__main__":
    main()
