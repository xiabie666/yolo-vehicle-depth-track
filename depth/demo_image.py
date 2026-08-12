"""
【深度】单目深度估计 - 图片 Demo
只做 YOLO26 Depth，不含车辆检测/跟踪。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import DATA_DIR, OUTPUTS_DIR, weight_path  # noqa: E402


def depth_stats(depth_m: np.ndarray) -> tuple[np.ndarray, float, float]:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        raise RuntimeError("深度图无效")
    lo, hi = np.percentile(depth_m[valid], [2, 98])
    return valid, float(lo), float(max(hi, lo + 1e-6))


def colorize_depth(depth_m: np.ndarray, lo: float, hi: float, valid: np.ndarray) -> np.ndarray:
    norm = np.clip((depth_m - lo) / (hi - lo), 0, 1)
    norm[~valid] = 0
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def draw_colorbar(img: np.ndarray, lo: float, hi: float, width: int = 36) -> np.ndarray:
    h, w = img.shape[:2]
    bar = np.linspace(1, 0, h, dtype=np.float32).reshape(h, 1)
    bar = np.repeat(bar, width, axis=1)
    bar_color = cv2.applyColorMap((bar * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    pad = 78
    canvas = np.full((h, w + width + pad, 3), 255, dtype=np.uint8)
    canvas[:, :w] = img
    canvas[:, w : w + width] = bar_color
    font = cv2.FONT_HERSHEY_SIMPLEX
    for t in np.linspace(0, 1, 6):
        y = int(t * (h - 1))
        meters = hi - t * (hi - lo)
        cv2.putText(canvas, f"{meters:.1f}m", (w + width + 10, y + 5), font, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def annotate_distances(orig: np.ndarray, depth_m: np.ndarray, rows: int = 4, cols: int = 5) -> np.ndarray:
    out = orig.copy()
    h, w = depth_m.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(rows):
        for j in range(cols):
            y = int((i + 0.5) * h / rows)
            x = int((j + 0.5) * w / cols)
            patch = depth_m[max(0, y - 4) : min(h, y + 5), max(0, x - 4) : min(w, x + 5)]
            patch = patch[np.isfinite(patch) & (patch > 0)]
            if patch.size == 0:
                continue
            text = f"{float(np.median(patch)):.1f}m"
            (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
            cv2.rectangle(out, (x - tw // 2 - 3, y - th - 6), (x + tw // 2 + 3, y + 4), (0, 0, 0), -1)
            cv2.circle(out, (x, y), 3, (0, 255, 255), -1)
            cv2.putText(out, text, (x - tw // 2, y - 4), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    source = DATA_DIR / "bus.jpg"
    model = YOLO(str(weight_path("yolo26n-depth.pt")))
    result = model(str(source) if source.exists() else "https://ultralytics.com/images/bus.jpg")[0]
    depth_m = result.depth.data.cpu().numpy()
    orig = result.orig_img
    if orig is None:
        raise RuntimeError("未能读取原图")
    if depth_m.shape[:2] != orig.shape[:2]:
        depth_m = cv2.resize(depth_m, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

    valid, lo, hi = depth_stats(depth_m)
    depth_vis = draw_colorbar(colorize_depth(depth_m, lo, hi, valid), lo, hi)
    annotated = annotate_distances(orig, depth_m)
    compare = np.hstack([annotated, depth_vis])

    out_dir = OUTPUTS_DIR / "depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "compare.jpg"), compare)
    cv2.imwrite(str(out_dir / "annotated.jpg"), annotated)
    print(f"saved: {out_dir / 'compare.jpg'}")


if __name__ == "__main__":
    main()
