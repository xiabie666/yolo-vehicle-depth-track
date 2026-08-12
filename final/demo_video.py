"""
【最终效果】车辆检测跟踪 + 单目深度测距
在跟踪框上显示「相机 → 该车」的估计距离（米）。
说明：单目深度为估算值，非激光/双目实测，请勿用于精确测距。
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
from common.vehicles import VEHICLE_IDS, VEHICLE_NAMES  # noqa: E402
from common.watermark import apply_watermark, build_watermark_layer  # noqa: E402


def box_distance(depth_m: np.ndarray, xyxy: np.ndarray) -> float | None:
    """取检测框中下部区域中位深度，作为该车到相机的距离（米）。"""
    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    bw, bh = x2 - x1, y2 - y1
    cx1 = x1 + int(bw * 0.25)
    cx2 = x1 + int(bw * 0.75)
    cy1 = y1 + int(bh * 0.45)
    cy2 = y1 + int(bh * 0.90)
    patch = depth_m[cy1:cy2, cx1:cx2]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size < 10:
        patch = depth_m[y1:y2, x1:x2]
        patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def draw_track(frame: np.ndarray, xyxy, track_id: int, name: str, dist_m: float | None) -> None:
    x1, y1, x2, y2 = map(int, xyxy)
    color = (0, 200, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID{track_id} {name} ?" if dist_m is None else f"ID{track_id} {name} {dist_m:.1f}m"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
    ty = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, ty), (x1 + tw + 8, ty + th + 8), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 4, ty + th + 2), font, 0.7, color, 2, cv2.LINE_AA)

    bw, bh = x2 - x1, y2 - y1
    sx1 = x1 + int(bw * 0.25)
    sx2 = x1 + int(bw * 0.75)
    sy1 = y1 + int(bh * 0.45)
    sy2 = y1 + int(bh * 0.90)
    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 1)


def main() -> None:
    source = resolve_video()
    detect_model = YOLO(str(weight_path("yolo26n.pt")))
    depth_model = YOLO(str(weight_path("yolo26n-depth.pt")))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(source)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = OUTPUTS_DIR / "final" / "video_vehicle_distance.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    wm = None
    idx = 0
    preview_saved = False

    print(f"[final] input: {source}")
    print("[final] pipeline: detect+track -> depth -> camera distance")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        tracks = detect_model.track(
            frame,
            persist=True,
            classes=list(VEHICLE_IDS),
            conf=0.25,
            verbose=False,
        )[0]
        depth_m = depth_model(frame, verbose=False)[0].depth.data.cpu().numpy()
        if depth_m.shape[:2] != frame.shape[:2]:
            depth_m = cv2.resize(depth_m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        out = frame.copy()
        if tracks.boxes is not None and len(tracks.boxes):
            xyxy = tracks.boxes.xyxy.cpu().numpy()
            cls = tracks.boxes.cls.cpu().numpy().astype(int)
            ids = tracks.boxes.id
            ids = ids.cpu().numpy().astype(int) if ids is not None else np.arange(len(xyxy))
            for box, c, tid in zip(xyxy, cls, ids):
                dist = box_distance(depth_m, box)
                draw_track(out, box, int(tid), VEHICLE_NAMES.get(int(c), str(int(c))), dist)

        cv2.putText(
            out,
            "vehicle track + camera distance (meters)",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if wm is None:
            wm = build_watermark_layer(out.shape[0], out.shape[1])
        out = apply_watermark(out, wm)

        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0]))
        writer.write(out)

        if not preview_saved and idx >= 160:
            preview = ROOT / "assets" / "04_final_vehicle_distance.jpg"
            cv2.imwrite(str(preview), out)
            preview_saved = True

        if idx % 50 == 0 or idx == total:
            print(f"[final] {idx}/{total}")

    cap.release()
    if writer:
        writer.release()
    print(f"[final] saved: {out_path}")


if __name__ == "__main__":
    main()
