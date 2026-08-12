"""
【速度 Demo】车辆跟踪 + 单目深度 → 用距离变化估算速度

理解方式：
  速度 ≈ Δ距离 / Δ时间

注意：
  这里估的是「相机到车辆」距离的变化率（径向接近/远离速度），
  不是路面真实车速；单目深度有噪声，已做平滑，结果仅供演示。
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import OUTPUTS_DIR, resolve_video, weight_path  # noqa: E402
from common.text_cn import put_text_cn  # noqa: E402
from common.vehicles import VEHICLE_IDS, VEHICLE_NAMES  # noqa: E402
from common.watermark import apply_watermark, build_watermark_layer  # noqa: E402


def box_distance(depth_m: np.ndarray, xyxy: np.ndarray) -> float | None:
    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    bw, bh = x2 - x1, y2 - y1
    patch = depth_m[
        y1 + int(bh * 0.45) : y1 + int(bh * 0.90),
        x1 + int(bw * 0.25) : x1 + int(bw * 0.75),
    ]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size < 10:
        patch = depth_m[y1:y2, x1:x2]
        patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return None
    return float(np.median(patch))


class SpeedEstimator:
    """按 track_id 记录距离，用 Δd/Δt 估速度，并做滑动平均平滑。"""

    def __init__(self, window: int = 8, min_dt: float = 0.08):
        self.window = window
        self.min_dt = min_dt
        self.hist: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id: int, t_sec: float, dist_m: float | None) -> tuple[float | None, float | None, str]:
        """
        返回: (距离m, 速度km/h, 方向文案)
        速度为正：远离相机；为负：靠近相机。显示时用绝对值 + 方向。
        """
        if dist_m is None:
            return None, None, ""

        q = self.hist[track_id]
        q.append((t_sec, dist_m))
        if len(q) < 2:
            return dist_m, None, ""

        # 用窗口两端估算，抗一点抖动
        t0, d0 = q[0]
        t1, d1 = q[-1]
        dt = t1 - t0
        if dt < self.min_dt:
            return dist_m, None, ""

        speed_mps = (d1 - d0) / dt  # m/s，相对相机径向
        # 再对最近几段瞬时速度做平均
        inst = []
        items = list(q)
        for i in range(1, len(items)):
            dti = items[i][0] - items[i - 1][0]
            if dti >= 1e-3:
                inst.append((items[i][1] - items[i - 1][1]) / dti)
        if inst:
            speed_mps = float(np.median(inst))

        speed_kmh = speed_mps * 3.6
        if abs(speed_kmh) < 1.0:
            direction = "静止/缓行"
        elif speed_kmh > 0:
            direction = "远离"
        else:
            direction = "靠近"
        return dist_m, speed_kmh, direction


def draw_label(frame, xyxy, track_id: int, name: str, dist_m, speed_kmh, direction: str) -> None:
    x1, y1, x2, y2 = map(int, xyxy)
    color = (0, 200, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    parts = [f"ID{track_id}", name]
    if dist_m is not None:
        parts.append(f"{dist_m:.1f}m")
    if speed_kmh is not None:
        parts.append(f"{abs(speed_kmh):.0f}km/h")
        if direction:
            parts.append(direction)
    label = " ".join(parts)

    # OpenCV putText 不支持中文，改用系统字体绘制
    font_size = 22
    # 预估背景高度，把标签放在框上方
    ty = max(0, y1 - font_size - 12)
    put_text_cn(frame, label, (x1, ty), font_size=font_size, color=color, bg=(0, 0, 0))


def main() -> None:
    source = resolve_video()
    detect_model = YOLO(str(weight_path("yolo26n.pt")))
    depth_model = YOLO(str(weight_path("yolo26n-depth.pt")))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(source)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = OUTPUTS_DIR / "speed" / "video_vehicle_speed.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    wm = None
    idx = 0
    estimator = SpeedEstimator(window=10)

    print(f"[speed] input: {source}")
    print("[speed] formula: speed ≈ Δdistance / Δtime  (range-rate to camera)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        t_sec = (idx - 1) / fps

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
                dist_s, speed_kmh, direction = estimator.update(int(tid), t_sec, dist)
                draw_label(
                    out,
                    box,
                    int(tid),
                    VEHICLE_NAMES.get(int(c), str(int(c))),
                    dist_s,
                    speed_kmh,
                    direction,
                )

        cv2.putText(
            out,
            "speed ~ ds/dt (camera range-rate, demo only)",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if wm is None:
            wm = build_watermark_layer(out.shape[0], out.shape[1])
        out = apply_watermark(out, wm)

        if writer is None:
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (out.shape[1], out.shape[0]),
            )
        writer.write(out)
        if idx % 50 == 0 or idx == total:
            print(f"[speed] {idx}/{total}")

    cap.release()
    if writer:
        writer.release()
    print(f"[speed] saved: {out_path}")


if __name__ == "__main__":
    main()
