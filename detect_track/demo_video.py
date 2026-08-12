"""
【检测跟踪】车辆检测 + 多目标跟踪 Demo
只做 YOLO detect/track，不估计距离。
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


def draw_track(frame: np.ndarray, xyxy, track_id: int, name: str) -> None:
    x1, y1, x2, y2 = map(int, xyxy)
    color = (0, 200, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID{track_id} {name}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
    ty = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, ty), (x1 + tw + 8, ty + th + 8), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 4, ty + th + 2), font, 0.7, color, 2, cv2.LINE_AA)


def main() -> None:
    source = resolve_video()
    model = YOLO(str(weight_path("yolo26n.pt")))
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(source)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = OUTPUTS_DIR / "detect_track" / "video_track.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    wm = None
    idx = 0
    preview_saved = False

    print(f"[detect_track] input: {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        tracks = model.track(
            frame,
            persist=True,
            classes=list(VEHICLE_IDS),
            conf=0.25,
            verbose=False,
        )[0]

        out = frame.copy()
        if tracks.boxes is not None and len(tracks.boxes):
            xyxy = tracks.boxes.xyxy.cpu().numpy()
            cls = tracks.boxes.cls.cpu().numpy().astype(int)
            ids = tracks.boxes.id
            ids = ids.cpu().numpy().astype(int) if ids is not None else np.arange(len(xyxy))
            for box, c, tid in zip(xyxy, cls, ids):
                draw_track(out, box, int(tid), VEHICLE_NAMES.get(int(c), str(int(c))))

        cv2.putText(out, "vehicle detect + track", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if wm is None:
            wm = build_watermark_layer(out.shape[0], out.shape[1])
        out = apply_watermark(out, wm)

        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0]))
        writer.write(out)

        if not preview_saved and idx >= 160:
            preview = ROOT / "assets" / "03_detect_track_preview.jpg"
            cv2.imwrite(str(preview), out)
            preview_saved = True
            print(f"[detect_track] preview: {preview}")

        if idx % 50 == 0 or idx == total:
            print(f"[detect_track] {idx}/{total}")

    cap.release()
    if writer:
        writer.release()
    print(f"[detect_track] saved: {out_path}")


if __name__ == "__main__":
    main()
