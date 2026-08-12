"""
【工具】给 data/video_raw.mp4 打水印，生成可公开发布的 data/video.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import DATA_DIR  # noqa: E402
from common.watermark import DEFAULT_TEXT, apply_watermark, build_watermark_layer  # noqa: E402


def main() -> None:
    raw = DATA_DIR / "video_raw.mp4"
    pub = DATA_DIR / "video.mp4"
    if not raw.exists():
        # 若只有 video.mp4，先备份为 raw 再加水印覆盖发布版
        if not pub.exists():
            raise FileNotFoundError(f"找不到 {raw} 或 {pub}")
        pub.replace(raw)
        print(f"backup -> {raw}")

    cap = cv2.VideoCapture(str(raw))
    if not cap.isOpened():
        raise RuntimeError(raw)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    layer = build_watermark_layer(h, w, DEFAULT_TEXT)
    writer = cv2.VideoWriter(str(pub), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(apply_watermark(frame, layer))
        idx += 1
        if idx % 100 == 0 or idx == total:
            print(f"[watermark] {idx}/{total}")

    cap.release()
    writer.release()
    print(f"[watermark] saved: {pub}  text={DEFAULT_TEXT}")
    print(f"[watermark] raw kept: {raw}  (请勿上传到公开仓库)")


if __name__ == "__main__":
    main()
