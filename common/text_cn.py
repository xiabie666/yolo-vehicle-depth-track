"""用系统中文字体画文字（OpenCV putText 不支持中文会乱码）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in _FONT_CANDIDATES:
        if path.exists():
            try:
                font = ImageFont.truetype(str(path), size=size)
                _FONT_CACHE[size] = font
                return font
            except OSError:
                continue
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def put_text_cn(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_size: int = 22,
    color: tuple[int, int, int] = (0, 200, 255),
    bg: tuple[int, int, int] | None = (0, 0, 0),
    padding: int = 4,
) -> np.ndarray:
    """
    在 BGR 图像上绘制可含中文的文字。
    org: 文字左上角（含背景时为背景框左上角）。
    """
    if not text:
        return frame

    font = _get_font(font_size)
    # 先量尺寸
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw, th = right - left, bottom - top

    x, y = org
    x2, y2 = x + tw + padding * 2, y + th + padding * 2
    h, w = frame.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x or y2 <= y:
        return frame

    roi = frame[y:y2, x:x2]
    pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    if bg is not None:
        draw.rectangle((0, 0, x2 - x - 1, y2 - y - 1), fill=(bg[2], bg[1], bg[0]))
    # PIL 用 RGB
    draw.text((padding - left, padding - top), text, font=font, fill=(color[2], color[1], color[0]))
    frame[y:y2, x:x2] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return frame
