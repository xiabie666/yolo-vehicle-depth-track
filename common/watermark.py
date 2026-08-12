"""统一水印：单个居中斜向大水印。"""

import cv2
import numpy as np

DEFAULT_TEXT = "XiaBie6666"


def build_watermark_layer(h: int, w: int, text: str = DEFAULT_TEXT) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(w, h) / 180.0
    thickness = max(5, int(scale * 2.5))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    pad = max(tw, th)
    big = np.zeros((h + pad * 2, w + pad * 2, 3), dtype=np.uint8)
    org = ((big.shape[1] - tw) // 2, (big.shape[0] + th) // 2)
    cv2.putText(big, text, org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    center = (big.shape[1] // 2, big.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, -25, 1.0)
    rotated = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]))
    return rotated[pad : pad + h, pad : pad + w]


def apply_watermark(frame: np.ndarray, layer: np.ndarray, alpha: float = 0.28) -> np.ndarray:
    out = frame.astype(np.float32)
    mask = (layer.sum(axis=2) > 0).astype(np.float32)[..., None]
    out = out * (1 - mask * alpha) + layer.astype(np.float32) * (mask * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)
