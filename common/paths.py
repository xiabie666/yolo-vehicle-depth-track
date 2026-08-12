"""项目路径约定。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
ASSETS_DIR = ROOT / "assets"
OUTPUTS_DIR = ROOT / "outputs"

OUTPUTS_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)


def resolve_video() -> Path:
    """推理优先用无水印原片，避免重复叠加水印。"""
    raw = DATA_DIR / "video_raw.mp4"
    pub = DATA_DIR / "video.mp4"
    if raw.exists():
        return raw
    if pub.exists():
        return pub
    raise FileNotFoundError(f"找不到演示视频: {raw} 或 {pub}")


def weight_path(name: str) -> Path:
    """本地 weights/ 优先，否则交给 Ultralytics 自动下载。"""
    local = WEIGHTS_DIR / name
    return local if local.exists() else Path(name)
