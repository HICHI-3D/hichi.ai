"""모델 가중치 다운로드 스크립트.

실행:
    uv run python scripts/download_weights.py

다운로드 대상:
- YOLOv8n (가구 검출 베이스라인)
- SAM ViT-B (가구 마스크 분할용, 약 375MB)
"""

from pathlib import Path

import requests
from tqdm import tqdm

WEIGHTS_DIR = Path(__file__).parent.parent / "models" / "weights"

DOWNLOADS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "sam_vit_b.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} 이미 존재")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with (
        open(dest, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"[ok] {dest}")


def main() -> None:
    for filename, url in DOWNLOADS.items():
        download(url, WEIGHTS_DIR / filename)
    print("\n완료. 모델 가중치 위치:", WEIGHTS_DIR)


if __name__ == "__main__":
    main()
