"""환경 점검 스크립트. 학습/추론 시작 전에 한 번 돌려본다.

실행:
    uv run python scripts/check_env.py
"""

import shutil
import sys


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    section("Python")
    print(f"버전: {sys.version}")

    section("PyTorch")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"MPS 사용 가능: {torch.backends.mps.is_available()}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"미설치: {e}")

    section("주요 라이브러리")
    for module in ("cv2", "numpy", "PIL", "ultralytics", "segment_anything", "fastapi"):
        try:
            m = __import__(module)
            ver = getattr(m, "__version__", "?")
            print(f"  ✓ {module}: {ver}")
        except ImportError:
            print(f"  ✗ {module}: 미설치")

    section("외부 CLI 도구 (3D 재구성용)")
    for tool in ("colmap", "meshlabserver", "DensifyPointCloud"):
        path = shutil.which(tool)
        print(f"  {'✓' if path else '✗'} {tool}: {path or '미설치'}")


if __name__ == "__main__":
    main()
