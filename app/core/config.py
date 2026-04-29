"""환경변수 기반 설정."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    app_port: int = 8001

    # 디바이스
    device: str = "cpu"  # mps / cuda / cpu

    # 모델 가중치 경로
    yolo_weights: Path = Path("models/weights/yolov8n.pt")
    sam_weights: Path = Path("models/weights/sam_vit_b.pth")

    # 외부 CLI 도구
    colmap_bin: str = "colmap"
    openmvs_bin_dir: str = "/usr/local/bin"
    meshlab_bin: str = "meshlabserver"

    # 작업 디렉터리
    work_dir: Path = Path("./outputs")


settings = Settings()
