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

    # Google Gemini Vision (도면 파서) — 키가 설정되면 OpenCV 대신 Vision을 우선 사용
    # 무료 키 발급: https://aistudio.google.com/apikey
    #
    # 무료 티어 모델 (2026 기준):
    #   - gemini-2.5-flash       (분당 10회, 하루 500회, 정확도 좋음) ← 기본
    #   - gemini-2.0-flash       (분당 15회, 하루 1500회, 덜 정확)
    #   - gemini-2.0-flash-lite  (분당 30회, 하루 1500회, 가장 빠름)
    # ⚠️  gemini-2.5-pro / gemini-1.5-pro 는 무료 티어 미지원 (429 RESOURCE_EXHAUSTED 발생)
    #
    # .env에서 VISION_PARSER_MODEL=gemini-2.0-flash 같이 덮어쓰기 가능.
    gemini_api_key: str | None = None
    vision_parser_model: str = "gemini-2.5-flash"


settings = Settings()
