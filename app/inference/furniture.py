"""가구 인식 추론 모듈.

YOLO로 가구 영역 검출 → SAM으로 정밀 분할.
TODO 마커 위치에 실제 모델 로드/추론 코드 작성.
"""

from pathlib import Path

from loguru import logger


class FurnitureDetector:
    """YOLO 기반 가구 검출기. (스켈레톤)"""

    def __init__(self, weights_path: Path, device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device
        self._model = None

    def load(self):
        # TODO: from ultralytics import YOLO; self._model = YOLO(self.weights_path)
        logger.info(f"YOLO 가중치 로드 (placeholder): {self.weights_path}")

    def detect(self, image_path: Path) -> list[dict]:
        """이미지 → [{class, bbox, confidence}, ...]"""
        # TODO: results = self._model(image_path); 결과 파싱
        return []


class FurnitureSegmenter:
    """SAM 기반 가구 마스크 분할기. (스켈레톤)"""

    def __init__(self, weights_path: Path, device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device
        self._predictor = None

    def load(self):
        # TODO: from segment_anything import sam_model_registry, SamPredictor
        logger.info(f"SAM 가중치 로드 (placeholder): {self.weights_path}")

    def segment(self, image_path: Path, bbox: list[float]) -> list[list[int]]:
        """bbox 힌트로 마스크 생성 → 윤곽선 좌표 리스트"""
        # TODO: 마스크 추론
        return []
