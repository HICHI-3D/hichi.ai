"""가구 인식 추론 모듈.

YOLO로 가구 영역 검출 → SAM으로 정밀 분할.

`classify_category(image_path)` 는 첫 사진에서 가장 가능성 높은 가구 카테고리를
한글 라벨로 반환한다. 학습된 YOLOv8 가중치 없이도 COCO 사전학습 가중치
(`yolov8n.pt`) 로 동작한다. 가중치 경로는 `settings.yolo_weights`
(기본값 `models/weights/yolov8n.pt`) 를 따르며, 없으면
`scripts/download_weights.py` 를 먼저 실행해야 한다 — 파일명만 넘기면
ultralytics 가 cwd 에 자동 다운로드해 워킹 트리를 더럽힌다.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

from loguru import logger

from app.core.config import settings

# 한글 카테고리 = 프론트엔드 FurnitureCategory 와 동일해야 함.
# (entities/furniture/model/types.ts 참고)
FurnitureCategoryKO = Literal["침대", "책상", "의자", "소파", "수납장", "기타"]

# COCO 클래스 id → 한글 가구 카테고리.
# ultralytics 의 yolov8n.pt 는 80개 COCO 클래스를 사용한다.
# 우리는 가구로 분류 가능한 것만 매핑하고 나머지는 무시한다.
_COCO_TO_CATEGORY: dict[int, FurnitureCategoryKO] = {
    56: "의자",      # chair
    57: "소파",      # couch
    59: "침대",      # bed
    60: "책상",      # dining table
    # 수납장에 직접 대응되는 COCO 클래스가 없어서, tv/book 다수가 잡히면 책장으로 추정.
    # 다만 그건 휴리스틱이라 일단 안 함. 부족분은 fallback 으로 처리.
}


class FurnitureClassifier:
    """첫 사진을 보고 가구 카테고리를 추정한다.

    - ultralytics 미설치/모델 다운로드 실패 시 `classify()` 는 '기타' 를 반환.
    - 가구로 분류된 박스가 여러 개면 confidence 합이 가장 큰 카테고리를 채택.
    """

    def __init__(self, weights: str | Path | None = None, device: str | None = None):
        # weights 를 명시 안 하면 settings.yolo_weights (=models/weights/yolov8n.pt).
        # 파일명만 넘기면 ultralytics 가 cwd 에 다운로드해서 워킹 트리를 더럽힘 — 그래서 항상 경로로 넘긴다.
        self.weights = str(weights) if weights else str(settings.yolo_weights)
        self.device = device or settings.device
        self._model = None
        self._load_failed = False

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            from ultralytics import YOLO  # type: ignore

            logger.info(f"YOLO 카테고리 분류기 로드: {self.weights} ({self.device})")
            self._model = YOLO(self.weights)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"YOLO 로드 실패, 카테고리 자동 인식 비활성화: {e}")
            self._load_failed = True
            return False

    def classify(self, image_path: Path) -> FurnitureCategoryKO:
        """이미지 한 장 → 가구 카테고리 (실패 시 '기타')."""
        if not self._ensure_loaded():
            return "기타"
        try:
            assert self._model is not None
            results = self._model.predict(
                source=str(image_path),
                device=self.device,
                conf=0.25,
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"YOLO 추론 실패: {e}")
            return "기타"

        # 카테고리별 누적 confidence 가 가장 큰 것을 채택
        scores: dict[FurnitureCategoryKO, float] = defaultdict(float)
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            # cls, conf 는 보통 tensor.
            cls_iter = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else list(boxes.cls)
            conf_iter = boxes.conf.tolist() if hasattr(boxes.conf, "tolist") else list(boxes.conf)
            for cls_id, conf in zip(cls_iter, conf_iter):
                category = _COCO_TO_CATEGORY.get(int(cls_id))
                if category is None:
                    continue
                scores[category] += float(conf)

        if not scores:
            return "기타"
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        logger.debug(f"카테고리 자동 인식: {best} (점수={dict(scores)})")
        return best


# 싱글톤 (모델 가중치는 첫 호출 시 lazy 로드)
classifier = FurnitureClassifier()


def classify_category(image_path: Path) -> FurnitureCategoryKO:
    """단일 이미지에 대해 가구 카테고리를 추정한다.

    Args:
        image_path: 분류할 이미지 (보통 업로드된 사진 중 첫 번째).

    Returns:
        한글 가구 카테고리. 모델 로드/추론에 실패하거나 가구가 검출되지 않으면 '기타'.
    """
    return classifier.classify(image_path)


# ──────────────────────────────────────────────────────────────────
# 아래는 기존 라우트(POST /detect, /segment) 가 참조하던 스켈레톤.
# reconstruction 라우트에서는 사용하지 않지만, 호환을 위해 남겨둔다.
# ──────────────────────────────────────────────────────────────────


class FurnitureDetector:
    """YOLO 기반 가구 검출기. (스켈레톤)"""

    def __init__(self, weights_path: Path, device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device
        self._model = None

    def load(self):
        logger.info(f"YOLO 가중치 로드 (placeholder): {self.weights_path}")

    def detect(self, image_path: Path) -> list[dict]:
        """이미지 → [{class, bbox, confidence}, ...]"""
        return []


class FurnitureSegmenter:
    """SAM 기반 가구 마스크 분할기. (스켈레톤)"""

    def __init__(self, weights_path: Path, device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device
        self._predictor = None

    def load(self):
        logger.info(f"SAM 가중치 로드 (placeholder): {self.weights_path}")

    def segment(self, image_path: Path, bbox: list[float]) -> list[list[int]]:
        return []
