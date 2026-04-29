"""도면 처리 추론 모듈 (OpenCV 베이스라인).

전략:
1. 그레이스케일 + 이진화 (벽이 검은 선이라 가정 → invert)
2. 모폴로지 closing 으로 끊긴 벽 연결
3. HoughLinesP 로 직선 검출
4. 거의 평행/근접한 선들 병합
5. mm 좌표계로 변환 (기본 1px = 10mm, 추후 캘리브레이션 UI에서 보정)

CubiCasa5K 학습 모델은 추후 이 파서를 대체/보강하는 식으로 추가.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


@dataclass
class WallSegment:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def angle_deg(self) -> float:
        """0 ~ 180 (선 방향은 무관)"""
        return float(np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1))) % 180


@dataclass
class ParseResult:
    walls: list[WallSegment]
    image_size: tuple[int, int]   # (width, height) px
    pixels_per_mm: float

    def to_dict(self) -> dict:
        return {
            "walls": [w.to_dict() for w in self.walls],
            "rooms": [],
            "openings": [],
            "meta": {
                "image_width": self.image_size[0],
                "image_height": self.image_size[1],
                "pixels_per_mm": self.pixels_per_mm,
                "coordinate_unit": "mm",
            },
        }


# ──────────────────────────────────────────────────────────────────
# 핵심 유틸
# ──────────────────────────────────────────────────────────────────


def _binarize(gray: np.ndarray) -> np.ndarray:
    """벽이 흰색이 되도록 이진화."""
    # 도면 배경은 보통 흰색, 벽은 검은색 → INV 적용
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # 끊어진 벽 잇기
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary


def _detect_lines(binary: np.ndarray, min_length_px: int) -> list[WallSegment]:
    """HoughLinesP로 직선 검출."""
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_length_px,
        maxLineGap=10,
    )
    if lines is None:
        return []
    return [
        WallSegment(float(x1), float(y1), float(x2), float(y2))
        for x1, y1, x2, y2 in lines[:, 0]
    ]


def _merge_collinear(
    walls: list[WallSegment],
    angle_tol_deg: float = 5.0,
    distance_tol_px: float = 8.0,
) -> list[WallSegment]:
    """거의 평행하고 가까운 선들을 그룹핑해서 하나로 병합."""
    if not walls:
        return walls

    used = [False] * len(walls)
    merged: list[WallSegment] = []

    for i, w in enumerate(walls):
        if used[i]:
            continue
        group_points: list[tuple[float, float]] = [(w.x1, w.y1), (w.x2, w.y2)]
        used[i] = True
        for j in range(i + 1, len(walls)):
            if used[j]:
                continue
            wj = walls[j]
            if abs(w.angle_deg - wj.angle_deg) > angle_tol_deg:
                # 0과 180 근처도 같은 방향으로 처리
                if abs(180 - abs(w.angle_deg - wj.angle_deg)) > angle_tol_deg:
                    continue
            # 두 선 사이 수직거리 (대표점 기준)
            dist = _point_to_line_distance(
                (wj.x1 + wj.x2) / 2,
                (wj.y1 + wj.y2) / 2,
                w.x1, w.y1, w.x2, w.y2,
            )
            if dist > distance_tol_px:
                continue
            group_points.extend([(wj.x1, wj.y1), (wj.x2, wj.y2)])
            used[j] = True

        # 그룹 내 최장 거리 두 점을 양 끝으로
        merged.append(_longest_pair_segment(group_points))
    return merged


def _point_to_line_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = float(np.hypot(y2 - y1, x2 - x1))
    return num / den if den > 1e-6 else 0.0


def _longest_pair_segment(points: list[tuple[float, float]]) -> WallSegment:
    best = (points[0], points[1], 0.0)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = float(
                np.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
            )
            if d > best[2]:
                best = (points[i], points[j], d)
    (x1, y1), (x2, y2), _ = best
    return WallSegment(x1, y1, x2, y2)


def _to_mm(walls: list[WallSegment], pixels_per_mm: float) -> list[WallSegment]:
    """이미지 픽셀 좌표 → mm 좌표 (1px = 1/pixels_per_mm mm)."""
    if pixels_per_mm <= 0:
        return walls
    inv = 1.0 / pixels_per_mm
    return [
        WallSegment(w.x1 * inv, w.y1 * inv, w.x2 * inv, w.y2 * inv)
        for w in walls
    ]


# ──────────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────────


class FloorPlanParser:
    """도면 이미지 → 벽 벡터 데이터.

    Args:
        pixels_per_mm: 이미지 1mm가 몇 픽셀인지. 기본 0.1 (= 10mm/px).
            예: 1000x800 도면이 약 10m x 8m 라고 가정.
        min_wall_length_mm: 이 길이보다 짧은 검출은 노이즈로 취급.
    """

    def __init__(
        self,
        pixels_per_mm: float = 0.1,
        min_wall_length_mm: float = 300.0,
    ):
        self.pixels_per_mm = pixels_per_mm
        self.min_wall_length_mm = min_wall_length_mm

    def parse_bytes(self, image_bytes: bytes) -> ParseResult:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("이미지 디코딩 실패")
        return self._parse(img)

    def parse_file(self, path: Path | str) -> ParseResult:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"이미지를 열 수 없음: {path}")
        return self._parse(img)

    def _parse(self, gray: np.ndarray) -> ParseResult:
        h, w = gray.shape[:2]
        logger.info(f"도면 파싱 시작: {w}x{h}, ppm={self.pixels_per_mm}")

        binary = _binarize(gray)
        min_length_px = max(20, int(self.min_wall_length_mm * self.pixels_per_mm))

        walls_px = _detect_lines(binary, min_length_px=min_length_px)
        logger.info(f"  Hough 검출 라인: {len(walls_px)}")

        walls_px = _merge_collinear(walls_px)
        logger.info(f"  병합 후 라인: {len(walls_px)}")

        walls_mm = _to_mm(walls_px, self.pixels_per_mm)
        return ParseResult(
            walls=walls_mm,
            image_size=(w, h),
            pixels_per_mm=self.pixels_per_mm,
        )
