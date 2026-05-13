"""도면 처리 추론 모듈 (OpenCV 베이스라인).

전략 (컨투어 기반, v2):
1. 그레이스케일 + 적응형 이진화 우선, 실패 시 dark-threshold/Otsu fallback
2. Opening 7×7 으로 텍스트/노이즈 제거 (두꺼운 마루 줄무늬도 제거)
3. Closing 7×7 으로 끊긴 벽 연결
4. Connected-component 필터로 작은 블롭 제거
5. [NEW] findContours + approxPolyDP 로 벽 외곽선 추출 → WallSegment
6. 직교 방향 필터 (수평/수직 ±8° 이내만 유지)
7. 축 스냅 (완전 수평/수직으로 정렬)
8. 거의 평행/근접한 선들 병합
9. 최소 길이 필터 (mm 변환 전)
10. mm 좌표계로 변환

[LEGACY] _detect_lines (HoughLinesP) 는 파일에 보존되나 기본 경로에서 호출되지 않음.
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


def _binarize(gray: np.ndarray, dark_threshold: int = 70) -> np.ndarray:
    """벽이 흰색이 되도록 이진화.

    1차 (NEW): adaptiveThreshold (Gaussian, blockSize=51, C=10) — 밝기가 고르지 않은
        연한-회색 벽 도면에 효과적. 흰 픽셀 비율이 0.5%~30% 사이일 때 채택.
    2차 fallback: 고정 dark_threshold 로 엄격하게 어두운 픽셀만 추출.
    3차 fallback: 2차 결과가 너무 희박하면 (< 0.5% 화소) Otsu fallback.
    """
    h, w = gray.shape[:2]
    total_pixels = h * w

    # 1차: Adaptive Gaussian Threshold (연한-회색 벽 처리)
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,   # blockSize
        10,   # C
    )
    adaptive_count = int(np.count_nonzero(adaptive))
    adaptive_ratio = adaptive_count / total_pixels

    if 0.005 <= adaptive_ratio <= 0.30:
        logger.info(
            f"  이진화: adaptiveThreshold(Gaussian, 51, 10) → 흰 픽셀 비율={adaptive_ratio:.4f} "
            f"→ 적응형 이진화 채택"
        )
        return adaptive

    logger.info(
        f"  이진화: adaptiveThreshold 결과={adaptive_ratio:.4f} (범위 0.5%~30% 벗어남) "
        f"→ fixed dark_threshold={dark_threshold} fallback"
    )

    # 2차: 고정 dark threshold
    _, binary = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    white_count = int(np.count_nonzero(binary))
    ratio = white_count / total_pixels

    if ratio < 0.005:
        logger.info(
            f"  이진화: dark_threshold={dark_threshold} → 흰 픽셀 비율={ratio:.4f} (<0.5%) "
            f"→ Otsu fallback 사용"
        )
        # 3차: Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        white_count = int(np.count_nonzero(binary))
        ratio = white_count / total_pixels
        logger.info(f"  이진화 (Otsu): 흰 픽셀 비율={ratio:.4f}")
    else:
        logger.info(
            f"  이진화: dark_threshold={dark_threshold} → 흰 픽셀 비율={ratio:.4f} (고정 임계값 사용)"
        )

    return binary


def _morphology_clean(binary: np.ndarray) -> np.ndarray:
    """Opening 7×7 + Closing 7×7 으로 노이즈 제거 및 벽 연결."""
    # Opening: 텍스트 획, 단일 픽셀 선, 바닥 텍스처 제거 (7×7 으로 두꺼운 마루 줄무늬도 제거)
    kernel_7_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_7_open)
    white_after_open = int(np.count_nonzero(opened))
    logger.info(f"  Opening 7×7 후 흰 픽셀: {white_after_open}")

    # Closing: 문 개구부 등으로 끊긴 벽 잇기
    kernel_7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_7)
    white_after_close = int(np.count_nonzero(closed))
    logger.info(f"  Closing 7×7 후 흰 픽셀: {white_after_close}")

    return closed


def _filter_components(binary: np.ndarray, min_area_ratio: float = 0.0008) -> np.ndarray:
    """Connected-component 필터: 가장 큰 컴포넌트(들)만 유지.

    실제 아파트 도면에서 벽은 하나의 거대한 연결 컴포넌트를 이루고,
    마루 줄무늬 / 타일 패턴은 서로 분리된 작은 블롭으로 나타남.
    → 상위 1~2개의 컴포넌트만 유지하고 나머지는 제거.
    """
    h, w = binary.shape[:2]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 배경(label 0) 제외한 (label_id, area) 리스트, 면적 내림차순
    label_areas = sorted(
        [(idx, int(stats[idx, cv2.CC_STAT_AREA])) for idx in range(1, num_labels)],
        key=lambda x: x[1],
        reverse=True,
    )

    total_fg = sum(a for _, a in label_areas)

    # 상위 3개 로그용 정보
    top3_info = [
        f"label={lid} area={area} ({area / total_fg * 100:.1f}%)"
        for lid, area in label_areas[:3]
    ]
    logger.info(
        f"  Connected-component: 총 {len(label_areas)}개 컴포넌트, "
        f"전경 면적 합={total_fg}px² | 상위 3: {top3_info}"
    )

    # 유지할 컴포넌트 결정: 기본 상위 1개
    keep_ids: list[int] = []
    if label_areas:
        top1_id, top1_area = label_areas[0]
        keep_ids.append(top1_id)

        # 발코니/서비스 영역 등 분리된 구조체가 있으면 상위 2개 유지
        if len(label_areas) >= 2:
            top2_id, top2_area = label_areas[1]
            add_second = (
                top1_area / max(total_fg, 1) < 0.25  # 1위도 전체의 25% 미만이면 분산됨
                or top2_area / max(top1_area, 1) > 0.4  # 2위가 1위의 40% 이상이면 독립 구조
            )
            if add_second:
                keep_ids.append(top2_id)

    # 벨트+멜빵: 절대 면적이 너무 작은 건 제거
    abs_min_area = max(0.01 * total_fg, min_area_ratio * h * w)
    keep_ids = [lid for lid in keep_ids if dict(label_areas)[lid] >= abs_min_area]

    logger.info(
        f"  → {len(keep_ids)}개 유지 (abs_min_area={abs_min_area:.0f}px²): "
        f"{[dict(label_areas)[lid] for lid in keep_ids]}"
    )

    if not keep_ids:
        return np.zeros_like(binary)

    kept_mask = np.isin(labels, keep_ids).astype(np.uint8) * 255
    return kept_mask


def _detect_lines(binary: np.ndarray, min_length_px: int) -> list[WallSegment]:
    """[구 방식] HoughLinesP 로 직선 검출. 호환성을 위해 남겨둠 (현재 미사용)."""
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 360,   # half-degree precision
        threshold=120,
        minLineLength=min_length_px,
        maxLineGap=15,
    )
    if lines is None:
        logger.info("  HoughLinesP: 검출된 라인 없음")
        return []
    result = [
        WallSegment(float(x1), float(y1), float(x2), float(y2))
        for x1, y1, x2, y2 in lines[:, 0]
    ]
    logger.info(f"  HoughLinesP 검출 라인: {len(result)}")
    return result


def _detect_walls_via_contours(
    binary: np.ndarray,
    min_contour_area_ratio: float = 0.005,
    max_contours: int = 8,
    epsilon_ratio: float = 0.003,
) -> list[WallSegment]:
    """벽 네트워크의 외곽선을 따서 다각형으로 단순화한 뒤, 각 변을 벽으로 변환.

    아이디어 (사용자 통찰):
        진짜 벽들은 하나의 연결된 검은 네트워크를 이룬다 (문이 있어도 형태적으로
        morphology close 로 메워짐). 그 네트워크의 외곽선이 곧 건물 윤곽선이며,
        내부 contour 들은 방 경계가 된다.
        → 외곽선 + 내부 contour 들을 cv2.approxPolyDP 로 단순화하고, 각 변을 벽 세그먼트로.

    Args:
        binary: 이미 morphology + 컴포넌트 필터를 거친 0/255 마스크.
        min_contour_area_ratio: 이미지 면적 대비 이 비율보다 작은 contour 무시.
        max_contours: 면적 내림차순 정렬 후 상위 N개만 유지.
        epsilon_ratio: approxPolyDP 의 epsilon = 둘레 * 이 값. 클수록 거칠게.
    """
    h, w = binary.shape[:2]
    image_area = float(h * w)

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        logger.info("  findContours: contour 없음")
        return []

    # 면적과 함께 보관, 면적 내림차순
    scored: list[tuple[float, np.ndarray]] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_contour_area_ratio * image_area:
            continue
        if len(c) < 3:
            continue
        scored.append((area, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    kept = scored[:max_contours]

    logger.info(
        f"  findContours: 총 {len(contours)}개 → 면적/길이 필터 후 {len(scored)}개 → "
        f"상위 {len(kept)}개 유지"
    )

    walls: list[WallSegment] = []
    for idx, (area, c) in enumerate(kept):
        perimeter = float(cv2.arcLength(c, closed=True))
        epsilon = max(2.0, epsilon_ratio * perimeter)
        approx = cv2.approxPolyDP(c, epsilon, closed=True)

        n = len(approx)
        if n < 3:
            continue

        pts = approx.reshape(-1, 2)  # shape (n, 2)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            walls.append(
                WallSegment(float(x1), float(y1), float(x2), float(y2))
            )

        logger.info(
            f"    contour #{idx}: area={area:.0f}px², perimeter={perimeter:.0f}px, "
            f"→ {n}개 꼭지점 → {n}개 벽"
        )

    logger.info(f"  Contour 기반 벽 추출 총: {len(walls)}개")
    return walls


def _orientation_filter(
    walls: list[WallSegment],
    angle_tol_deg: float = 8.0,
) -> list[WallSegment]:
    """수평(0°/180°) 또는 수직(90°) 에서 ±angle_tol_deg 이내인 선만 유지."""
    kept: list[WallSegment] = []
    for w in walls:
        a = w.angle_deg  # 0 ~ 180
        is_horiz = a <= angle_tol_deg or a >= (180.0 - angle_tol_deg)
        is_vert = abs(a - 90.0) <= angle_tol_deg
        if is_horiz or is_vert:
            kept.append(w)
    logger.info(
        f"  직교 방향 필터 (±{angle_tol_deg}°): {len(walls)} → {len(kept)}"
    )
    return kept


def _snap_to_axis(walls: list[WallSegment]) -> list[WallSegment]:
    """각 선을 완전한 수평 또는 수직으로 스냅 (중점 보존, 길이 보존)."""
    snapped: list[WallSegment] = []
    for w in walls:
        a = w.angle_deg
        cx = (w.x1 + w.x2) / 2.0
        cy = (w.y1 + w.y2) / 2.0
        half_len = w.length / 2.0

        # 90° 에 더 가까우면 수직, 아니면 수평
        is_vert = abs(a - 90.0) <= 45.0
        if is_vert:
            # 수직: x 고정, y ±half_len
            snapped.append(WallSegment(cx, cy - half_len, cx, cy + half_len))
        else:
            # 수평: y 고정, x ±half_len
            snapped.append(WallSegment(cx - half_len, cy, cx + half_len, cy))
    logger.info(f"  축 스냅 완료: {len(snapped)}개")
    return snapped


def _merge_collinear(
    walls: list[WallSegment],
    angle_tol_deg: float = 3.0,
    distance_tol_px: float = 12.0,
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
            angle_diff = abs(w.angle_deg - wj.angle_deg)
            # 0°와 180° 근처도 같은 방향으로 처리
            if angle_diff > angle_tol_deg and abs(180.0 - angle_diff) > angle_tol_deg:
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

    logger.info(f"  병합 후 라인: {len(merged)}")
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
        min_wall_length_mm: 이 길이보다 짧은 검출은 노이즈로 취급. 기본 800mm.
        assume_orthogonal: True이면 수평/수직 ±8° 이내 선만 유지.
        dark_threshold: 벽으로 간주할 최대 밝기 (0-255). 기본 70.
    """

    def __init__(
        self,
        pixels_per_mm: float = 0.1,
        min_wall_length_mm: float = 800.0,
        assume_orthogonal: bool = True,
        dark_threshold: int = 70,
    ):
        self.pixels_per_mm = pixels_per_mm
        self.min_wall_length_mm = min_wall_length_mm
        self.assume_orthogonal = assume_orthogonal
        self.dark_threshold = dark_threshold

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
        logger.info(
            f"도면 파싱 시작: {w}x{h}, ppm={self.pixels_per_mm}, "
            f"dark_threshold={self.dark_threshold}, "
            f"assume_orthogonal={self.assume_orthogonal}"
        )

        # Step 1: 이진화 (엄격한 dark threshold + Otsu fallback)
        binary = _binarize(gray, dark_threshold=self.dark_threshold)

        # Step 2 & 3: Opening 7×7 → Closing 7×7
        binary = _morphology_clean(binary)

        # Step 4: Connected-component 필터
        binary = _filter_components(binary, min_area_ratio=0.0008)

        # Step 5: Contour 기반 벽 추출 (NEW)
        # — 외곽선을 따서 다각형으로 단순화한 뒤, 각 변을 벽으로 변환
        # — Hough 보다 누락이 적고, 자동으로 연결성이 보장됨
        walls_px = _detect_walls_via_contours(binary)

        if not walls_px:
            logger.info("  검출된 벽 없음 → 빈 결과 반환")
            return ParseResult(walls=[], image_size=(w, h), pixels_per_mm=self.pixels_per_mm)

        # Step 6: 직교 방향 필터 (assume_orthogonal=True일 때)
        if self.assume_orthogonal:
            walls_px = _orientation_filter(walls_px, angle_tol_deg=8.0)

        # Step 7: 축 스냅
        if self.assume_orthogonal and walls_px:
            walls_px = _snap_to_axis(walls_px)

        # Step 8: 병합 (엄격한 tolerance)
        walls_px = _merge_collinear(walls_px, angle_tol_deg=3.0, distance_tol_px=12.0)

        # Step 9: 최소 길이 필터 (mm 변환 전, px 단위)
        min_length_px_final = max(40, int(self.min_wall_length_mm * self.pixels_per_mm))
        before_len_filter = len(walls_px)
        walls_px = [w for w in walls_px if w.length >= min_length_px_final]
        logger.info(
            f"  최소 길이 필터 ({self.min_wall_length_mm}mm = {min_length_px_final}px): "
            f"{before_len_filter} → {len(walls_px)}"
        )

        # Step 10: mm 변환
        walls_mm = _to_mm(walls_px, self.pixels_per_mm)
        logger.info(f"  최종 벽 수: {len(walls_mm)}")

        return ParseResult(
            walls=walls_mm,
            image_size=(w, h),
            pixels_per_mm=self.pixels_per_mm,
        )
