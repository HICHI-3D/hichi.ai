"""Google Gemini Vision 기반 도면 파서 (무료 할당량 사용 가능).

OpenCV 기반 `FloorPlanParser`(HoughLinesP)는 나무 바닥 무늬·타일·텍스트를
벽으로 잘못 인식하는 한계가 있어, 이를 보완하기 위해 Gemini 비전 모델을
호출하는 파서를 추가한다.

흐름:
    image bytes → base64 → Gemini API → JSON {walls:[...], image_width, image_height}
    → ParseResult (mm 좌표)

결과 데이터 모양은 `FloorPlanParser`와 동일하므로 라우트는 그대로 응답한다.

무료 할당량 (2026년 기준):
    - gemini-2.5-flash      : 분당 10회, 하루 500회 (기본, 정확도 좋음)
    - gemini-2.0-flash      : 분당 15회, 하루 1,500회
    - gemini-2.0-flash-lite : 분당 30회, 하루 1,500회 (가장 빠름)
    ⚠️  gemini-2.5-pro / gemini-1.5-pro 는 무료 티어 미지원 (429 발생)
    키 발급: https://aistudio.google.com/apikey
"""

from __future__ import annotations

import base64
import json
import re

from loguru import logger

from app.inference.floor_plan import ParseResult, WallSegment, _to_mm

# ─── 프롬프트 ────────────────────────────────────────────────────────


_USER_PROMPT = (
    "You are a precise architectural drawing analyst. "
    "Analyze this Korean apartment floor plan in TWO PHASES.\n\n"
    "═══ PHASE 1 — Trace the building's OUTER outline ═══\n"
    "The entire apartment is enclosed by ONE continuous outer wall (외벽). "
    "Imagine walking along that outer wall clockwise, starting from the top-left corner. "
    "List every corner (꺾이는 지점) as an (x, y) pixel coordinate. The result is a CLOSED "
    "polygon — the last vertex implicitly connects back to the first.\n\n"
    "Rules for outline:\n"
    "- Include EVERY corner. A typical apartment outline has 8-20 vertices.\n"
    "- Korean apartment outlines often have rectangular bumps (발코니, 다용도실 등 외부 돌출) — "
    "include those.\n"
    "- Ignore the elevator/stair shaft if it's drawn as a separate diagram outside the apartment.\n"
    "- Ignore furniture, text labels, the NAVER watermark.\n"
    "- Snap to axis-aligned coordinates when corners are clearly meant to be at 90°.\n\n"
    "═══ PHASE 2 — List the INTERIOR partition walls ═══\n"
    "Inside the outline, walls divide the apartment into rooms (거실, 침실, 욕실, 발코니, "
    "주방, 현관, 다용도실 등). Each interior wall is a straight line segment from (x1,y1) to (x2,y2).\n\n"
    "Rules for interior walls:\n"
    "- A typical 3-bedroom apartment has 10-25 interior wall segments.\n"
    "- Snap to perfectly horizontal or vertical when clearly axis-aligned.\n"
    "- Treat door/window gaps as wall continuations — do NOT break walls at doors.\n"
    "- Ignore floor textures, tile/hatched patterns, furniture, fixtures (toilet, bathtub, sink).\n\n"
    "═══ OUTPUT FORMAT ═══\n"
    "Return STRICT JSON only (no prose, no markdown fences):\n"
    "{\n"
    '  "outline": [[x, y], [x, y], ...],\n'
    '  "interior_walls": [{"x1": int, "y1": int, "x2": int, "y2": int}, ...],\n'
    '  "image_width": int,\n'
    '  "image_height": int\n'
    "}\n\n"
    "Coordinates: pixel positions in the image (top-left origin, +x right, +y down).\n\n"
    "Process the image now and return only the JSON."
)


# ─── JSON 파싱 헬퍼 ──────────────────────────────────────────────────


def _extract_json(text: str) -> dict:
    """응답에서 JSON 객체를 뽑아낸다. ```json ... ``` 코드 펜스도 지원."""
    text = text.strip()

    # 우선 그대로 파싱 시도
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ```json ... ``` 또는 ``` ... ``` 코드 펜스 제거
    fence_match = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # 첫 '{' ~ 마지막 '}' 사이를 추출
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as e:
            raise ValueError(f"응답 JSON 파싱 실패: {e}\n응답 일부: {snippet[:300]}") from e

    raise ValueError(f"응답에 JSON 객체가 없음: {text[:300]}")


def _walls_from_outline(outline_raw: list) -> list[WallSegment]:
    """외곽 다각형 꼭지점 리스트를 연속된 벽 세그먼트들로 분해.

    [[x0,y0], [x1,y1], [x2,y2], ...] → walls = [(0→1), (1→2), ..., (N-1→0)]
    마지막 꼭지점과 첫 꼭지점도 연결(폐곡선)한다.
    """
    pts: list[tuple[float, float]] = []
    for v in outline_raw:
        try:
            if isinstance(v, dict):
                pts.append((float(v["x"]), float(v["y"])))
            else:
                pts.append((float(v[0]), float(v[1])))
        except (KeyError, TypeError, ValueError, IndexError) as e:
            logger.warning(f"  외곽 꼭지점 무시 (형식 오류): {v} – {e}")
            continue

    if len(pts) < 3:
        logger.warning(f"  외곽선이 너무 짧음 ({len(pts)} 꼭지점) — 외곽 무시")
        return []

    walls: list[WallSegment] = []
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        walls.append(WallSegment(x1, y1, x2, y2))
    return walls


def _walls_from_interior(interior_raw: list) -> list[WallSegment]:
    walls: list[WallSegment] = []
    for w in interior_raw:
        if not isinstance(w, dict):
            continue
        try:
            walls.append(
                WallSegment(
                    float(w["x1"]),
                    float(w["y1"]),
                    float(w["x2"]),
                    float(w["y2"]),
                )
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"  내부 벽 무시 (형식 오류): {w} – {e}")
            continue
    return walls


def _walls_from_payload(payload: dict) -> list[WallSegment]:
    """새 포맷(outline + interior_walls) 우선, 구 포맷(walls) fallback."""
    walls: list[WallSegment] = []

    outline = payload.get("outline")
    interior = payload.get("interior_walls")

    if isinstance(outline, list) or isinstance(interior, list):
        # 신규 포맷
        if isinstance(outline, list):
            outline_walls = _walls_from_outline(outline)
            walls.extend(outline_walls)
            logger.info(f"  외곽 다각형: {len(outline)}개 꼭지점 → {len(outline_walls)}개 벽")
        if isinstance(interior, list):
            interior_walls = _walls_from_interior(interior)
            walls.extend(interior_walls)
            logger.info(f"  내부 칸막이: {len(interior_walls)}개 벽")
        return walls

    # 구 포맷: walls 리스트 직접
    walls_raw = payload.get("walls", [])
    if not isinstance(walls_raw, list):
        raise ValueError(
            f"응답에 outline/interior_walls/walls 어느 것도 없음. keys={list(payload.keys())}"
        )
    walls = _walls_from_interior(walls_raw)
    logger.info(f"  구 포맷 walls: {len(walls)}개")
    return walls


# ─── 본체 ────────────────────────────────────────────────────────────


class VisionFloorPlanParser:
    """Google Gemini로 도면을 파싱한다.

    Args:
        api_key: GEMINI_API_KEY. https://aistudio.google.com/apikey 에서 발급.
        model: 사용할 모델. 기본 'gemini-2.0-flash' (무료 할당량 분당 15/일 1500).
        pixels_per_mm: 이미지 1mm당 픽셀. 기본 0.1.
        max_tokens: 응답 최대 토큰. 기본 4096.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        pixels_per_mm: float = 0.1,
        max_tokens: int = 8192,
    ):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY가 필요합니다. .env의 gemini_api_key를 채우세요. "
                "키 발급: https://aistudio.google.com/apikey"
            )

        try:
            from google import genai  # noqa: WPS433 (런타임 의존성)
        except ImportError as e:
            raise ImportError(
                "google-genai 패키지가 필요합니다. `uv add google-genai` 또는 "
                "`pip install google-genai`로 설치하세요."
            ) from e

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.pixels_per_mm = pixels_per_mm
        self.max_tokens = max_tokens

    def parse_bytes(self, image_bytes: bytes, mime: str = "image/png") -> ParseResult:
        if not image_bytes:
            raise ValueError("빈 이미지 바이트")

        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        logger.info(
            f"Gemini Vision 파싱 요청 시작: model={self.model}, "
            f"image_bytes={len(image_bytes)}, mime={mime}"
        )

        try:
            from google.genai import types  # noqa: WPS433

            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type=mime),
                            types.Part.from_text(text=_USER_PROMPT),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=0.1,
                ),
            )
        except Exception as e:
            logger.error(f"Gemini API 호출 실패: {e}")
            raise ValueError(f"Vision API 호출 실패: {e}") from e

        raw_text = (response.text or "").strip()
        if not raw_text:
            raise ValueError("Gemini 응답이 비어 있음")

        logger.debug(f"Gemini 응답 (앞 300자): {raw_text[:300]}")
        # b64 변수는 일부 SDK 버전 호환 대비로 계산만 해뒀음 (실제로는 from_bytes 사용)
        del b64

        payload = _extract_json(raw_text)

        image_width = int(payload.get("image_width") or 0)
        image_height = int(payload.get("image_height") or 0)
        walls_px = _walls_from_payload(payload)

        logger.info(
            f"Gemini 파싱 결과: walls={len(walls_px)}, "
            f"image={image_width}x{image_height}"
        )

        # 픽셀 → mm 변환
        walls_mm = _to_mm(walls_px, self.pixels_per_mm)

        return ParseResult(
            walls=walls_mm,
            image_size=(image_width, image_height),
            pixels_per_mm=self.pixels_per_mm,
        )
