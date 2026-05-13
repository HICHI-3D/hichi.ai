"""도면 처리 라우트.

흐름:
    1) GEMINI_API_KEY가 설정돼 있으면 Gemini Vision 파서 우선 시도
    2) Vision이 실패하면 OpenCV(HoughLinesP) 파서로 fallback
    두 파서 모두 같은 ParseResult 형태를 반환하므로 응답 shape은 동일하다.
"""

import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

from app.core.config import settings
from app.inference.floor_plan import FloorPlanParser

router = APIRouter()


# ─── 파서 싱글톤 ──────────────────────────────────────────────────────

# OpenCV(fallback) 파서
_opencv_parser = FloorPlanParser(pixels_per_mm=0.1, min_wall_length_mm=800.0)

# Vision 파서 상태 (진단용)
_vision_parser_error: str | None = None
_last_vision_failure: str | None = None  # 마지막 런타임 파싱 실패


def _build_vision_parser():
    """Vision 파서를 만든다. 키가 없거나 패키지가 없으면 None 반환."""
    global _vision_parser_error

    if not settings.gemini_api_key:
        _vision_parser_error = "GEMINI_API_KEY 미설정 (.env 파일 확인 필요)"
        logger.warning("⚠️  GEMINI_API_KEY 미설정 — Vision 파서 비활성, OpenCV만 사용")
        return None
    try:
        from app.inference.floor_plan_vision import VisionFloorPlanParser

        parser = VisionFloorPlanParser(
            api_key=settings.gemini_api_key,
            model=settings.vision_parser_model,
            pixels_per_mm=0.1,
        )
        _vision_parser_error = None
        logger.info(
            f"✅ Gemini Vision 파서 활성: model={settings.vision_parser_model}, "
            f"key=AIza...{settings.gemini_api_key[-4:]}"
        )
        return parser
    except Exception as e:  # noqa: BLE001 — 부팅 단계 호환성 보호
        _vision_parser_error = f"{type(e).__name__}: {e}"
        logger.warning(f"⚠️  Vision 파서 초기화 실패: {e} — OpenCV로만 동작")
        return None


_vision_parser = _build_vision_parser()


# ─── 라우트 ───────────────────────────────────────────────────────────


@router.post("/parse")
async def parse_floor_plan(
    file: UploadFile = File(...),
    pixels_per_mm: float | None = None,
):
    """도면 이미지를 받아 벽/방/개구부 정보를 JSON으로 반환.

    Args:
        file: PNG/JPG 도면 이미지
        pixels_per_mm: 이미지 1mm가 몇 픽셀인지 (선택). 미지정 시 기본값 사용.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 허용됩니다")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="빈 파일입니다")

    logger.info("===== 도면 파싱 시작 =====")
    logger.info(
        f"Vision 파서: {'활성 (Gemini 사용)' if _vision_parser else '비활성 (OpenCV 사용 예정)'}, "
        f"file={file.filename}, size={len(image_bytes)}"
    )

    used_parser_name = "opencv"
    result = None

    # 1) Vision 우선
    if _vision_parser is not None:
        ppm = pixels_per_mm if (pixels_per_mm and pixels_per_mm > 0) else 0.1
        try:
            # ppm이 다르면 인스턴스 prop만 잠시 바꿔 사용
            original_ppm = _vision_parser.pixels_per_mm
            _vision_parser.pixels_per_mm = ppm
            try:
                result = _vision_parser.parse_bytes(
                    image_bytes,
                    mime=file.content_type or "image/png",
                )
            finally:
                _vision_parser.pixels_per_mm = original_ppm
            used_parser_name = "vision"
        except Exception as e:  # noqa: BLE001 — fallback이 목적
            global _last_vision_failure
            err_str = str(e)
            _last_vision_failure = f"{type(e).__name__}: {err_str[:300]}"
            # 429 RESOURCE_EXHAUSTED 는 흔하니 따로 안내
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                logger.warning(
                    "⚠️  Gemini 할당량 초과 — OpenCV로 대체합니다. "
                    "잠시 후 재시도하거나, .env에 더 관대한 모델 지정: "
                    "VISION_PARSER_MODEL=gemini-2.0-flash (분당 15회/하루 1500회)"
                )
            else:
                logger.warning(f"⚠️  Vision 파싱 실패 → OpenCV fallback: {e}")
            result = None

    # 2) Fallback: OpenCV
    if result is None:
        parser = _opencv_parser
        if pixels_per_mm and pixels_per_mm > 0:
            parser = FloorPlanParser(pixels_per_mm=pixels_per_mm)
        try:
            result = parser.parse_bytes(image_bytes)
        except ValueError as e:
            logger.error(f"파싱 실패 (OpenCV): {e}")
            raise HTTPException(status_code=400, detail=str(e)) from e

    payload = result.to_dict()
    payload["filename"] = file.filename
    payload["status"] = "ok"
    payload["parser"] = used_parser_name
    logger.info(
        f"===== 도면 파싱 완료: parser={used_parser_name}, "
        f"walls={len(result.walls)} ====="
    )
    return payload


# ─── 진단 ─────────────────────────────────────────────────────────────


@router.get("/diagnostics")
async def floor_plan_diagnostics():
    """Vision 파서가 왜 동작/미동작하는지 한눈에 보여주는 진단 endpoint.

    브라우저로 http://localhost:8001/api/floor-plans/diagnostics 열면 JSON 응답.
    """
    cwd = Path.cwd()
    env_file = cwd / ".env"

    # google-genai 설치 여부 직접 확인
    try:
        import google.genai  # noqa: F401

        genai_installed = True
        genai_error = None
    except Exception as e:  # noqa: BLE001
        genai_installed = False
        genai_error = f"{type(e).__name__}: {e}"

    key_preview = None
    if settings.gemini_api_key:
        k = settings.gemini_api_key
        key_preview = f"{k[:6]}...{k[-4:]}" if len(k) > 12 else "(짧음, 키가 잘림?)"

    # 환경변수 직접 확인 (pydantic-settings를 우회)
    raw_env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("gemini_api_key")

    next_steps: list[str] = []
    if not settings.gemini_api_key:
        next_steps.append(
            ".env 파일을 확인하세요. 위치: "
            f"{env_file} (존재: {env_file.exists()})"
        )
        next_steps.append(
            "파일 내용에 정확히 'GEMINI_API_KEY=AIza...' 한 줄 있어야 함. "
            "공백, 따옴표 없이."
        )
        if raw_env_key:
            next_steps.append(
                "⚠️  os.environ에는 키가 있는데 settings에는 없음. "
                "서버를 .env가 있는 디렉토리에서 실행했는지 확인."
            )
        else:
            next_steps.append(
                "현재 디렉토리에서 서버를 실행했는지 확인: "
                f"`cd {cwd} && uvicorn app.main:app --reload --port 8001` "
                "(또는 `uv run` 사용)"
            )
    if not genai_installed:
        next_steps.append(
            f"google-genai 패키지 미설치 → 설치하세요: `uv sync` 또는 "
            f"`pip install google-genai`. 에러: {genai_error}"
        )
    if _vision_parser_error and settings.gemini_api_key:
        next_steps.append(
            f"파서 초기화 에러: {_vision_parser_error}. "
            "API 키가 유효한지 https://aistudio.google.com/apikey 에서 확인."
        )
    if _vision_parser and _last_vision_failure:
        next_steps.append(
            f"가장 최근 Vision 호출이 런타임에 실패함: {_last_vision_failure}. "
            "키 권한/할당량/네트워크 확인."
        )
    if not next_steps:
        next_steps.append("✅ 모든 항목 정상. 도면을 업로드해보세요.")

    return {
        "vision_parser_initialized": _vision_parser is not None,
        "vision_parser_model": settings.vision_parser_model,
        "vision_parser_init_error": _vision_parser_error,
        "last_vision_runtime_failure": _last_vision_failure,
        "gemini_api_key_set_in_settings": bool(settings.gemini_api_key),
        "gemini_api_key_preview": key_preview,
        "gemini_api_key_in_os_environ": bool(raw_env_key),
        "google_genai_installed": genai_installed,
        "google_genai_import_error": genai_error,
        "cwd": str(cwd),
        "env_file_path": str(env_file),
        "env_file_exists": env_file.exists(),
        "next_steps": next_steps,
    }
