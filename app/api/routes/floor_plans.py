"""도면 처리 라우트 (실제 OpenCV 파서에 연결)."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

from app.inference.floor_plan import FloorPlanParser

router = APIRouter()

# 파서 인스턴스는 싱글톤으로 (lifespan에서 초기화해도 OK, 가벼우니 모듈 레벨)
_parser = FloorPlanParser(pixels_per_mm=0.1, min_wall_length_mm=300.0)


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

    parser = _parser
    if pixels_per_mm and pixels_per_mm > 0:
        parser = FloorPlanParser(pixels_per_mm=pixels_per_mm)

    try:
        result = parser.parse_bytes(image_bytes)
    except ValueError as e:
        logger.error(f"파싱 실패: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e

    payload = result.to_dict()
    payload["filename"] = file.filename
    payload["status"] = "ok"
    return payload
