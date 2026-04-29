"""가구 인식 라우트.

여러 각도의 가구 사진 → YOLO 검출 + SAM 분할 → 크기/형태 추정.
"""

from fastapi import APIRouter, File, UploadFile

router = APIRouter()


@router.post("/detect")
async def detect_furniture(file: UploadFile = File(...)):
    """단일 이미지에서 가구 검출 (YOLO).

    TODO: app.inference.furniture.detect 호출
    """
    return {
        "filename": file.filename,
        "detections": [],
        "status": "not_implemented",
    }


@router.post("/segment")
async def segment_furniture(file: UploadFile = File(...)):
    """가구 마스크 분할 (SAM).

    TODO: app.inference.furniture.segment 호출
    """
    return {
        "filename": file.filename,
        "masks": [],
        "status": "not_implemented",
    }
