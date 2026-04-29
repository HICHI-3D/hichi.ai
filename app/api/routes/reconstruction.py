"""3D 재구성 라우트.

API:
    POST   /api/reconstruction/jobs              - 사진 여러 장 업로드, 잡 생성
    GET    /api/reconstruction/jobs/{id}         - 상태 조회 (status, progress, stage, result)
    GET    /api/reconstruction/jobs/{id}/model.glb - 결과 .glb 다운로드
    DELETE /api/reconstruction/jobs/{id}         - 취소
    GET    /api/reconstruction/jobs              - 목록 (디버그/관리용)
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from app.core.config import settings
from app.jobs import JobNotFoundError, job_queue
from app.jobs.reconstruction import pipeline

router = APIRouter()

UPLOAD_TMP_ROOT = settings.work_dir / "uploads"


def _save_uploads(files: list[UploadFile]) -> list[Path]:
    """업로드된 이미지를 임시 폴더에 저장하고 경로 리스트 반환."""
    upload_id = uuid.uuid4().hex
    target = UPLOAD_TMP_ROOT / upload_id
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, f in enumerate(files):
        if not f.content_type or not f.content_type.startswith("image/"):
            continue
        suffix = Path(f.filename or "").suffix.lower() or ".jpg"
        out = target / f"{i:03d}{suffix}"
        out.write_bytes(f.file.read())
        paths.append(out)
    return paths


@router.post("/jobs")
async def create_job(files: list[UploadFile] = File(...)):
    """사진 여러 장을 받아 재구성 잡을 생성하고 즉시 job_id 반환."""
    if len(files) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"최소 5장 이상의 사진이 필요합니다 (받은 수: {len(files)}). 권장 20장 이상.",
        )

    image_paths = _save_uploads(files)
    if len(image_paths) < 5:
        raise HTTPException(status_code=400, detail="유효한 이미지가 부족합니다")

    state = job_queue.create(metadata={"image_count": len(image_paths)})

    async def runner(s, prog):
        return await pipeline.run(s, prog, image_paths=image_paths)

    job_queue.schedule(state.id, runner)
    logger.info(f"재구성 잡 생성: {state.id} ({len(image_paths)}장)")
    return state.to_dict()


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        state = job_queue.get(job_id)
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail="잡을 찾을 수 없습니다") from e
    return state.to_dict()


@router.get("/jobs")
async def list_jobs():
    return {"jobs": [s.to_dict() for s in job_queue.list()]}


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    try:
        state = job_queue.get(job_id)
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail="잡을 찾을 수 없습니다") from e
    job_queue.cancel(job_id)
    return state.to_dict()


@router.get("/jobs/{job_id}/model.glb")
async def download_model(job_id: str):
    try:
        state = job_queue.get(job_id)
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail="잡을 찾을 수 없습니다") from e

    if state.status.value != "completed":
        raise HTTPException(status_code=409, detail=f"아직 준비 안 됨 (status={state.status.value})")

    model_path = state.result.get("model_path")
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="모델 파일이 없습니다")

    return FileResponse(
        path=model_path,
        media_type="model/gltf-binary",
        filename=f"furniture_{job_id}.glb",
    )


@router.post("/build")
async def build_legacy(files: list[UploadFile] = File(...)):
    """이전 라우트 호환용. 새 코드는 POST /jobs 사용."""
    return await create_job(files)


# 워밍업: COLMAP 가용성을 부팅 직후 한번 체크하고 싶으면 main lifespan에서 호출
def check_dependencies() -> dict[str, bool]:
    return {
        "colmap": shutil.which(settings.colmap_bin) is not None,
        "open3d": _can_import("open3d"),
        "trimesh": _can_import("trimesh"),
    }


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False
