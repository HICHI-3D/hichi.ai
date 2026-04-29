"""AI 추론 API 라우터."""

from fastapi import APIRouter

from app.api.routes import floor_plans, furniture, reconstruction

router = APIRouter()
router.include_router(floor_plans.router, prefix="/floor-plans", tags=["floor-plans"])
router.include_router(furniture.router, prefix="/furniture", tags=["furniture"])
router.include_router(reconstruction.router, prefix="/reconstruction", tags=["reconstruction"])
