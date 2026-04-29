"""AI 추론 서버 FastAPI 앱."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api import router as api_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"하이치 AI 서버 시작 - device={settings.device}, port={settings.app_port}")
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    # TODO: 여기서 모델 가중치 미리 로드 (lazy load 대신)
    yield
    logger.info("하이치 AI 서버 종료")


app = FastAPI(
    title="하이치 AI 추론 서버",
    description="가구 인식, 3D 재구성, 도면 분석",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/", tags=["health"])
async def root():
    return {
        "app": "hichi-ai",
        "version": "0.1.0",
        "device": settings.device,
        "status": "ok",
    }


@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy"}
