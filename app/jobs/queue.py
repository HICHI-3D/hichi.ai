"""인메모리 잡 큐.

- asyncio.create_task로 백그라운드 실행
- 잡 상태/진행률은 dict에 보관
- 단순한 졸업작품 데모 용도. 프로덕션이면 redis/celery/arq.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobNotFoundError(Exception):
    pass


@dataclass
class JobState:
    id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0.0 ~ 1.0
    stage: str = "queued"
    error: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "progress": round(self.progress, 3),
            "stage": self.stage,
            "error": self.error,
            "result": self.result,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# 진행 콜백 시그니처
ProgressFn = Callable[[float, str], None]


class JobQueue:
    """잡 단일 인스턴스 컨테이너."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def create(self, metadata: dict[str, Any] | None = None) -> JobState:
        job_id = uuid.uuid4().hex
        state = JobState(id=job_id, metadata=metadata or {})
        self._jobs[job_id] = state
        return state

    def get(self, job_id: str) -> JobState:
        try:
            return self._jobs[job_id]
        except KeyError as e:
            raise JobNotFoundError(job_id) from e

    def list(self) -> list[JobState]:
        return list(self._jobs.values())

    def schedule(
        self,
        job_id: str,
        runner: Callable[[JobState, ProgressFn], Awaitable[dict[str, Any]]],
    ) -> None:
        """잡을 백그라운드 태스크로 등록."""
        state = self.get(job_id)

        def progress(p: float, stage: str) -> None:
            state.progress = max(0.0, min(1.0, p))
            state.stage = stage
            state.updated_at = datetime.utcnow()

        async def _wrapper() -> None:
            state.status = JobStatus.RUNNING
            state.stage = "starting"
            state.updated_at = datetime.utcnow()
            try:
                result = await runner(state, progress)
                state.result = result
                state.status = JobStatus.COMPLETED
                state.progress = 1.0
                state.stage = "completed"
                logger.info(f"잡 완료: {job_id}")
            except asyncio.CancelledError:
                state.status = JobStatus.CANCELLED
                state.stage = "cancelled"
                logger.info(f"잡 취소: {job_id}")
                raise
            except Exception as e:  # noqa: BLE001
                state.status = JobStatus.FAILED
                state.error = str(e)
                state.stage = "failed"
                logger.exception(f"잡 실패: {job_id}")
            finally:
                state.updated_at = datetime.utcnow()

        task = asyncio.create_task(_wrapper(), name=f"job:{job_id}")
        self._tasks[job_id] = task

    def cancel(self, job_id: str) -> None:
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()


# 싱글톤
job_queue = JobQueue()
