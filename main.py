"""하이치 AI 추론 서버 진입점.

실행:
    uv run uvicorn main:app --reload --port 8001
"""

from app.main import app

__all__ = ["app"]
