"""헬스체크 테스트."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["app"] == "hichi-ai"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
