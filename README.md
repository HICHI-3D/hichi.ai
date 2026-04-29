# hichi.ai

하이치 AI 워크스페이스. **모델 학습/실험**과 **추론 서버**가 같은 프로젝트에 들어있습니다.

- **스택**: Python 3.11 / PyTorch / YOLO + SAM / OpenCV / FastAPI / uv
- **역할**:
  - 가구 인식 (YOLO + SAM)
  - 가구 3D 재구성 (COLMAP + OpenMVS + MeshLab CLI)
  - 도면 분석 (OpenCV ± 학습 모델)
  - `hichi.server`가 호출하는 추론 API 제공

## 사전 준비

uv가 없다면 설치:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3D 재구성용 외부 CLI 도구는 별도 설치 필요 (재구성 기능을 실제로 쓸 때만):

```bash
# macOS
brew install colmap

# OpenMVS, MeshLab은 깃헙 릴리즈/공식 사이트에서 받기
# https://github.com/cdcseacave/openMVS/releases
# https://www.meshlab.net/#download
```

## 시작하기

```bash
# 1) 의존성 설치 (.venv 자동 생성)
uv sync

# 2) (선택) 도면 처리 / 3D 재구성용 추가 패키지
uv sync --extra floorplan
uv sync --extra reconstruction

# 3) 환경변수
cp .env.example .env

# 4) 환경 점검 (PyTorch, CUDA/MPS, CLI 도구 확인)
uv run python scripts/check_env.py

# 5) 모델 가중치 다운로드 (YOLOv8n + SAM ViT-B 약 380MB)
uv run python scripts/download_weights.py

# 6) 추론 서버 실행
uv run uvicorn main:app --reload --port 8001
```

서버 실행 후:

- API 문서: <http://localhost:8001/docs>

## 폴더 구조

```
hichi.ai/
├── main.py                  # 추론 서버 진입점
├── pyproject.toml
├── app/
│   ├── main.py              # FastAPI 앱
│   ├── core/config.py
│   ├── api/routes/          # 추론 엔드포인트
│   │   ├── floor_plans.py
│   │   ├── furniture.py
│   │   └── reconstruction.py
│   └── inference/           # 실제 ML 로직
│       ├── device.py        # MPS/CUDA/CPU 자동 감지
│       ├── floor_plan.py
│       ├── furniture.py
│       └── reconstruction.py
├── notebooks/               # 실험용 Jupyter
│   ├── 01_furniture_yolo_baseline.ipynb
│   └── 02_floor_plan_opencv.ipynb
├── scripts/                 # 학습/유틸 스크립트
│   ├── check_env.py
│   └── download_weights.py
├── data/                    # 데이터셋 (gitignored)
├── models/                  # 가중치 (gitignored)
└── tests/
```

## 디바이스

- **macOS Apple Silicon**: PyTorch가 자동으로 MPS 백엔드 사용
- **CUDA GPU**: `uv add "torch>=2.5.0" --index-url https://download.pytorch.org/whl/cu121` 로 재설치
- **Colab**: 노트북에서 `!pip install ultralytics segment-anything` 후 사용

## 주요 학습/실험 흐름

1. **데이터 수집**: `data/furniture/` 에 가구 사진 모으기
2. **YOLO fine-tuning**: `scripts/train_yolo.py` (작성 예정)
3. **SAM 평가**: `notebooks/01_furniture_yolo_baseline.ipynb`
4. **도면 OpenCV 베이스라인**: `notebooks/02_floor_plan_opencv.ipynb`
5. **3D 재구성 파이프라인**: `app/inference/reconstruction.py`

## 자주 쓰는 명령

```bash
uv run pytest                    # 테스트
uv run jupyter lab               # 노트북
uv run ruff check .              # 린트
uv add <패키지명>                # 의존성 추가
```
