"""3D 재구성 파이프라인 (COLMAP → OpenMVS → MeshLab).

각 단계가 외부 CLI 호출이라 subprocess로 묶는다.
오래 걸리는 작업이므로 실제로는 jobs 큐(예: arq, celery)로 빼는 게 좋다.
"""

import shutil
import subprocess
from pathlib import Path

from loguru import logger


class ReconstructionPipeline:
    """다중 사진 → 3D 메쉬 파이프라인. (스켈레톤)

    TODO 단계:
    1. COLMAP feature_extractor / exhaustive_matcher / mapper → sparse model
    2. COLMAP image_undistorter → dense workspace
    3. OpenMVS InterfaceCOLMAP → DensifyPointCloud → ReconstructMesh → RefineMesh → TextureMesh
    4. MeshLab CLI 또는 trimesh로 cleanup, glb 변환
    """

    def __init__(self, work_dir: Path, colmap_bin: str, openmvs_bin_dir: str):
        self.work_dir = work_dir
        self.colmap_bin = colmap_bin
        self.openmvs_bin_dir = openmvs_bin_dir

    def check_tools(self) -> dict[str, bool]:
        """필수 CLI 도구가 시스템에 설치되어 있는지 점검."""
        return {
            "colmap": shutil.which(self.colmap_bin) is not None,
            "DensifyPointCloud": (Path(self.openmvs_bin_dir) / "DensifyPointCloud").exists(),
        }

    def run(self, image_dir: Path, output_dir: Path) -> Path:
        logger.info(f"재구성 시작: {image_dir} → {output_dir}")
        # TODO: subprocess.run([self.colmap_bin, "feature_extractor", ...])
        raise NotImplementedError("재구성 파이프라인 구현 예정")
