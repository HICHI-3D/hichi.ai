"""COLMAP 기반 가구 3D 재구성 파이프라인.

흐름 (CPU 가능 경로):
    1. 이미지를 workspace/images/ 에 저장
    2. COLMAP feature_extractor (CPU, SIFT)
    3. COLMAP exhaustive_matcher
    4. COLMAP mapper → sparse model
    5. sparse points3D 를 .ply 로 변환
    6. Open3D Poisson 또는 BPA 메쉬 생성
    7. trimesh 로 .glb 익스포트

OpenMVS 가 PATH 에 있으면 dense 경로로 업그레이드 (TODO).
"""

from __future__ import annotations

import asyncio
import shutil
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from app.core.config import settings
from app.jobs.queue import JobState

ProgressFn = Callable[[float, str], None]


class ReconstructionError(Exception):
    pass


# ──────────────────────────────────────────────────────────────────
# COLMAP 호출 헬퍼
# ──────────────────────────────────────────────────────────────────


async def _run(cmd: list[str], cwd: Path | None = None) -> None:
    """subprocess.run async 래퍼. stdout/stderr는 로그로."""
    logger.debug(f"$ {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        out = (stderr or stdout or b"").decode(errors="replace")[-2000:]
        raise ReconstructionError(
            f"명령 실패 (code={proc.returncode}): {' '.join(cmd[:2])}\n{out}"
        )


# ──────────────────────────────────────────────────────────────────
# COLMAP 산출물 파싱 (points3D.bin → ply)
# ──────────────────────────────────────────────────────────────────


def _read_next_bytes(f, n: int, fmt: str) -> tuple:
    data = f.read(n)
    return struct.unpack(fmt, data)


def colmap_points3d_to_ply(points_bin: Path, ply_out: Path) -> int:
    """COLMAP sparse/0/points3D.bin → ASCII PLY (xyz + rgb).

    포맷 참고: https://colmap.github.io/format.html
    """
    points: list[tuple[float, float, float, int, int, int]] = []
    with open(points_bin, "rb") as f:
        (num_points,) = _read_next_bytes(f, 8, "<Q")
        for _ in range(num_points):
            point_id = _read_next_bytes(f, 8, "<Q")[0]  # noqa: F841
            x, y, z = _read_next_bytes(f, 24, "<ddd")
            r, g, b = _read_next_bytes(f, 3, "<BBB")
            error = _read_next_bytes(f, 8, "<d")[0]  # noqa: F841
            (track_len,) = _read_next_bytes(f, 8, "<Q")
            f.read(track_len * 8)  # IMAGE_ID, POINT2D_IDX 페어 건너뜀
            points.append((x, y, z, r, g, b))

    ply_out.parent.mkdir(parents=True, exist_ok=True)
    with open(ply_out, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in points:
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    return len(points)


# ──────────────────────────────────────────────────────────────────
# 메쉬 생성 (Open3D Poisson)
# ──────────────────────────────────────────────────────────────────


def points_to_mesh_glb(ply_path: Path, glb_out: Path, voxel_size: float = 0.005) -> dict:
    """포인트 클라우드 → 메쉬 → .glb.

    Open3D 미설치 시 ImportError 던짐 (호출부에서 안내).
    """
    try:
        import numpy as np
        import open3d as o3d
        import trimesh
    except ImportError as e:
        raise ReconstructionError(
            f"메쉬 생성 의존성 미설치: {e}. 'uv sync --extra reconstruction' 실행 필요."
        ) from e

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) < 100:
        raise ReconstructionError(
            f"sparse 포인트가 너무 적습니다 ({len(pcd.points)}개). 사진 매칭이 잘 안 된 듯."
        )

    # 다운샘플 + 노멀 추정
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(20)

    # Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, scale=1.1, linear_fit=False
    )
    # 밀도 낮은 영역 잘라내기 (떠다니는 메쉬 제거)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(densities < threshold)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    # trimesh로 .glb 익스포트
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    if colors is not None and colors.size:
        # open3d는 0~1 float, trimesh는 0~255 uint8
        colors = (colors * 255).clip(0, 255).astype(np.uint8)
    tri = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, process=False)

    glb_out.parent.mkdir(parents=True, exist_ok=True)
    tri.export(glb_out, file_type="glb")
    return {
        "vertices": int(len(verts)),
        "faces": int(len(faces)),
        "size_bytes": glb_out.stat().st_size,
    }


# ──────────────────────────────────────────────────────────────────
# 파이프라인
# ──────────────────────────────────────────────────────────────────


class FurnitureReconstructionPipeline:
    """가구 사진들 → .glb 변환.

    Args:
        colmap_bin: colmap 실행 파일 (PATH 의존 가능)
        work_root: 잡별 작업 디렉터리 부모
    """

    def __init__(self, colmap_bin: str | None = None, work_root: Path | None = None):
        self.colmap_bin = colmap_bin or settings.colmap_bin
        self.work_root = work_root or settings.work_dir / "reconstruction"

    def check_colmap(self) -> bool:
        return shutil.which(self.colmap_bin) is not None

    async def run(
        self,
        state: JobState,
        progress: ProgressFn,
        image_paths: list[Path],
    ) -> dict[str, Any]:
        if not self.check_colmap():
            raise ReconstructionError(
                f"COLMAP을 찾을 수 없습니다 ({self.colmap_bin}). "
                "macOS: brew install colmap"
            )

        work = self.work_root / state.id
        images_dir = work / "images"
        sparse_dir = work / "sparse"
        db_path = work / "database.db"
        ply_path = work / "points3D.ply"
        glb_path = work / "model.glb"

        images_dir.mkdir(parents=True, exist_ok=True)

        # ─── 0) 입력 이미지 복사 ──────────────────────────────────
        progress(0.05, "preparing")
        for i, src in enumerate(image_paths):
            dst = images_dir / f"img_{i:03d}{src.suffix.lower() or '.jpg'}"
            shutil.copy(src, dst)

        # ─── 1) feature_extractor ────────────────────────────────
        # NOTE: --SiftExtraction.use_gpu / --SiftMatching.use_gpu 는 macOS brew 빌드
        # (CUDA 없이 컴파일된 COLMAP) 에서 옵션 자체가 존재하지 않아 파싱 에러를 낸다.
        # 해당 빌드는 어차피 CPU 만 쓰므로 플래그를 생략하고 COLMAP 기본 동작에 맡긴다.
        # CUDA 빌드를 쓰면서 GPU 사용을 강제로 막아야 하는 환경이 생기면 그때 다시 추가.
        progress(0.10, "feature_extraction")
        await _run([
            self.colmap_bin, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
        ])

        # ─── 2) exhaustive_matcher ───────────────────────────────
        progress(0.30, "matching")
        await _run([
            self.colmap_bin, "exhaustive_matcher",
            "--database_path", str(db_path),
        ])

        # ─── 3) mapper (SfM) ─────────────────────────────────────
        progress(0.55, "sfm_mapping")
        sparse_dir.mkdir(parents=True, exist_ok=True)
        await _run([
            self.colmap_bin, "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
        ])

        # COLMAP은 sparse/0, sparse/1, ... 식으로 저장
        models = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])
        if not models:
            raise ReconstructionError(
                "SfM 실패: 사진이 너무 적거나 매칭이 안 됐을 수 있어요. "
                "여러 각도에서 20장 이상 권장."
            )
        sfm_model = models[0]

        # ─── 4) points3D → PLY ───────────────────────────────────
        progress(0.75, "exporting_pointcloud")
        n_points = colmap_points3d_to_ply(sfm_model / "points3D.bin", ply_path)
        logger.info(f"포인트 {n_points}개 → {ply_path}")

        # ─── 5) Poisson 메쉬 + glTF ──────────────────────────────
        progress(0.85, "meshing")
        mesh_info = await asyncio.to_thread(points_to_mesh_glb, ply_path, glb_path)

        progress(1.0, "completed")
        return {
            "model_path": str(glb_path),
            "model_url": f"/api/reconstruction/jobs/{state.id}/model.glb",
            "image_count": len(image_paths),
            "points": n_points,
            **mesh_info,
        }


# 싱글톤
pipeline = FurnitureReconstructionPipeline()
