"""디바이스 자동 감지 (mps / cuda / cpu)."""

import torch
from loguru import logger


def get_device(prefer: str = "auto") -> torch.device:
    """사용 가능한 가장 빠른 디바이스를 반환.

    Args:
        prefer: "auto" | "mps" | "cuda" | "cpu"
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        logger.info("디바이스: Apple MPS")
        return torch.device("mps")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        logger.info(f"디바이스: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    logger.info("디바이스: CPU")
    return torch.device("cpu")
