import platform

import torch
import torchvision

from .file_io import dump_text


def collect_env_info(device, runtime_info: dict | None = None):
    gpu_names = []
    npu_available = False
    npu_device_count = 0
    npu_names = []
    if hasattr(torch, "npu"):
        try:
            npu_available = bool(torch.npu.is_available()) if hasattr(torch.npu, "is_available") else True
            npu_device_count = int(torch.npu.device_count()) if hasattr(torch.npu, "device_count") else 0
            if hasattr(torch.npu, "get_device_name"):
                npu_names = [torch.npu.get_device_name(i) for i in range(npu_device_count)]
        except Exception:
            npu_available = False
            npu_device_count = 0
            npu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_names": gpu_names,
        "npu_available": npu_available,
        "npu_device_count": npu_device_count,
        "npu_names": npu_names,
        "runtime_device_type": str(device),
    }
    if runtime_info:
        info.update(
            {
                "distributed_enabled": bool(runtime_info.get("DISTRIBUTED", False)),
                "dist_backend": runtime_info.get("DIST_BACKEND", ""),
                "world_size": int(runtime_info.get("WORLD_SIZE", 1)),
                "rank": int(runtime_info.get("RANK", 0)),
                "local_rank": int(runtime_info.get("LOCAL_RANK", 0)),
            }
        )
    return info


def format_env_info(info: dict) -> str:
    lines = [f"{k}: {v}" for k, v in info.items()]
    return "\n".join(lines) + "\n"


def save_env_info(info: dict, path: str):
    dump_text(format_env_info(info), path)
