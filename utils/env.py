import platform

import torch
import torchvision

from .file_io import dump_text


def collect_env_info(device):
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_names": gpu_names,
    }


def format_env_info(info: dict) -> str:
    lines = [f"{k}: {v}" for k, v in info.items()]
    return "\n".join(lines) + "\n"


def save_env_info(info: dict, path: str):
    dump_text(format_env_info(info), path)
