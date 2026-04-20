import platform

import torch
import torchvision


def collect_env_info(device):
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
