from __future__ import annotations

import os
from collections import OrderedDict

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def save_on_master(fn, *args, **kwargs):
    if is_main_process():
        return fn(*args, **kwargs)
    return None


def all_gather(data):
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, data)
    return gather_list


def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        device = None
        for k in sorted(input_dict.keys()):
            names.append(k)
            v = input_dict[k]
            if not isinstance(v, torch.Tensor):
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                v = torch.tensor(float(v), dtype=torch.float32, device=device)
            elif device is None:
                device = v.device
            values.append(v.detach().float())
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced = OrderedDict({k: v for k, v in zip(names, values)})
    return reduced


def _select_backend(device_str: str, configured: str = "") -> str:
    if configured:
        return str(configured).lower()
    d = str(device_str).lower()
    if d.startswith("npu"):
        return "hccl"
    if d.startswith("cuda"):
        return "nccl"
    return "gloo"


def init_distributed_mode(cfg):
    runtime = cfg["RUNTIME"]
    if str(runtime.get("MODE", "train")).lower() == "infer":
        runtime["DISTRIBUTED"] = False
        runtime["WORLD_SIZE"] = 1
        runtime["RANK"] = 0
        runtime["LOCAL_RANK"] = 0
        return False
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dist_requested = bool(runtime.get("DISTRIBUTED", False)) or env_world_size > 1
    runtime["DISTRIBUTED"] = dist_requested

    if not dist_requested:
        runtime["WORLD_SIZE"] = 1
        runtime["RANK"] = 0
        runtime["LOCAL_RANK"] = 0
        return False

    rank = int(os.environ.get("RANK", runtime.get("RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", runtime.get("WORLD_SIZE", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", runtime.get("LOCAL_RANK", 0)))

    runtime["RANK"] = rank
    runtime["WORLD_SIZE"] = world_size
    runtime["LOCAL_RANK"] = local_rank
    runtime["DIST_URL"] = runtime.get("DIST_URL", "env://")
    runtime["DIST_BACKEND"] = _select_backend(runtime.get("DEVICE", "cpu"), runtime.get("DIST_BACKEND", "")).lower()
    backend = runtime["DIST_BACKEND"]
    device_str = str(runtime.get("DEVICE", "cpu")).lower()
    ascend_visible = os.environ.get("ASCEND_VISIBLE_DEVICES", "")

    # IMPORTANT for NPU/HCCL:
    # 1) import torch_npu first
    # 2) set local device before init_process_group
    if backend == "hccl" or device_str.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except Exception as e:
            raise ImportError(
                "Distributed backend is HCCL/NPU, but torch_npu is not available. "
                "Please install torch_npu compatible with your CANN/PyTorch."
            ) from e
        if not hasattr(torch, "npu"):
            raise RuntimeError("torch_npu imported but torch.npu backend is unavailable.")
        npu_count = int(torch.npu.device_count())
        if local_rank < 0 or local_rank >= npu_count:
            raise RuntimeError(
                f"Invalid LOCAL_RANK={local_rank} for NPU device_count={npu_count}. "
                f"ASCEND_VISIBLE_DEVICES={ascend_visible!r}. Please check torchrun rank mapping."
            )
        torch.npu.set_device(local_rank)
        runtime["DEVICE"] = f"npu:{local_rank}"
    elif backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed backend is NCCL but torch.cuda.is_available() is False.")
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"Invalid LOCAL_RANK={local_rank} for CUDA device_count={torch.cuda.device_count()}."
            )
        torch.cuda.set_device(local_rank)
        runtime["DEVICE"] = f"cuda:{local_rank}"

    print(
        "[dist] init requested "
        f"rank={rank} local_rank={local_rank} world_size={world_size} "
        f"device={runtime.get('DEVICE')} backend={backend} ASCEND_VISIBLE_DEVICES={ascend_visible!r}"
    )

    if is_dist_avail_and_initialized():
        return True
    dist.init_process_group(
        backend=backend,
        init_method=runtime["DIST_URL"],
        world_size=world_size,
        rank=rank,
    )
    return True
