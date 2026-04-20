import os

import torch


def save_checkpoint(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, map_location="cpu"):
    if not path:
        raise ValueError("Checkpoint path is empty.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint format invalid (expect dict): {path}")
    return ckpt
