import os

import torch


def save_checkpoint(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)
