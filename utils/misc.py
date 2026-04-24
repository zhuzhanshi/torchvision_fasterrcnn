from __future__ import annotations

import datetime as dt
import math


def now_str():
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


def to_device(images, targets, device):
    images = [img.to(device) for img in images]
    moved = []
    for t in targets:
        moved.append({k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()})
    return images, moved


def is_finite_number(x: float) -> bool:
    return math.isfinite(float(x))
