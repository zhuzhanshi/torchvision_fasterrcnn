from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        if isinstance(image, Image.Image):
            image = F.to_tensor(image)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        return F.normalize(image, self.mean, self.std), target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        _, w = image.shape[-2:]
        image = torch.flip(image, dims=[2])
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        target["boxes"] = boxes
        return image, target


class RandomVerticalFlip:
    def __init__(self, p=0.0):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        h, _ = image.shape[-2:]
        image = torch.flip(image, dims=[1])
        boxes = target["boxes"].clone()
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        target["boxes"] = boxes
        return image, target


class ColorJitter:
    def __init__(self, brightness, contrast, saturation, hue):
        import torchvision.transforms as T

        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image, target):
        return self.jitter(image), target


class RandomResize:
    def __init__(self, scales: List[int], max_size: int = 1333):
        self.scales = scales
        self.max_size = max_size

    def __call__(self, image, target):
        h, w = image.shape[-2:]
        short = random.choice(self.scales)
        min_side = min(h, w)
        max_side = max(h, w)
        scale = short / min_side
        if max_side * scale > self.max_size:
            scale = self.max_size / max_side

        new_h = int(h * scale)
        new_w = int(w * scale)
        image = F.resize(image, [new_h, new_w])

        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] *= float(new_w) / w
        boxes[:, [1, 3]] *= float(new_h) / h
        target["boxes"] = boxes
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return image, target


def build_transforms(cfg: Dict, is_train: bool = True) -> Callable:
    t = [ToTensor()]
    if is_train:
        rr_cfg = cfg["AUG"]["RANDOM_RESIZE"]
        if rr_cfg.get("ENABLED", False):
            t.append(RandomResize(rr_cfg.get("SCALES", [cfg["INPUT"]["MIN_SIZE"]]), rr_cfg.get("MAX_SIZE", cfg["INPUT"]["MAX_SIZE"])))
        t.append(RandomHorizontalFlip(cfg["AUG"].get("HFLIP_PROB", 0.5)))
        t.append(RandomVerticalFlip(cfg["AUG"].get("VFLIP_PROB", 0.0)))
        cj = cfg["AUG"].get("COLOR_JITTER", {})
        if cj.get("ENABLED", False):
            t.append(ColorJitter(cj["BRIGHTNESS"], cj["CONTRAST"], cj["SATURATION"], cj["HUE"]))
    t.append(Normalize(cfg["INPUT"]["MEAN"], cfg["INPUT"]["STD"]))
    return Compose(t)
