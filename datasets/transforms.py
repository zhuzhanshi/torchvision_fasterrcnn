from __future__ import annotations

import random
import warnings
from typing import Callable, Dict, List

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


def _ensure_tensor_types(target):
    target["boxes"] = target["boxes"].to(dtype=torch.float32)
    target["labels"] = target["labels"].to(dtype=torch.int64)
    target["image_id"] = target["image_id"].to(dtype=torch.int64)
    target["area"] = target["area"].to(dtype=torch.float32)
    target["iscrowd"] = target["iscrowd"].to(dtype=torch.int64)
    return target


def _sanitize_boxes(target, h: int, w: int):
    boxes = target["boxes"]
    if boxes.numel() == 0:
        target["area"] = torch.zeros((0,), dtype=torch.float32)
        return target

    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, max(w - 1, 0))
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, max(h - 1, 0))

    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    target["boxes"] = boxes[keep]
    target["labels"] = target["labels"][keep]
    target["iscrowd"] = target["iscrowd"][keep]
    target["area"] = (target["boxes"][:, 2] - target["boxes"][:, 0]) * (target["boxes"][:, 3] - target["boxes"][:, 1])
    return target


class ToTensor:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def __call__(self, image, target):
        if self.enabled and isinstance(image, Image.Image):
            image = F.to_tensor(image)
        return image, _ensure_tensor_types(target)


class Normalize:
    def __init__(self, mean, std, enabled=True):
        self.mean = mean
        self.std = std
        self.enabled = enabled

    def __call__(self, image, target):
        if self.enabled:
            image = F.normalize(image, self.mean, self.std)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        _, w = image.shape[-2:]
        image = torch.flip(image, dims=[2])
        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] = (w - 1) - boxes[:, [2, 0]]
        target["boxes"] = boxes
        _sanitize_boxes(target, image.shape[-2], image.shape[-1])
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
        if boxes.numel() > 0:
            boxes[:, [1, 3]] = (h - 1) - boxes[:, [3, 1]]
        target["boxes"] = boxes
        _sanitize_boxes(target, image.shape[-2], image.shape[-1])
        return image, target


class ColorJitter:
    def __init__(self, brightness, contrast, saturation, hue):
        import torchvision.transforms as T

        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image, target):
        return self.jitter(image), target


class ResizeByShortSide:
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def _compute(self, h, w):
        min_side = min(h, w)
        max_side = max(h, w)
        scale = self.min_size / float(min_side)
        if max_side * scale > self.max_size:
            scale = self.max_size / float(max_side)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        return new_h, new_w

    def __call__(self, image, target):
        h, w = image.shape[-2:]
        new_h, new_w = self._compute(h, w)
        if new_h == h and new_w == w:
            return image, target

        image = F.resize(image, [new_h, new_w])
        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= float(new_w) / float(w)
            boxes[:, [1, 3]] *= float(new_h) / float(h)
        target["boxes"] = boxes
        _sanitize_boxes(target, new_h, new_w)
        return image, target


class RandomResize:
    def __init__(self, scales: List[int], max_size: int = 1333):
        self.scales = scales
        self.max_size = max_size

    def __call__(self, image, target):
        short = random.choice(self.scales)
        return ResizeByShortSide(min_size=short, max_size=self.max_size)(image, target)


class RandomCrop:
    def __init__(self, size):
        self.crop_h, self.crop_w = int(size[0]), int(size[1])

    def __call__(self, image, target):
        h, w = image.shape[-2:]
        if h <= self.crop_h or w <= self.crop_w:
            return image, target

        top = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)
        image = F.crop(image, top, left, self.crop_h, self.crop_w)

        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] -= left
            boxes[:, [1, 3]] -= top
        target["boxes"] = boxes
        _sanitize_boxes(target, self.crop_h, self.crop_w)
        return image, target


class RandomRotate:
    def __init__(self, degrees=0):
        self.degrees = float(degrees)
        if self.degrees > 0:
            warnings.warn(
                "RandomRotate is configured but bbox-safe arbitrary-angle rotation is not implemented. "
                "This transform is currently a no-op.",
                RuntimeWarning,
            )

    def __call__(self, image, target):
        # TODO: implement bbox-safe rotate if needed.
        return image, target


def build_transforms(cfg: Dict, is_train: bool = True) -> Callable:
    """Detection transforms: always image,target -> image,target.

    Strategy:
    - dataset/transforms consume INPUT.* and AUG.*
    - model-side transform params are consumed by model builder from MODEL.*
    """

    input_cfg = cfg["INPUT"]
    aug_cfg = cfg["AUG"]
    train_aug = aug_cfg.get("TRAIN", {})
    test_aug = aug_cfg.get("TEST", {})

    transforms = []

    to_tensor_enabled = bool(input_cfg.get("TO_TENSOR", True))
    transforms.append(ToTensor(enabled=to_tensor_enabled))

    if is_train:
        if train_aug.get("ENABLE", True):
            rr_cfg = train_aug.get("RANDOM_RESIZE", {})
            if rr_cfg.get("ENABLED", False):
                transforms.append(RandomResize(rr_cfg.get("SCALES", [input_cfg["MIN_SIZE"]]), rr_cfg.get("MAX_SIZE", input_cfg["MAX_SIZE"])))

            transforms.append(RandomHorizontalFlip(train_aug.get("HFLIP_PROB", 0.0)))
            transforms.append(RandomVerticalFlip(train_aug.get("VFLIP_PROB", 0.0)))

            cj = train_aug.get("COLOR_JITTER", {})
            if cj.get("ENABLED", False):
                transforms.append(ColorJitter(cj["BRIGHTNESS"], cj["CONTRAST"], cj["SATURATION"], cj["HUE"]))

            crop = train_aug.get("RANDOM_CROP", {})
            if crop.get("ENABLED", False):
                transforms.append(RandomCrop(crop.get("SIZE", [input_cfg["MIN_SIZE"], input_cfg["MIN_SIZE"]])))

            rot = train_aug.get("RANDOM_ROTATE", {})
            if rot.get("ENABLED", False):
                transforms.append(RandomRotate(rot.get("DEGREES", 0)))
    else:
        if test_aug.get("ENABLE", False):
            resize_cfg = test_aug.get("RESIZE", {})
            if resize_cfg.get("ENABLED", False):
                transforms.append(
                    ResizeByShortSide(
                        min_size=resize_cfg.get("MIN_SIZE", input_cfg["MIN_SIZE"]),
                        max_size=resize_cfg.get("MAX_SIZE", input_cfg["MAX_SIZE"]),
                    )
                )

    transforms.append(
        Normalize(
            mean=input_cfg.get("IMAGE_MEAN", input_cfg.get("MEAN")),
            std=input_cfg.get("IMAGE_STD", input_cfg.get("STD")),
            enabled=bool(input_cfg.get("NORMALIZE", True)),
        )
    )

    return Compose(transforms)
