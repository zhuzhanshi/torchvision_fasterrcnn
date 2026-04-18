from __future__ import annotations

import os

from torch.utils.data import DataLoader

from .coco import COCODataset
from .collate import detection_collate_fn
from .sampler import build_sampler
from .transforms import build_transforms
from .voc import VOCDataset


def build_dataset(cfg, split="train"):
    ds_cfg = cfg["DATASET"]
    t = build_transforms(cfg, is_train=(split == "train"))
    ds_type = ds_cfg["TYPE"].lower()

    if ds_type == "voc":
        dataset = VOCDataset(
            root=ds_cfg["DATA_ROOT"],
            image_set=ds_cfg[f"{split.upper()}_SPLIT"],
            classes=ds_cfg["CLASSES"],
            transforms=t,
            allow_empty=ds_cfg.get("ALLOW_EMPTY", False),
        )
    elif ds_type == "coco":
        image_root = os.path.join(ds_cfg["DATA_ROOT"], split)
        ann_file = os.path.join(ds_cfg["DATA_ROOT"], "annotations", f"instances_{split}.json")
        dataset = COCODataset(
            root=image_root,
            ann_file=ann_file,
            classes=ds_cfg["CLASSES"],
            transforms=t,
            allow_empty=ds_cfg.get("ALLOW_EMPTY", False),
        )
    else:
        raise ValueError(f"Unsupported dataset type: {ds_type}")
    return dataset


def build_dataloader(cfg, split="train"):
    dataset = build_dataset(cfg, split)
    shuffle = cfg["DATALOADER"]["SHUFFLE"] if split == "train" else False
    sampler = build_sampler(dataset, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=cfg["DATALOADER"]["BATCH_SIZE"],
        sampler=sampler,
        num_workers=cfg["DATALOADER"]["NUM_WORKERS"],
        pin_memory=cfg["DATALOADER"]["PIN_MEMORY"],
        drop_last=cfg["DATALOADER"].get("DROP_LAST", False) if split == "train" else False,
        collate_fn=detection_collate_fn,
    )
    return dataset, loader


def build_dataloaders(cfg, mode="train"):
    loaders = {}
    datasets = {}
    if mode == "train":
        datasets["train"], loaders["train"] = build_dataloader(cfg, split="train")
        if cfg["EVAL"].get("ENABLED", True):
            datasets["val"], loaders["val"] = build_dataloader(cfg, split="val")
    elif mode == "test":
        datasets["test"], loaders["test"] = build_dataloader(cfg, split="test")
    else:
        return {}, {}
    return datasets, loaders
