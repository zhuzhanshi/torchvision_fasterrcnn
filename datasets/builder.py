from __future__ import annotations

import os

from torch.utils.data import DataLoader

from .coco import COCODataset
from .collate import detection_collate_fn
from .sampler import build_sampler
from .transforms import build_transforms
from .voc import VOCDataset


def _get_data_root(cfg):
    ds_cfg = cfg["DATASET"]
    return ds_cfg.get("ROOT", ds_cfg.get("DATA_ROOT", "data"))


def build_dataset(cfg, split="train"):
    ds_cfg = cfg["DATASET"]
    data_root = _get_data_root(cfg)
    t = build_transforms(cfg, is_train=(split == "train"))
    ds_type = ds_cfg["TYPE"].lower()

    if ds_type == "voc":
        dataset = VOCDataset(
            root=data_root,
            image_set=ds_cfg[f"{split.upper()}_SPLIT"],
            classes=ds_cfg["CLASSES"],
            transforms=t,
            allow_empty=ds_cfg.get("ALLOW_EMPTY", False),
        )
    elif ds_type == "coco":
        image_root = os.path.join(data_root, split)
        ann_file = os.path.join(data_root, "annotations", f"instances_{split}.json")
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


def _batch_size_for_split(cfg, split):
    dl = cfg["DATALOADER"]
    if split == "train":
        return dl.get("TRAIN_BATCH_SIZE", dl.get("BATCH_SIZE", 1))
    if split == "val":
        return dl.get("VAL_BATCH_SIZE", dl.get("BATCH_SIZE", 1))
    return dl.get("TEST_BATCH_SIZE", dl.get("BATCH_SIZE", 1))


def build_dataloader(cfg, split="train"):
    dataset = build_dataset(cfg, split)
    shuffle = cfg["DATALOADER"].get("SHUFFLE", True) if split == "train" else False
    sampler = build_sampler(dataset, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=_batch_size_for_split(cfg, split),
        sampler=sampler,
        num_workers=cfg["DATALOADER"].get("NUM_WORKERS", cfg["RUNTIME"].get("NUM_WORKERS", 4)),
        pin_memory=cfg["DATALOADER"].get("PIN_MEMORY", cfg["RUNTIME"].get("PIN_MEMORY", True)),
        drop_last=cfg["DATALOADER"].get("DROP_LAST", False) if split == "train" else False,
        collate_fn=detection_collate_fn,
    )
    return dataset, loader


def build_dataloaders(cfg, mode="train"):
    loaders = {}
    datasets = {}
    if mode == "train":
        datasets["train"], loaders["train"] = build_dataloader(cfg, split="train")
        if cfg["EVAL"].get("ENABLE", cfg["EVAL"].get("ENABLED", True)):
            datasets["val"], loaders["val"] = build_dataloader(cfg, split="val")
    elif mode == "test":
        datasets["test"], loaders["test"] = build_dataloader(cfg, split="test")
    else:
        return {}, {}
    return datasets, loaders
