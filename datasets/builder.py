from __future__ import annotations

import os
from typing import Tuple

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .coco import COCODataset
from .collate import detection_collate_fn
from .sampler import build_sampler
from .transforms import build_transforms
from .voc import VOCDataset


def _validate_dataset_config(cfg):
    ds_cfg = cfg["DATASET"]
    classes = ds_cfg.get("CLASSES", [])
    if not isinstance(classes, (list, tuple)) or len(classes) == 0:
        raise ValueError("DATASET.CLASSES must be a non-empty list of foreground classes.")
    if ds_cfg.get("NUM_CLASSES", len(classes)) != len(classes):
        raise ValueError(
            f"DATASET.NUM_CLASSES ({ds_cfg.get('NUM_CLASSES')}) must equal len(DATASET.CLASSES) ({len(classes)})."
        )


def _get_data_root(cfg):
    ds_cfg = cfg["DATASET"]
    return ds_cfg.get("ROOT", ds_cfg.get("DATA_ROOT", "data"))


def _resolve_split_name(cfg, split: str) -> str:
    key = f"{split.upper()}_SPLIT"
    split_name = cfg["DATASET"].get(key)
    if not split_name:
        raise ValueError(f"Missing DATASET.{key} for split={split}")
    return split_name


def build_dataset(cfg, split="train"):
    _validate_dataset_config(cfg)

    ds_cfg = cfg["DATASET"]
    data_root = _get_data_root(cfg)
    split_name = _resolve_split_name(cfg, split)
    t = build_transforms(cfg, is_train=(split == "train"))
    ds_type = ds_cfg["TYPE"].lower()

    filter_empty = bool(ds_cfg.get("FILTER_EMPTY_GT", False)) if split == "train" else False
    min_box_area = float(ds_cfg.get("MIN_BOX_AREA", 0.0))
    check_files = bool(ds_cfg.get("CHECK_DATASET", False))

    if ds_type == "voc":
        dataset = VOCDataset(
            root=data_root,
            image_set=split_name,
            classes=ds_cfg["CLASSES"],
            transforms=t,
            filter_empty_gt=filter_empty,
            ignore_difficult=bool(ds_cfg.get("IGNORE_DIFFICULT", False)),
            min_box_area=min_box_area,
            check_files=check_files,
        )
    elif ds_type == "coco":
        image_root = os.path.join(data_root, split_name)
        ann_file = os.path.join(data_root, "annotations", f"instances_{split_name}.json")
        dataset = COCODataset(
            root=image_root,
            ann_file=ann_file,
            classes=ds_cfg["CLASSES"],
            transforms=t,
            filter_empty_gt=filter_empty,
            min_box_area=min_box_area,
            check_files=check_files,
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
    runtime = cfg.get("RUNTIME", {})
    distributed = bool(runtime.get("DISTRIBUTED", False))
    shuffle = cfg["DATALOADER"].get("SHUFFLE", True) if split == "train" else False
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(runtime.get("WORLD_SIZE", 1)),
            rank=int(runtime.get("RANK", 0)),
            shuffle=shuffle,
            drop_last=bool(cfg["DATALOADER"].get("DROP_LAST", False) if split == "train" else False),
        )
    else:
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
