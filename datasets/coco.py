from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(self, root: str, ann_file: str, classes: List[str], transforms=None, allow_empty=False):
        self.root = root
        self.coco = COCO(ann_file)
        self.transforms = transforms
        self.allow_empty = allow_empty

        cats = self.coco.loadCats(self.coco.getCatIds())
        cat_name_to_id = {c["name"]: c["id"] for c in cats}
        self.cat_id_to_label = {}
        for i, cls_name in enumerate(classes):
            if cls_name in cat_name_to_id:
                self.cat_id_to_label[cat_name_to_id[cls_name]] = i + 1

        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(self.root, file_name)
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, area, iscrowd = [], [], [], []
        for ann in anns:
            cid = ann["category_id"]
            if cid not in self.cat_id_to_label:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[cid])
            area.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0 and self.allow_empty:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = torch.tensor(area, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
            "file_name": file_name,
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
