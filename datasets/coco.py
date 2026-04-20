from __future__ import annotations

import os
from typing import List

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(
        self,
        root: str,
        ann_file: str,
        classes: List[str],
        transforms=None,
        filter_empty_gt: bool = False,
        min_box_area: float = 0.0,
        check_files: bool = False,
    ):
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"COCO image root not found: {root}")

        self.root = root
        self.coco = COCO(ann_file)
        self.transforms = transforms
        self.filter_empty_gt = filter_empty_gt
        self.min_box_area = float(min_box_area)
        self.check_files = check_files

        self.classes = list(classes)
        cats = self.coco.loadCats(self.coco.getCatIds())
        cat_name_to_id = {c["name"]: c["id"] for c in cats}

        self.cat_id_to_label = {}
        for i, cls_name in enumerate(self.classes):
            if cls_name in cat_name_to_id:
                self.cat_id_to_label[cat_name_to_id[cls_name]] = i + 1

        if len(self.cat_id_to_label) == 0:
            raise ValueError("None of DATASET.CLASSES were found in COCO categories.")

        all_ids = list(sorted(self.coco.imgs.keys()))
        if self.filter_empty_gt:
            self.ids = [img_id for img_id in all_ids if self._has_valid_annotation(img_id)]
        else:
            self.ids = all_ids

        if len(self.ids) == 0:
            raise RuntimeError(f"No valid COCO samples found under root={root}, ann={ann_file}")

    def __len__(self):
        return len(self.ids)

    def _image_path(self, file_name: str):
        return os.path.join(self.root, file_name)

    def _build_targets(self, anns, img_w, img_h):
        boxes, labels, areas, iscrowd = [], [], [], []

        for ann in anns:
            cid = ann.get("category_id", None)
            if cid not in self.cat_id_to_label:
                continue

            bbox = ann.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            if (w * h) < self.min_box_area:
                continue

            x1 = max(0.0, min(float(x), img_w - 1))
            y1 = max(0.0, min(float(y), img_h - 1))
            x2 = max(0.0, min(float(x + w), img_w - 1))
            y2 = max(0.0, min(float(y + h), img_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_label[cid])
            areas.append(float(ann.get("area", (x2 - x1) * (y2 - y1))))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        return boxes, labels, areas, iscrowd

    def _has_valid_annotation(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs([img_id])[0]
        boxes, _, _, _ = self._build_targets(anns, img_info["width"], img_info["height"])
        return len(boxes) > 0

    @staticmethod
    def _empty_target(image_id):
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
        }

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]
        img_path = self._image_path(file_name)

        if self.check_files and not os.path.isfile(img_path):
            raise FileNotFoundError(f"COCO image file not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels, areas, crowds = self._build_targets(anns, img_w, img_h)

        if len(boxes) == 0:
            target = self._empty_target(img_id)
        else:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([img_id], dtype=torch.int64),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.tensor(crowds, dtype=torch.int64),
            }
        target["file_name"] = file_name

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
