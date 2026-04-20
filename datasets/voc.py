from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, root: str, image_set: str, classes: List[str], transforms=None, allow_empty=False):
        self.root = root
        self.image_set = image_set
        self.transforms = transforms
        self.allow_empty = allow_empty

        self.class_to_idx = {c: i + 1 for i, c in enumerate(classes)}
        split_file = os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")
        with open(split_file, "r", encoding="utf-8") as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def _parse_xml(self, xml_path: str):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes, labels, iscrowd = [], [], []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in self.class_to_idx:
                continue
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text) - 1
            ymin = float(bbox.find("ymin").text) - 1
            xmax = float(bbox.find("xmax").text) - 1
            ymax = float(bbox.find("ymax").text) - 1
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name])
            difficult = obj.find("difficult")
            iscrowd.append(int(difficult.text) if difficult is not None else 0)

        if len(boxes) == 0 and self.allow_empty:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else torch.zeros((0,), dtype=torch.float32)
        return boxes, labels, area, iscrowd

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_path = os.path.join(self.root, "JPEGImages", f"{image_id}.jpg")
        xml_path = os.path.join(self.root, "Annotations", f"{image_id}.xml")

        image = Image.open(img_path).convert("RGB")
        boxes, labels, area, iscrowd = self._parse_xml(xml_path)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
            "file_name": image_id,
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
