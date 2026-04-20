from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_set: str,
        classes: List[str],
        transforms=None,
        filter_empty_gt: bool = False,
        ignore_difficult: bool = False,
        min_box_area: float = 0.0,
        check_files: bool = False,
    ):
        self.root = root
        self.image_set = image_set
        self.transforms = transforms
        self.filter_empty_gt = filter_empty_gt
        self.ignore_difficult = ignore_difficult
        self.min_box_area = float(min_box_area)
        self.check_files = check_files

        self.class_to_idx = {c: i + 1 for i, c in enumerate(classes)}
        split_file = os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"VOC split file not found: {split_file}")

        with open(split_file, "r", encoding="utf-8") as f:
            all_ids = [line.strip() for line in f if line.strip()]

        if self.filter_empty_gt:
            self.ids = [i for i in all_ids if self._has_valid_annotation(i)]
        else:
            self.ids = all_ids

        if len(self.ids) == 0:
            raise RuntimeError(f"No valid VOC samples for split={image_set} under {root}")

    def __len__(self):
        return len(self.ids)

    def _image_path(self, image_id: str) -> str:
        return os.path.join(self.root, "JPEGImages", f"{image_id}.jpg")

    def _xml_path(self, image_id: str) -> str:
        return os.path.join(self.root, "Annotations", f"{image_id}.xml")

    def _safe_parse_xml(self, xml_path: str):
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"VOC annotation xml not found: {xml_path}")
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as e:
            raise ValueError(f"Broken VOC xml file: {xml_path}, err={e}") from e
        return tree.getroot()

    def _extract_boxes_labels(self, root_xml, image_wh: Tuple[int, int] | None = None):
        boxes, labels, iscrowd = [], [], []
        img_w, img_h = image_wh if image_wh is not None else (None, None)

        for obj in root_xml.findall("object"):
            name_node = obj.find("name")
            if name_node is None or not name_node.text:
                continue
            class_name = name_node.text.strip()
            if class_name not in self.class_to_idx:
                continue

            difficult_node = obj.find("difficult")
            difficult = int(difficult_node.text) if difficult_node is not None and difficult_node.text is not None else 0
            if self.ignore_difficult and difficult == 1:
                continue

            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            try:
                xmin = float(bbox.find("xmin").text) - 1
                ymin = float(bbox.find("ymin").text) - 1
                xmax = float(bbox.find("xmax").text) - 1
                ymax = float(bbox.find("ymax").text) - 1
            except Exception:
                continue

            if img_w is not None and img_h is not None:
                xmin = max(0.0, min(xmin, img_w - 1))
                xmax = max(0.0, min(xmax, img_w - 1))
                ymin = max(0.0, min(ymin, img_h - 1))
                ymax = max(0.0, min(ymax, img_h - 1))

            w = xmax - xmin
            h = ymax - ymin
            if w <= 0 or h <= 0 or (w * h) < self.min_box_area:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[class_name])
            iscrowd.append(difficult)

        return boxes, labels, iscrowd

    def _has_valid_annotation(self, image_id: str) -> bool:
        xml_root = self._safe_parse_xml(self._xml_path(image_id))
        boxes, _, _ = self._extract_boxes_labels(xml_root)
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
        image_id = self.ids[idx]
        img_path = self._image_path(image_id)
        xml_path = self._xml_path(image_id)

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"VOC image file not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        xml_root = self._safe_parse_xml(xml_path)
        boxes, labels, iscrowd = self._extract_boxes_labels(xml_root, image_wh=(img_w, img_h))

        if len(boxes) == 0:
            target = self._empty_target(idx)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
            area_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
            target = {
                "boxes": boxes_t,
                "labels": labels_t,
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": area_t,
                "iscrowd": iscrowd_t,
            }

        target["file_name"] = image_id

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
