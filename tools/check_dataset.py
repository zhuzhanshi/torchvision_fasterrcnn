from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.builder import build_dataset
from utils.config import load_config
from utils.file_io import dump_json


def parse_args():
    parser = argparse.ArgumentParser("Check detection dataset integrity")
    parser.add_argument("--config", required=True, help="Path to config python file.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=0, help="0 means check all samples.")
    parser.add_argument("--output-json", default="", help="Optional path to save report json.")
    return parser.parse_args()


def _image_hw(image):
    if isinstance(image, torch.Tensor):
        return int(image.shape[-2]), int(image.shape[-1])
    if hasattr(image, "size"):  # PIL
        w, h = image.size
        return int(h), int(w)
    raise TypeError(f"Unsupported image type in dataset sample: {type(image)}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset = build_dataset(cfg, split=args.split)

    num_samples = len(dataset) if args.max_samples <= 0 else min(len(dataset), int(args.max_samples))
    num_classes = int(cfg["DATASET"]["NUM_CLASSES"])
    class_names = cfg["DATASET"]["CLASSES"]

    report = {
        "split": args.split,
        "dataset_len": len(dataset),
        "checked_samples": num_samples,
        "missing_fields": 0,
        "empty_targets": 0,
        "invalid_bbox": 0,
        "bbox_out_of_bounds": 0,
        "invalid_label": 0,
        "invalid_area": 0,
        "class_count": {},
    }
    cls_counter = Counter()

    for i in range(num_samples):
        image, target = dataset[i]
        required = ["boxes", "labels", "image_id", "area", "iscrowd"]
        if any(k not in target for k in required):
            report["missing_fields"] += 1
            continue

        h, w = _image_hw(image)
        boxes = target["boxes"]
        labels = target["labels"]
        areas = target["area"]

        if len(boxes) == 0:
            report["empty_targets"] += 1
            continue

        for box, label, area in zip(boxes, labels, areas):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            label_i = int(label)
            area_f = float(area)

            if x2 <= x1 or y2 <= y1:
                report["invalid_bbox"] += 1
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                report["bbox_out_of_bounds"] += 1
            if not (1 <= label_i <= num_classes):
                report["invalid_label"] += 1
            if area_f < 0:
                report["invalid_area"] += 1
            if 1 <= label_i <= num_classes:
                cls_counter[label_i] += 1

    report["class_count"] = {class_names[k - 1] if k - 1 < len(class_names) else str(k): v for k, v in sorted(cls_counter.items())}

    print("=== Dataset Check Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    if args.output_json:
        dump_json(report, args.output_json)
        print(f"Saved report json to: {args.output_json}")


if __name__ == "__main__":
    main()
