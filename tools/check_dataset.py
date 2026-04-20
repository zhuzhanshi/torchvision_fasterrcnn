from __future__ import annotations

import argparse
from collections import Counter

from datasets.builder import build_dataset
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser("Check detection dataset integrity")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = build_dataset(cfg, split=args.split)

    num_empty = 0
    invalid_bbox = 0
    out_of_bounds = 0
    invalid_label = 0
    missing_fields = 0
    cls_count = Counter()

    for i in range(len(ds)):
        image, target = ds[i]
        if any(k not in target for k in ["boxes", "labels", "image_id", "area", "iscrowd"]):
            missing_fields += 1
            continue

        h, w = image.shape[-2:]
        boxes = target["boxes"]
        labels = target["labels"]

        if len(boxes) == 0:
            num_empty += 1
            continue

        for b, l in zip(boxes, labels):
            x1, y1, x2, y2 = b.tolist()
            if x2 <= x1 or y2 <= y1:
                invalid_bbox += 1
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                out_of_bounds += 1
            if not (1 <= int(l) <= cfg["DATASET"]["NUM_CLASSES"]):
                invalid_label += 1
            cls_count[int(l)] += 1

    print("=== Dataset Check Report ===")
    print(f"split={args.split}, total={len(ds)}")
    print(f"missing_required_target_fields={missing_fields}")
    print(f"num_empty={num_empty}")
    print(f"invalid_bbox={invalid_bbox}")
    print(f"out_of_bounds={out_of_bounds}")
    print(f"invalid_label={invalid_label}")
    print("class_count=", dict(cls_count))


if __name__ == "__main__":
    main()
