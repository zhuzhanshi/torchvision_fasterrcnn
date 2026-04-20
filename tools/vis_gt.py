from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torchvision.transforms.functional import to_pil_image

from datasets.builder import build_dataset
from utils.config import load_config
from utils.file_io import dump_json, ensure_dir
from utils.visualize import draw_predictions


def parse_args():
    parser = argparse.ArgumentParser("Visualize ground-truth boxes from dataset split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/vis_gt")
    parser.add_argument("--disable-train-aug", action="store_true", help="Disable random train augmentation for stable visualization.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg["INPUT"]["NORMALIZE"] = False
    if args.disable_train_aug:
        cfg["AUG"]["TRAIN"]["ENABLE"] = False

    dataset = build_dataset(cfg, split=args.split)
    ensure_dir(args.output_dir)
    records = []

    n = min(int(args.max_samples), len(dataset))
    for i in range(n):
        image_tensor, target = dataset[i]
        image = to_pil_image(image_tensor)

        labels = target["labels"].tolist()
        boxes = target["boxes"].tolist()
        scores = [1.0] * len(labels)
        vis = draw_predictions(
            image=image,
            boxes=boxes,
            labels=labels,
            scores=scores,
            class_names=cfg["DATASET"]["CLASSES"],
            draw_label=True,
            draw_score=False,
            line_thickness=2,
        )
        file_name = target.get("file_name", f"{i:06d}")
        out_name = f"{i:04d}_{str(file_name).replace('/', '__')}.jpg"
        out_path = os.path.join(args.output_dir, out_name)
        vis.save(out_path)
        records.append({"index": i, "file_name": str(file_name), "output": out_path, "num_boxes": len(boxes)})

    dump_json({"split": args.split, "num_saved": len(records), "samples": records}, os.path.join(args.output_dir, "meta.json"))
    print(f"Saved {len(records)} GT visualization images to: {args.output_dir}")


if __name__ == "__main__":
    main()
