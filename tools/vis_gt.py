from __future__ import annotations

import argparse
import os

from datasets.builder import build_dataset
from utils.config import load_config
from utils.visualize import draw_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_dir", default="vis_gt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # For visualization, keep geometry transforms but disable normalization for readable images.
    cfg["INPUT"]["NORMALIZE"] = False
    ds = build_dataset(cfg, split=args.split)
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(min(args.num_samples, len(ds))):
        image_t, target = ds[i]
        from torchvision.transforms.functional import to_pil_image

        img = to_pil_image(image_t)
        labels = target.get("labels", [])
        scores = [1.0] * len(labels)
        vis = draw_predictions(
            img,
            target.get("boxes", []).tolist() if hasattr(target.get("boxes", []), "tolist") else [],
            labels.tolist() if hasattr(labels, "tolist") else [],
            scores,
            cfg["DATASET"]["CLASSES"],
        )
        vis.save(os.path.join(args.output_dir, f"{i:04d}.jpg"))


if __name__ == "__main__":
    main()
