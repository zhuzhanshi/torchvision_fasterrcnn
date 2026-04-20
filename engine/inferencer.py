from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, normalize

from utils.file_io import ensure_dir
from utils.visualize import draw_predictions


class Inferencer:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger

    def _gather_inputs(self, path):
        p = Path(path)
        if p.is_file():
            return [str(p)]
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        files = []
        for e in exts:
            files.extend(glob.glob(str(p / e)))
        return sorted(files)

    @torch.no_grad()
    def run(self, model, device, output_dir):
        model.eval()
        model.roi_heads.score_thresh = self.cfg["INFER"]["SCORE_THRESH"]
        model.roi_heads.nms_thresh = self.cfg["INFER"]["NMS_THRESH"]
        model.roi_heads.detections_per_img = self.cfg["INFER"]["MAX_DETS"]

        inp = self.cfg["INFER"]["INPUT_PATH"]
        files = self._gather_inputs(inp)
        class_names = self.cfg["DATASET"]["CLASSES"]

        vis_dir = os.path.join(output_dir, "vis")
        js_dir = os.path.join(output_dir, "json")
        txt_dir = os.path.join(output_dir, "txt")
        ensure_dir(vis_dir)
        ensure_dir(js_dir)
        ensure_dir(txt_dir)

        mean = self.cfg["INPUT"]["MEAN"]
        std = self.cfg["INPUT"]["STD"]
        class_filter = set(self.cfg["INFER"].get("CLASS_FILTER", []))

        for fp in files:
            image = Image.open(fp).convert("RGB")
            t = normalize(to_tensor(image), mean=mean, std=std).to(device)
            output = model([t])[0]

            boxes = output["boxes"].detach().cpu()
            labels = output["labels"].detach().cpu()
            scores = output["scores"].detach().cpu()

            keep = scores >= self.cfg["INFER"]["SCORE_THRESH"]
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
            if class_filter:
                keep_cf = torch.tensor([(int(l.item()) in class_filter) for l in labels], dtype=torch.bool)
                boxes, labels, scores = boxes[keep_cf], labels[keep_cf], scores[keep_cf]

            stem = Path(fp).stem
            records = []
            for b, l, s in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
                records.append({"box": b, "label": int(l), "score": float(s)})

            if self.cfg["INFER"].get("SAVE_JSON", True):
                with open(os.path.join(js_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
            if self.cfg["INFER"].get("SAVE_TXT", False):
                with open(os.path.join(txt_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(f"{r['label']} {r['score']:.6f} {' '.join(map(str, r['box']))}\n")
            if self.cfg["INFER"].get("SAVE_VIS", True):
                vis = draw_predictions(image, boxes.tolist(), labels.tolist(), scores.tolist(), class_names)
                vis.save(os.path.join(vis_dir, f"{stem}.jpg"))

        if self.logger:
            self.logger.info(f"Inference finished. Processed {len(files)} images.")
