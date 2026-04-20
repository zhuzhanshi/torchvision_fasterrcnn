from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.file_io import dump_json
from utils.metrics import save_eval_outputs
from utils.misc import to_device


class Evaluator:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger

    @torch.no_grad()
    def evaluate(self, model, dataloader, device, output_dir=None):
        model.eval()
        start = time.time()
        predictions = []

        for images, targets in dataloader:
            images, targets = to_device(images, targets, device)
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                img_id = int(tgt["image_id"].item())
                boxes = out["boxes"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                labels = out["labels"].detach().cpu().numpy()
                for b, s, l in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = b.tolist()
                    predictions.append(
                        {
                            "image_id": img_id,
                            "category_id": int(l),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(s),
                        }
                    )

        metrics, per_class_ap = self._compute_metrics(dataloader.dataset, predictions)
        metrics["eval_time"] = time.time() - start

        if output_dir:
            save_eval_outputs(metrics, per_class_ap, output_dir)
            dump_json(predictions, os.path.join(output_dir, "predictions.json"))

        if self.logger:
            self.logger.info(f"Eval metrics: {metrics}")

        return metrics, per_class_ap

    def _compute_metrics(self, dataset, predictions):
        if hasattr(dataset, "coco"):
            return self._compute_coco_metrics(dataset, predictions)
        # VOC fallback: TODO implement VOC-style mAP; currently prediction-count placeholder
        metrics = {
            "mAP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AR": 0.0,
            "num_predictions": len(predictions),
            "note": "VOC metrics TODO: use COCO conversion for exact AP/AR.",
        }
        return metrics, []

    def _compute_coco_metrics(self, dataset, predictions):
        coco_gt = dataset.coco
        if len(predictions) == 0:
            metrics = {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}
            return metrics, []

        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        metrics = {
            "mAP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "AR": float(stats[8]),
        }

        per_class = []
        precisions = coco_eval.eval["precision"]
        cat_ids = coco_gt.getCatIds()
        cats = coco_gt.loadCats(cat_ids)
        for cid, cat in zip(cat_ids, cats):
            idx = cat_ids.index(cid)
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = float(np.mean(precision)) if precision.size else 0.0
            per_class.append({"category_id": cid, "category_name": cat["name"], "AP": ap})

        return metrics, per_class
