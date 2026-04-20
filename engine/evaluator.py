from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import numpy as np
import torch

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception:  # pragma: no cover - import guard for optional dependency
    COCO = None
    COCOeval = None

from utils.file_io import dump_json
from utils.dist import all_gather, barrier, is_main_process
from utils.metrics import save_eval_outputs
from utils.misc import to_device


class Evaluator:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger

    @torch.no_grad()
    def evaluate(self, model, dataloader, device, output_dir=None):
        if COCO is None or COCOeval is None:
            raise ImportError(
                "pycocotools is required for evaluation. Please install dependencies from requirements.txt first."
            )
        eval_cfg = self.cfg.get("EVAL", {})
        score_thresh = float(eval_cfg.get("SCORE_THRESH", 0.05))
        max_dets = int(eval_cfg.get("MAX_DETS", 100))
        metric_name = str(eval_cfg.get("METRIC", "bbox")).lower()
        if metric_name != "bbox":
            raise ValueError(f"Unsupported EVAL.METRIC={metric_name}. This project currently supports bbox only.")
        use_coco_eval = bool(eval_cfg.get("USE_COCO_EVAL", True))
        compute_per_class = bool(eval_cfg.get("PER_CLASS_AP", True))
        save_predictions = bool(eval_cfg.get("SAVE_PREDICTIONS", True))
        save_gt = bool(eval_cfg.get("SAVE_GT", False))
        if bool(eval_cfg.get("VISUALIZE", False)) and self.logger:
            self.logger.warning("EVAL.VISUALIZE is set, but evaluator visualization is not implemented yet. Skipping.")

        model.eval()
        start = time.time()
        predictions = []
        gt_coco = {"images": [], "annotations": [], "categories": []}
        ann_id = 1
        image_ids = set()
        if dataloader is None or len(dataloader) == 0:
            raise RuntimeError("Evaluator received an empty dataloader. Please check dataset split and dataloader config.")
        label_to_cat_id, label_to_cat_name = self._build_category_mapping(dataloader.dataset)
        gt_coco["categories"] = [
            {"id": int(cat_id), "name": str(label_to_cat_name[label])}
            for label, cat_id in sorted(label_to_cat_id.items(), key=lambda x: int(x[0]))
        ]

        for images, targets in dataloader:
            images, targets = to_device(images, targets, device)
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                img_id = int(tgt["image_id"].item())
                if img_id not in image_ids:
                    image_ids.add(img_id)
                    gt_coco["images"].append({"id": img_id})
                    boxes_gt = tgt.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).detach().cpu().numpy()
                    labels_gt = tgt.get("labels", torch.zeros((0,), dtype=torch.int64)).detach().cpu().numpy()
                    crowds_gt = tgt.get("iscrowd", torch.zeros((len(boxes_gt),), dtype=torch.int64)).detach().cpu().numpy()
                    areas_gt = tgt.get("area", torch.zeros((len(boxes_gt),), dtype=torch.float32)).detach().cpu().numpy()
                    for i, (b, lbl) in enumerate(zip(boxes_gt, labels_gt)):
                        if int(lbl) <= 0:
                            continue
                        x1, y1, x2, y2 = b.tolist()
                        w = max(0.0, float(x2 - x1))
                        h = max(0.0, float(y2 - y1))
                        if w <= 0 or h <= 0:
                            continue
                        cat_id = label_to_cat_id.get(int(lbl))
                        if cat_id is None:
                            continue
                        area = float(areas_gt[i]) if i < len(areas_gt) else float(w * h)
                        iscrowd = int(crowds_gt[i]) if i < len(crowds_gt) else 0
                        gt_coco["annotations"].append(
                            {
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": int(cat_id),
                                "bbox": [float(x1), float(y1), w, h],
                                "area": area,
                                "iscrowd": iscrowd,
                            }
                        )
                        ann_id += 1

                if not all(k in out for k in ["boxes", "scores", "labels"]):
                    missing = [k for k in ["boxes", "scores", "labels"] if k not in out]
                    raise KeyError(f"Model prediction output missing fields: {missing}")

                boxes = out["boxes"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                labels = out["labels"].detach().cpu().numpy()
                keep = scores >= score_thresh
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                if max_dets > 0 and len(scores) > max_dets:
                    order = np.argsort(-scores)[:max_dets]
                    boxes, scores, labels = boxes[order], scores[order], labels[order]
                for b, s, l in zip(boxes, scores, labels):
                    if int(l) <= 0:
                        continue
                    cat_id = label_to_cat_id.get(int(l))
                    if cat_id is None:
                        continue
                    x1, y1, x2, y2 = b.tolist()
                    predictions.append(
                        {
                            "image_id": img_id,
                            "category_id": int(cat_id),
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(s),
                        }
                    )

        gathered_predictions = []
        gathered_gt = []
        for part in all_gather(predictions):
            gathered_predictions.extend(part)
        for part in all_gather(gt_coco):
            gathered_gt.append(part)

        if is_main_process():
            merged_gt = self._merge_gt_dicts(gathered_gt)
            metrics, per_class_ap = self._compute_metrics(
                gt_dict=merged_gt,
                predictions=gathered_predictions,
                per_class=compute_per_class,
                use_coco_eval=use_coco_eval,
                label_to_cat_name=label_to_cat_name,
                label_to_cat_id=label_to_cat_id,
            )
            metrics["eval_time"] = time.time() - start
            metrics["num_images"] = len(merged_gt["images"])
            metrics["num_predictions"] = len(gathered_predictions)
            metrics["metric_name"] = metric_name
            metrics["best_metric_key"] = "map"
            metrics["mAP"] = metrics["map"]  # backward-compatible aliases
            metrics["AP50"] = metrics["map50"]
            metrics["AP75"] = metrics["map75"]
            metrics["AR"] = metrics["ar"]
        else:
            merged_gt = None
            metrics, per_class_ap = {}, []

        gathered_result = all_gather({"metrics": metrics, "per_class_ap": per_class_ap})
        metrics = gathered_result[0]["metrics"]
        per_class_ap = gathered_result[0]["per_class_ap"]

        if is_main_process() and output_dir:
            save_eval_outputs(metrics, per_class_ap, output_dir)
            if save_predictions:
                dump_json(gathered_predictions, os.path.join(output_dir, "predictions.json"))
            if save_gt:
                dump_json(merged_gt, os.path.join(output_dir, "ground_truth.json"))

        if is_main_process() and self.logger:
            self.logger.info(
                "Eval bbox metrics | "
                f"mAP={metrics['map']:.4f} AP50={metrics['map50']:.4f} AP75={metrics['map75']:.4f} "
                f"AR={metrics['ar']:.4f} eval_time={metrics['eval_time']:.2f}s"
            )
            if per_class_ap:
                self.logger.info(f"Eval per-class AP: {per_class_ap}")

        barrier()
        return metrics, per_class_ap

    def _merge_gt_dicts(self, parts):
        merged = {"images": [], "annotations": [], "categories": []}
        seen_img = set()
        seen_ann = set()
        cat_map = {}
        next_ann_id = 1
        for gt in parts:
            for c in gt.get("categories", []):
                cat_map[int(c["id"])] = c
            for img in gt.get("images", []):
                iid = int(img["id"])
                if iid in seen_img:
                    continue
                seen_img.add(iid)
                merged["images"].append({"id": iid})
            for ann in gt.get("annotations", []):
                raw_id = int(ann.get("id", next_ann_id))
                if raw_id in seen_ann:
                    raw_id = next_ann_id
                seen_ann.add(raw_id)
                next_ann_id = max(next_ann_id, raw_id + 1)
                copied = dict(ann)
                copied["id"] = raw_id
                merged["annotations"].append(copied)
        merged["categories"] = [cat_map[k] for k in sorted(cat_map.keys())]
        return merged

    def _build_category_mapping(self, dataset) -> Tuple[Dict[int, int], Dict[int, str]]:
        class_names = list(self.cfg["DATASET"].get("CLASSES", []))
        label_to_cat_id = {}
        label_to_cat_name = {}

        if hasattr(dataset, "cat_id_to_label"):
            for cat_id, label in dataset.cat_id_to_label.items():
                label_i = int(label)
                label_to_cat_id[label_i] = int(cat_id)
                label_to_cat_name[label_i] = class_names[label_i - 1] if 0 < label_i <= len(class_names) else str(cat_id)
        else:
            for i, name in enumerate(class_names, start=1):
                label_to_cat_id[i] = i
                label_to_cat_name[i] = name

        if not label_to_cat_id:
            raise ValueError("No valid foreground categories were resolved for evaluator.")
        return label_to_cat_id, label_to_cat_name

    def _compute_metrics(self, gt_dict, predictions, per_class, use_coco_eval, label_to_cat_name, label_to_cat_id):
        if not use_coco_eval:
            raise ValueError("EVAL.USE_COCO_EVAL=False is not implemented yet; please enable COCO-style bbox eval.")

        if len(gt_dict.get("images", [])) == 0:
            raise RuntimeError("No images were collected during evaluation.")

        if len(gt_dict.get("annotations", [])) == 0:
            if self.logger:
                self.logger.warning("No GT boxes found in evaluation split. Returning zero bbox metrics.")
            return self._empty_metrics()

        if len(predictions) == 0:
            if self.logger:
                self.logger.warning("No predictions after score/max_dets filtering. Returning zero bbox metrics.")
            return self._empty_metrics()

        return self._compute_coco_metrics(gt_dict, predictions, per_class, label_to_cat_name, label_to_cat_id)

    @staticmethod
    def _empty_metrics():
        metrics = {
            "map": 0.0,
            "map50": 0.0,
            "map75": 0.0,
            "ar": 0.0,
        }
        return metrics, []

    def _compute_coco_metrics(self, gt_dict, predictions, compute_per_class, label_to_cat_name, label_to_cat_id):
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        if self.logger:
            coco_eval.summarize()

        stats = coco_eval.stats
        metrics = {
            "map": float(stats[0]),
            "map50": float(stats[1]),
            "map75": float(stats[2]),
            "ar": float(stats[8]),
        }

        per_class = []
        if compute_per_class:
            precisions = coco_eval.eval["precision"]  # [TxRxKxAxM]
            cat_ids = coco_eval.params.catIds
            cat_to_k = {int(cid): i for i, cid in enumerate(cat_ids)}
            for label, cat_id in sorted(label_to_cat_id.items(), key=lambda x: int(x[0])):
                k = cat_to_k.get(int(cat_id))
                if k is None:
                    continue
                precision = precisions[:, :, k, 0, -1]
                precision = precision[precision > -1]
                ap = float(np.mean(precision)) if precision.size else 0.0
                per_class.append(
                    {
                        "label_id": int(label),
                        "category_id": int(cat_id),
                        "category_name": label_to_cat_name[int(label)],
                        "ap": ap,
                    }
                )

        return metrics, per_class
