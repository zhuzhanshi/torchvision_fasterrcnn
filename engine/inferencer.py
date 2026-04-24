from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image

from datasets.transforms import build_transforms
from utils.file_io import dump_json, dump_text, ensure_dir
from utils.visualize import draw_predictions


class Inferencer:
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.class_names = list(cfg["DATASET"].get("CLASSES", []))
        self.class_filter_ids = self._resolve_class_filter(cfg["INFER"].get("CLASS_FILTER", []))
        self.preprocess = build_transforms(cfg, is_train=False)

    def _resolve_class_filter(self, class_filter_cfg) -> set[int]:
        if not class_filter_cfg:
            return set()

        name_to_label = {name: i + 1 for i, name in enumerate(self.class_names)}
        filter_ids = set()
        for item in class_filter_cfg:
            if isinstance(item, int):
                filter_ids.add(int(item))
                continue
            text = str(item).strip()
            if text.isdigit():
                filter_ids.add(int(text))
                continue
            if text in name_to_label:
                filter_ids.add(name_to_label[text])
                continue
            raise ValueError(
                f"INFER.CLASS_FILTER item={item!r} is invalid. Use class name from DATASET.CLASSES or label id (>=1)."
            )
        return {x for x in filter_ids if x > 0}

    def _gather_inputs(self, input_path: str) -> Tuple[List[Path], Path]:
        if not input_path:
            raise ValueError("INFER.INPUT_PATH is empty. Please set --input-path or INFER.INPUT_PATH.")

        p = Path(input_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Infer input path does not exist: {p}")
        if p.is_file():
            if p.suffix.lower() not in self.IMAGE_EXTS:
                raise ValueError(f"Input file is not a supported image type: {p}")
            return [p], p.parent

        recursive = bool(self.cfg["INFER"].get("RECURSIVE", False))
        iter_paths = p.rglob("*") if recursive else p.glob("*")
        files = sorted([x for x in iter_paths if x.is_file() and x.suffix.lower() in self.IMAGE_EXTS])
        if not files:
            raise RuntimeError(
                f"No image files found under directory: {p} (recursive={recursive}, supported_exts={sorted(self.IMAGE_EXTS)})"
            )
        return files, p

    def _relative_stem(self, file_path: Path, root_dir: Path) -> str:
        try:
            rel = file_path.relative_to(root_dir)
        except ValueError:
            rel = Path(file_path.name)
        return str(rel.with_suffix("")).replace("\\", "/")

    def _build_output_paths(self, output_dir: str, rel_stem: str):
        safe_stem = rel_stem.replace("/", "__")
        vis_path = os.path.join(output_dir, "vis", f"{safe_stem}.jpg")
        json_path = os.path.join(output_dir, "json", f"{safe_stem}.json")
        txt_path = os.path.join(output_dir, "txt", f"{safe_stem}.txt")
        return vis_path, json_path, txt_path

    def _postprocess(self, pred: Dict) -> Dict:
        infer_cfg = self.cfg["INFER"]
        score_thresh = float(infer_cfg.get("SCORE_THRESH", 0.5))
        max_dets = int(infer_cfg.get("MAX_DETS", 100))

        if not all(k in pred for k in ["boxes", "labels", "scores"]):
            missing = [k for k in ["boxes", "labels", "scores"] if k not in pred]
            raise KeyError(f"Inference output missing required fields: {missing}")

        boxes = pred["boxes"].detach().cpu()
        labels = pred["labels"].detach().cpu().to(torch.int64)
        scores = pred["scores"].detach().cpu()

        keep = scores >= score_thresh
        keep &= labels > 0  # remove background
        if self.class_filter_ids:
            filter_tensor = torch.tensor([int(x) in self.class_filter_ids for x in labels.tolist()], dtype=torch.bool)
            keep &= filter_tensor

        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
        if max_dets > 0 and len(scores) > max_dets:
            order = torch.argsort(scores, descending=True)[:max_dets]
            boxes, labels, scores = boxes[order], labels[order], scores[order]

        label_names = [
            self.class_names[int(l) - 1] if 0 < int(l) <= len(self.class_names) else f"cls_{int(l)}" for l in labels.tolist()
        ]
        return {
            "boxes": boxes.tolist(),
            "labels": [int(x) for x in labels.tolist()],
            "scores": [float(x) for x in scores.tolist()],
            "label_names": label_names,
        }

    def _save_txt(self, path: str, result: Dict):
        lines = []
        for box, label, score, label_name in zip(result["boxes"], result["labels"], result["scores"], result["label_names"]):
            box_str = " ".join(f"{float(v):.2f}" for v in box)
            lines.append(f"{label} {label_name} {score:.6f} {box_str}")
        dump_text("\n".join(lines) + ("\n" if lines else ""), path)

    @torch.no_grad()
    def run(self, model, device, output_dir):
        model.eval()
        infer_cfg = self.cfg["INFER"]
        model_ref = model.module if hasattr(model, "module") else model

        min_size = int(infer_cfg.get("MIN_SIZE", 0) or 0)
        max_size = int(infer_cfg.get("MAX_SIZE", 0) or 0)
        if hasattr(model_ref, "transform"):
            if min_size > 0:
                model_ref.transform.min_size = (min_size,)
            if max_size > 0:
                model_ref.transform.max_size = max_size
            if self.logger and (min_size > 0 or max_size > 0):
                self.logger.info(
                    f"Infer override model transform size: min_size={model_ref.transform.min_size}, "
                    f"max_size={model_ref.transform.max_size}"
                )

        # torchvisions's ROIHeads already performs NMS; we expose infer-time override for deployment convenience.
        model_ref.roi_heads.score_thresh = float(infer_cfg.get("SCORE_THRESH", model_ref.roi_heads.score_thresh))
        model_ref.roi_heads.nms_thresh = float(infer_cfg.get("NMS_THRESH", model_ref.roi_heads.nms_thresh))
        model_ref.roi_heads.detections_per_img = int(infer_cfg.get("MAX_DETS", model_ref.roi_heads.detections_per_img))

        files, root_dir = self._gather_inputs(infer_cfg.get("INPUT_PATH", ""))
        for sub in ["vis", "json", "txt"]:
            ensure_dir(os.path.join(output_dir, sub))

        if self.logger:
            self.logger.info(
                f"Inference start: files={len(files)}, recursive={infer_cfg.get('RECURSIVE', False)}, "
                f"save_vis={infer_cfg.get('SAVE_VIS', True)}, save_json={infer_cfg.get('SAVE_JSON', True)}, "
                f"save_txt={infer_cfg.get('SAVE_TXT', False)}"
            )

        all_results = []
        failed_files = []

        for file_path in files:
            try:
                image = Image.open(file_path).convert(self.cfg["INPUT"].get("IMAGE_FORMAT", "RGB"))
                dummy_target = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "image_id": torch.tensor([0], dtype=torch.int64),
                    "area": torch.zeros((0,), dtype=torch.float32),
                    "iscrowd": torch.zeros((0,), dtype=torch.int64),
                }
                img_tensor, _ = self.preprocess(image, dummy_target)
                output = model([img_tensor.to(device)])[0]
                result = self._postprocess(output)

                rel_stem = self._relative_stem(file_path, root_dir)
                vis_path, json_path, txt_path = self._build_output_paths(output_dir, rel_stem)

                image_result = {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "relative_id": rel_stem,
                    "boxes": result["boxes"],
                    "labels": result["labels"],
                    "scores": result["scores"],
                    "label_names": result["label_names"],
                }
                all_results.append(image_result)

                if infer_cfg.get("SAVE_JSON", True):
                    dump_json(image_result, json_path)
                if infer_cfg.get("SAVE_TXT", False):
                    self._save_txt(txt_path, image_result)
                if infer_cfg.get("SAVE_VIS", True):
                    vis_image = draw_predictions(
                        image=image,
                        boxes=image_result["boxes"],
                        labels=image_result["labels"],
                        scores=image_result["scores"],
                        class_names=self.class_names,
                        draw_label=bool(infer_cfg.get("DRAW_LABEL", True)),
                        draw_score=bool(infer_cfg.get("DRAW_SCORE", True)),
                        line_thickness=int(infer_cfg.get("LINE_THICKNESS", 2)),
                    )
                    vis_image.save(vis_path)
            except Exception as e:  # continue for per-image failures
                failed_files.append({"file_path": str(file_path), "error": str(e)})
                if self.logger:
                    self.logger.warning(f"Inference skipped for {file_path}: {e}")

        summary = {
            "num_total": len(files),
            "num_success": len(all_results),
            "num_failed": len(failed_files),
            "results": all_results,
            "failed": failed_files,
        }
        dump_json(summary, os.path.join(output_dir, "predictions_all.json"))

        if self.logger:
            self.logger.info(
                f"Inference finished. success={summary['num_success']}/{summary['num_total']} failed={summary['num_failed']} "
                f"output_dir={output_dir}"
            )
        if summary["num_success"] == 0:
            raise RuntimeError("Inference finished with zero successful images. Check input files and logs.")
        return summary
