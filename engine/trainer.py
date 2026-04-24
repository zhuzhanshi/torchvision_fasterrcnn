from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import nullcontext

import torch

from .evaluator import Evaluator
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.dist import is_main_process, reduce_dict
from utils.file_io import dump_json
from utils.misc import is_finite_number, to_device


class Trainer:
    def __init__(self, cfg, model, optimizer, scheduler, train_loader, val_loader, device, logger, output_dir):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.output_dir = output_dir

        self.amp_enabled = bool(cfg["RUNTIME"].get("USE_AMP", cfg["TRAIN"].get("AMP", True)))
        self.scaler = self._build_scaler(device)

        self.best_metric = float("-inf")
        self.start_epoch = int(cfg["TRAIN"].get("START_EPOCH", 0))
        self.evaluator = Evaluator(cfg, logger=logger)

        resume_path = cfg["RUNTIME"].get("RESUME", cfg["TRAIN"].get("RESUME", ""))
        if resume_path:
            self.resume(resume_path)

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    def _build_scaler(self, device):
        if not self.amp_enabled:
            return torch.cuda.amp.GradScaler(enabled=False)
        if device.type == "cuda":
            return torch.cuda.amp.GradScaler(enabled=True)
        if device.type == "npu":
            # Conservative fallback for broad torch/torch_npu compatibility.
            # If future torch_npu exposes a stable GradScaler API, this can be upgraded.
            self.logger.warning("AMP requested on NPU; GradScaler fallback is disabled for compatibility in this build.")
            return torch.cuda.amp.GradScaler(enabled=False)
        return torch.cuda.amp.GradScaler(enabled=False)

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=True)
        if self.device.type == "npu":
            try:
                return torch.autocast(device_type="npu", enabled=True)
            except Exception:
                self.logger.warning("AMP autocast for NPU is unavailable; training will continue in FP32.")
                return nullcontext()
        return nullcontext()

    def resume(self, path):
        ckpt = load_checkpoint(path, map_location="cpu")

        required = ["model", "optimizer", "epoch"]
        for k in required:
            if k not in ckpt:
                raise KeyError(f"Resume checkpoint missing required key: {k}")

        self._unwrap_model(self.model).load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.scaler is not None and ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])

        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.best_metric = float(ckpt.get("best_metric", float("-inf")))
        self.logger.info(f"Resumed training from {path}, next_epoch={self.start_epoch}, best_metric={self.best_metric:.6f}")

    def _build_checkpoint_state(self, epoch):
        return {
            "model": self._unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler and self.scaler.is_enabled() else None,
            "epoch": epoch,
            "best_metric": self.best_metric,
            "cfg": {
                "model": self.cfg["MODEL"]["NAME"],
                "num_classes": self.cfg["DATASET"]["NUM_CLASSES"],
                "epochs": self.cfg["TRAIN"]["EPOCHS"],
                "optimizer": self.cfg["OPTIMIZER"]["NAME"],
                "scheduler": self.cfg["SCHEDULER"]["NAME"],
                "amp": self.cfg["RUNTIME"].get("USE_AMP", False),
            },
        }

    def save_checkpoint(self, epoch, is_best=False):
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        state = self._build_checkpoint_state(epoch)

        runtime_cfg = self.cfg["RUNTIME"]
        save_best_only = bool(runtime_cfg.get("SAVE_BEST_ONLY", False))
        save_every = bool(runtime_cfg.get("SAVE_EVERY_EPOCH", True))
        save_interval = int(runtime_cfg.get("SAVE_INTERVAL", 1))

        should_save_latest = save_every and ((epoch + 1) % max(save_interval, 1) == 0)
        if should_save_latest and not save_best_only:
            save_checkpoint(os.path.join(ckpt_dir, "latest.pth"), state)

        if is_best:
            save_checkpoint(os.path.join(ckpt_dir, "best.pth"), state)
            if save_best_only:
                save_checkpoint(os.path.join(ckpt_dir, "latest.pth"), state)

    def train(self):
        if self.train_loader is None or len(self.train_loader) == 0:
            raise RuntimeError("train_loader is empty. Cannot start training.")

        total_epochs = int(self.cfg["TRAIN"]["EPOCHS"])
        eval_cfg = self.cfg.get("EVAL", {})
        eval_during_train = bool(eval_cfg.get("DURING_TRAIN", eval_cfg.get("ENABLE", eval_cfg.get("ENABLED", True))))
        eval_after_train = bool(eval_cfg.get("AFTER_TRAIN", False))
        eval_interval = int(eval_cfg.get("INTERVAL", 1))
        best_metric_name = str(eval_cfg.get("BEST_METRIC", self.cfg["TRAIN"].get("SAVE_BEST_METRIC", "map")))
        if is_main_process():
            self.logger.info(
                f"Training started: start_epoch={self.start_epoch}, total_epochs={total_epochs}, "
                f"accumulation={self.cfg['TRAIN'].get('ACCUMULATION_STEPS', 1)}, amp={self.amp_enabled}, device={self.device}, "
                f"eval_during_train={eval_during_train}, eval_interval={eval_interval}, eval_after_train={eval_after_train}"
            )

        for epoch in range(self.start_epoch, total_epochs):
            if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            train_stats = self.train_one_epoch(epoch)

            do_val = self.val_loader is not None and eval_during_train and ((epoch + 1) % max(eval_interval, 1) == 0)

            metric_value = None
            if do_val:
                metrics = self.validate(epoch)
                metric_value = float(metrics.get(best_metric_name, 0.0))
                if metric_value > self.best_metric:
                    self.best_metric = metric_value
                    self.save_checkpoint(epoch, is_best=True)

            self.save_checkpoint(epoch, is_best=False)

            if self.scheduler:
                self.scheduler.step()

            if is_main_process():
                self.logger.info(
                    f"epoch={epoch} done | avg_loss_total={train_stats['avg_loss_total']:.6f} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.8f} best_metric={self.best_metric:.6f}"
                )

            if self.cfg["TRAIN"].get("EMPTY_CACHE_PER_EPOCH", False) and self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Final evaluation at the end of training (optional)
        if self.val_loader is not None and eval_after_train:
            final_metrics = self.validate(total_epochs - 1, tag="final")
            if is_main_process():
                final_dir = os.path.join(self.output_dir, "eval", "final")
                os.makedirs(final_dir, exist_ok=True)
                dump_json(final_metrics, os.path.join(final_dir, "final_metrics.json"))
                self.logger.info(
                    f"Final eval completed after training. best_metric_during_train={self.best_metric:.6f} "
                    f"final_metrics={final_metrics}"
                )

    def _grad_clip_cfg(self):
        cfg = self.cfg["TRAIN"].get("GRAD_CLIP", 0.0)
        if isinstance(cfg, dict):
            return bool(cfg.get("ENABLE", False)), float(cfg.get("MAX_NORM", 0.0)), float(cfg.get("NORM_TYPE", 2.0))
        # backward compatibility: float value means enabled by max_norm>0
        max_norm = float(cfg if cfg is not None else self.cfg["TRAIN"].get("GRAD_CLIP_NORM", 0.0))
        return max_norm > 0, max_norm, 2.0

    @staticmethod
    def _target_stats_per_image(image, target):
        boxes = target.get("boxes")
        labels = target.get("labels")
        h = int(image.shape[-2]) if hasattr(image, "shape") else -1
        w = int(image.shape[-1]) if hasattr(image, "shape") else -1
        stat = {
            "image_hw": [h, w],
            "num_boxes": 0,
            "is_empty_target": True,
            "boxes_min": None,
            "boxes_max": None,
            "box_w_minmax": None,
            "box_h_minmax": None,
            "labels_unique": [],
            "has_label_zero": False,
            "coords_likely_normalized_0_1": False,
        }

        if boxes is not None and boxes.numel() > 0:
            stat["num_boxes"] = int(boxes.shape[0])
            stat["is_empty_target"] = False
            stat["boxes_min"] = float(boxes.min().item())
            stat["boxes_max"] = float(boxes.max().item())
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
            stat["box_w_minmax"] = [float(bw.min().item()), float(bw.max().item())]
            stat["box_h_minmax"] = [float(bh.min().item()), float(bh.max().item())]
            # Heuristic: all coords <= 1 is usually normalized coordinates, suspicious for torchvision detection inputs.
            stat["coords_likely_normalized_0_1"] = bool(float(boxes.max().item()) <= 1.0)

        if labels is not None and labels.numel() > 0:
            uniq = torch.unique(labels).tolist()
            stat["labels_unique"] = [int(x) for x in uniq]
            stat["has_label_zero"] = any(int(x) == 0 for x in uniq)

        return stat

    def train_one_epoch(self, epoch):
        self.model.train()
        if len(self.train_loader) == 0:
            raise RuntimeError("train_loader has zero length.")

        meter = defaultdict(float)
        iters = len(self.train_loader)
        accum = max(1, int(self.cfg["TRAIN"].get("ACCUMULATION_STEPS", 1)))
        clip_enable, clip_max_norm, clip_norm_type = self._grad_clip_cfg()

        self.optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        last_time = time.time()
        debug_iters = int(self.cfg["RUNTIME"].get("DEBUG_ITERS", 3))

        for i, (images, targets) in enumerate(self.train_loader):
            data_time = time.time() - last_time
            images, targets = to_device(images, targets, self.device)

            if i < debug_iters and is_main_process():
                per_img_stats = [self._target_stats_per_image(img, tgt) for img, tgt in zip(images, targets)]
                num_empty = sum(int(s["is_empty_target"]) for s in per_img_stats)
                self.logger.info(
                    f"[debug][target] epoch={epoch} iter={i+1}/{iters} "
                    f"batch_size={len(images)} empty_targets={num_empty}/{len(images)} stats={per_img_stats}"
                )

            with self._autocast_context():
                loss_dict = self.model(images, targets)
                loss_total_unscaled = sum(loss for loss in loss_dict.values())
                loss_scaled = loss_total_unscaled / accum

            loss_dict_reduced = reduce_dict(loss_dict, average=True)
            loss_total_reduced_unscaled = sum(v for v in loss_dict_reduced.values())
            loss_value_unscaled = float(loss_total_reduced_unscaled.item())
            if not is_finite_number(loss_value_unscaled):
                raise FloatingPointError(
                    f"Non-finite loss detected at epoch={epoch}, iter={i}: "
                    f"loss_total_unscaled={loss_value_unscaled}"
                )

            if i < debug_iters and is_main_process():
                loss_tensor_meta = {
                    k: {
                        "value": float(v.detach().item()),
                        "dtype": str(v.dtype),
                        "device": str(v.device),
                    }
                    for k, v in loss_dict.items()
                }
                self.logger.info(
                    f"[debug][loss_dict] epoch={epoch} iter={i+1}/{iters} "
                    f"unscaled_total={loss_value_unscaled:.6f} scaled_for_backward={float(loss_scaled.detach().item()):.6f} "
                    f"details={loss_tensor_meta}"
                )

            self.scaler.scale(loss_scaled).backward()

            should_step = ((i + 1) % accum == 0) or ((i + 1) == iters)
            if should_step:
                if clip_enable and clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_max_norm, norm_type=clip_norm_type)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            iter_time = time.time() - last_time
            last_time = time.time()

            meter["loss_total"] += loss_value_unscaled
            meter["loss_classifier"] += float(loss_dict_reduced.get("loss_classifier", torch.tensor(0.0)).item())
            meter["loss_box_reg"] += float(loss_dict_reduced.get("loss_box_reg", torch.tensor(0.0)).item())
            meter["loss_objectness"] += float(loss_dict_reduced.get("loss_objectness", torch.tensor(0.0)).item())
            meter["loss_rpn_box_reg"] += float(loss_dict_reduced.get("loss_rpn_box_reg", torch.tensor(0.0)).item())

            lr = self.optimizer.param_groups[0]["lr"]
            print_freq = int(self.cfg["RUNTIME"].get("PRINT_FREQ", self.cfg["TRAIN"].get("PRINT_FREQ", 20)))
            if (i + 1) % print_freq == 0:
                eta_sec = (iters - i - 1) * (time.time() - epoch_start) / max(1, i + 1)
                if self.device.type == "cuda":
                    gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
                elif self.device.type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "memory_allocated"):
                    gpu_mem = torch.npu.memory_allocated() / 1024**2
                else:
                    gpu_mem = 0.0
                msg = (
                    f"epoch={epoch} iter={i+1}/{iters} lr={lr:.8f} "
                    f"loss_total={loss_value_unscaled:.4f} "
                    f"loss_classifier={float(loss_dict_reduced.get('loss_classifier', torch.tensor(0.)).item()):.4f} "
                    f"loss_box_reg={float(loss_dict_reduced.get('loss_box_reg', torch.tensor(0.)).item()):.4f} "
                    f"loss_objectness={float(loss_dict_reduced.get('loss_objectness', torch.tensor(0.)).item()):.4f} "
                    f"loss_rpn_box_reg={float(loss_dict_reduced.get('loss_rpn_box_reg', torch.tensor(0.)).item()):.4f} "
                    f"data_time={data_time:.3f}s iter_time={iter_time:.3f}s eta={eta_sec/60:.1f}m "
                    + (f"gpu_memory={gpu_mem:.1f}MB" if self.cfg["LOG"].get("LOG_MEMORY", True) else "gpu_memory=disabled")
                )
                if is_main_process():
                    self.logger.info(msg)

            global_step = epoch * iters + i
            if self.cfg["LOG"].get("LOG_ITER_LOSS", True):
                iter_scalars = {
                    "loss_total": loss_value_unscaled,
                    "loss_classifier": float(loss_dict_reduced.get("loss_classifier", torch.tensor(0.0)).item()),
                    "loss_box_reg": float(loss_dict_reduced.get("loss_box_reg", torch.tensor(0.0)).item()),
                    "loss_objectness": float(loss_dict_reduced.get("loss_objectness", torch.tensor(0.0)).item()),
                    "loss_rpn_box_reg": float(loss_dict_reduced.get("loss_rpn_box_reg", torch.tensor(0.0)).item()),
                }
                if self.cfg["LOG"].get("LOG_LR", True):
                    iter_scalars["lr"] = lr
                if is_main_process():
                    self.logger.log_scalars("train_iter", iter_scalars, global_step)

        epoch_time = time.time() - epoch_start
        avg = {
            "avg_loss_total": meter["loss_total"] / iters,
            "avg_loss_classifier": meter["loss_classifier"] / iters,
            "avg_loss_box_reg": meter["loss_box_reg"] / iters,
            "avg_loss_objectness": meter["loss_objectness"] / iters,
            "avg_loss_rpn_box_reg": meter["loss_rpn_box_reg"] / iters,
            "epoch_time": epoch_time,
            "current_lr": self.optimizer.param_groups[0]["lr"],
            "best_metric_so_far": self.best_metric,
        }
        if is_main_process():
            self.logger.info(f"epoch={epoch} summary={avg}")
        if self.cfg["LOG"].get("LOG_EPOCH_LOSS", True):
            epoch_scalars = dict(avg)
            if not self.cfg["LOG"].get("LOG_LR", True):
                epoch_scalars.pop("current_lr", None)
            if is_main_process():
                self.logger.log_scalars("train_epoch", epoch_scalars, epoch)
        return avg

    def validate(self, epoch, tag: str | None = None):
        if self.val_loader is None:
            self.logger.warning("validate() called but val_loader is None. Skip validation.")
            return {}
        if tag:
            out_dir = os.path.join(self.output_dir, "eval", tag)
        else:
            out_dir = os.path.join(self.output_dir, "eval", f"epoch_{epoch:03d}")
        metrics, per_class_ap = self.evaluator.evaluate(self.model, self.val_loader, self.device, output_dir=out_dir)
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if is_main_process():
            self.logger.log_scalars("val", scalar_metrics, epoch)
        if per_class_ap and is_main_process():
            self.logger.info(f"epoch={epoch} per-class AP rows={len(per_class_ap)}")
        return metrics
