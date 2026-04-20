from __future__ import annotations

import os
import time
from collections import defaultdict

import torch
from torch.cuda.amp import GradScaler, autocast

from .evaluator import Evaluator
from utils.checkpoint import load_checkpoint, save_checkpoint
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

        amp_enabled = bool(cfg["RUNTIME"].get("USE_AMP", cfg["TRAIN"].get("AMP", True))) and device.type == "cuda"
        self.scaler = GradScaler(enabled=amp_enabled)

        self.best_metric = float("-inf")
        self.start_epoch = int(cfg["TRAIN"].get("START_EPOCH", 0))
        self.evaluator = Evaluator(cfg, logger=logger)

        resume_path = cfg["RUNTIME"].get("RESUME", cfg["TRAIN"].get("RESUME", ""))
        if resume_path:
            self.resume(resume_path)

    def resume(self, path):
        ckpt = load_checkpoint(path, map_location="cpu")

        required = ["model", "optimizer", "epoch"]
        for k in required:
            if k not in ckpt:
                raise KeyError(f"Resume checkpoint missing required key: {k}")

        self.model.load_state_dict(ckpt["model"])
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
            "model": self.model.state_dict(),
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
        self.logger.info(
            f"Training started: start_epoch={self.start_epoch}, total_epochs={total_epochs}, "
            f"accumulation={self.cfg['TRAIN'].get('ACCUMULATION_STEPS', 1)}, amp={self.scaler.is_enabled()}"
        )

        for epoch in range(self.start_epoch, total_epochs):
            train_stats = self.train_one_epoch(epoch)

            eval_enabled = bool(self.cfg["EVAL"].get("ENABLE", self.cfg["EVAL"].get("ENABLED", True)))
            eval_interval = int(
                self.cfg["EVAL"].get(
                    "INTERVAL",
                    self.cfg["TRAIN"].get("VALIDATE_EVERY_EPOCH", self.cfg["TRAIN"].get("VAL_EVERY_EPOCH", 1)),
                )
            )
            do_val = self.val_loader is not None and eval_enabled and ((epoch + 1) % max(eval_interval, 1) == 0)

            metric_value = None
            if do_val:
                metrics = self.validate(epoch)
                metric_name = self.cfg["TRAIN"].get("SAVE_BEST_METRIC", metrics.get("best_metric_key", "map"))
                metric_value = float(metrics.get(metric_name, 0.0))
                if metric_value > self.best_metric:
                    self.best_metric = metric_value
                    self.save_checkpoint(epoch, is_best=True)

            self.save_checkpoint(epoch, is_best=False)

            if self.scheduler:
                self.scheduler.step()

            self.logger.info(
                f"epoch={epoch} done | avg_loss_total={train_stats['avg_loss_total']:.6f} "
                f"lr={self.optimizer.param_groups[0]['lr']:.8f} best_metric={self.best_metric:.6f}"
            )

            if self.cfg["TRAIN"].get("EMPTY_CACHE_PER_EPOCH", False) and self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _grad_clip_cfg(self):
        cfg = self.cfg["TRAIN"].get("GRAD_CLIP", 0.0)
        if isinstance(cfg, dict):
            return bool(cfg.get("ENABLE", False)), float(cfg.get("MAX_NORM", 0.0)), float(cfg.get("NORM_TYPE", 2.0))
        # backward compatibility: float value means enabled by max_norm>0
        max_norm = float(cfg if cfg is not None else self.cfg["TRAIN"].get("GRAD_CLIP_NORM", 0.0))
        return max_norm > 0, max_norm, 2.0

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

        for i, (images, targets) in enumerate(self.train_loader):
            data_time = time.time() - last_time
            images, targets = to_device(images, targets, self.device)

            with autocast(enabled=self.scaler.is_enabled()):
                loss_dict = self.model(images, targets)
                loss_total = sum(loss for loss in loss_dict.values())
                loss = loss_total / accum

            loss_value = float(loss_total.item())
            if not is_finite_number(loss_value):
                raise FloatingPointError(f"Non-finite loss detected at epoch={epoch}, iter={i}: loss_total={loss_value}")

            self.scaler.scale(loss).backward()

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

            meter["loss_total"] += loss_value
            meter["loss_classifier"] += float(loss_dict.get("loss_classifier", torch.tensor(0.0)).item())
            meter["loss_box_reg"] += float(loss_dict.get("loss_box_reg", torch.tensor(0.0)).item())
            meter["loss_objectness"] += float(loss_dict.get("loss_objectness", torch.tensor(0.0)).item())
            meter["loss_rpn_box_reg"] += float(loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item())

            lr = self.optimizer.param_groups[0]["lr"]
            if (i + 1) % int(self.cfg["RUNTIME"].get("PRINT_FREQ", self.cfg["TRAIN"].get("PRINT_FREQ", 20))) == 0:
                eta_sec = (iters - i - 1) * (time.time() - epoch_start) / max(1, i + 1)
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if self.device.type == "cuda" else 0.0
                msg = (
                    f"epoch={epoch} iter={i+1}/{iters} lr={lr:.8f} "
                    f"loss_total={loss_value:.4f} "
                    f"loss_classifier={loss_dict.get('loss_classifier', torch.tensor(0.)).item():.4f} "
                    f"loss_box_reg={loss_dict.get('loss_box_reg', torch.tensor(0.)).item():.4f} "
                    f"loss_objectness={loss_dict.get('loss_objectness', torch.tensor(0.)).item():.4f} "
                    f"loss_rpn_box_reg={loss_dict.get('loss_rpn_box_reg', torch.tensor(0.)).item():.4f} "
                    f"data_time={data_time:.3f}s iter_time={iter_time:.3f}s eta={eta_sec/60:.1f}m gpu_memory={gpu_mem:.1f}MB"
                )
                self.logger.info(msg)

            global_step = epoch * iters + i
            self.logger.log_scalars(
                "train_iter",
                {
                    "loss_total": loss_value,
                    "loss_classifier": float(loss_dict.get("loss_classifier", torch.tensor(0.0)).item()),
                    "loss_box_reg": float(loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()),
                    "loss_objectness": float(loss_dict.get("loss_objectness", torch.tensor(0.0)).item()),
                    "loss_rpn_box_reg": float(loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()),
                    "lr": lr,
                },
                global_step,
            )

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
        self.logger.info(f"epoch={epoch} summary={avg}")
        self.logger.log_scalars("train_epoch", avg, epoch)
        return avg

    def validate(self, epoch):
        if self.val_loader is None:
            self.logger.warning("validate() called but val_loader is None. Skip validation.")
            return {}
        out_dir = os.path.join(self.output_dir, "eval", f"epoch_{epoch:03d}")
        metrics, per_class_ap = self.evaluator.evaluate(self.model, self.val_loader, self.device, output_dir=out_dir)
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        self.logger.log_scalars("val", scalar_metrics, epoch)
        if per_class_ap:
            self.logger.info(f"epoch={epoch} per-class AP rows={len(per_class_ap)}")
        return metrics
