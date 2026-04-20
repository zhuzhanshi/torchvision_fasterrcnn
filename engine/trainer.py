from __future__ import annotations

import os
import time
from collections import defaultdict

import torch
from torch.cuda.amp import GradScaler, autocast

from .evaluator import Evaluator
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.misc import to_device


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
        self.scaler = GradScaler(enabled=cfg["RUNTIME"].get("USE_AMP", cfg["TRAIN"].get("AMP", True)) and device.type == "cuda")
        self.best_metric = -1.0
        self.start_epoch = 0
        self.evaluator = Evaluator(cfg, logger=logger)

        resume_path = cfg["RUNTIME"].get("RESUME", cfg["TRAIN"].get("RESUME", ""))
        if resume_path:
            self.resume(resume_path)

    def resume(self, path):
        ckpt = load_checkpoint(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") and self.scaler is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_metric = ckpt.get("best_metric", -1.0)
        self.logger.info(f"Resumed training from {path}, epoch={self.start_epoch}")

    def save_checkpoint(self, epoch, is_best=False):
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": epoch,
            "best_metric": self.best_metric,
            "cfg": {
                "model": self.cfg["MODEL"]["NAME"],
                "num_classes": self.cfg["DATASET"]["NUM_CLASSES"],
                "epochs": self.cfg["TRAIN"]["EPOCHS"],
            },
        }
        save_checkpoint(os.path.join(ckpt_dir, "latest.pth"), state)
        if is_best:
            save_checkpoint(os.path.join(ckpt_dir, "best.pth"), state)

    def train(self):
        total_epochs = self.cfg["TRAIN"]["EPOCHS"]
        for epoch in range(self.start_epoch, total_epochs):
            self.train_one_epoch(epoch)
            do_val = self.val_loader is not None and (epoch + 1) % self.cfg["TRAIN"].get("VALIDATE_EVERY_EPOCH", self.cfg["TRAIN"].get("VAL_EVERY_EPOCH", 1)) == 0
            metric_value = None
            if do_val:
                metrics = self.validate(epoch)
                metric_name = self.cfg["TRAIN"].get("SAVE_BEST_METRIC", "mAP")
                metric_value = metrics.get(metric_name, 0.0)
                if metric_value > self.best_metric:
                    self.best_metric = metric_value
                    self.save_checkpoint(epoch, is_best=True)
            self.save_checkpoint(epoch, is_best=False)
            if self.scheduler:
                self.scheduler.step()

    def train_one_epoch(self, epoch):
        self.model.train()
        meter = defaultdict(float)
        iters = len(self.train_loader)
        accum = max(1, self.cfg["TRAIN"].get("ACCUMULATION_STEPS", 1))
        grad_clip = self.cfg["TRAIN"].get("GRAD_CLIP", self.cfg["TRAIN"].get("GRAD_CLIP_NORM", 0.0))

        self.optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        last_time = time.time()
        for i, (images, targets) in enumerate(self.train_loader):
            data_time = time.time() - last_time
            images, targets = to_device(images, targets, self.device)

            with autocast(enabled=self.scaler.is_enabled()):
                loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values()) / accum

            self.scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                if grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            iter_time = time.time() - last_time
            last_time = time.time()

            meter["loss_total"] += float(loss.item() * accum)
            for k, v in loss_dict.items():
                meter[k] += float(v.item())

            if (i + 1) % self.cfg["RUNTIME"].get("PRINT_FREQ", self.cfg["TRAIN"].get("PRINT_FREQ", 20)) == 0:
                avg_total = meter["loss_total"] / (i + 1)
                lr = self.optimizer.param_groups[0]["lr"]
                eta_sec = (iters - i - 1) * (time.time() - epoch_start) / max(1, i + 1)
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if self.device.type == "cuda" else 0
                msg = (
                    f"epoch={epoch} iter={i+1}/{iters} lr={lr:.6f} loss_total={avg_total:.4f} "
                    f"loss_classifier={loss_dict.get('loss_classifier', torch.tensor(0.)).item():.4f} "
                    f"loss_box_reg={loss_dict.get('loss_box_reg', torch.tensor(0.)).item():.4f} "
                    f"loss_objectness={loss_dict.get('loss_objectness', torch.tensor(0.)).item():.4f} "
                    f"loss_rpn_box_reg={loss_dict.get('loss_rpn_box_reg', torch.tensor(0.)).item():.4f} "
                    f"data_time={data_time:.3f}s iter_time={iter_time:.3f}s eta={eta_sec/60:.1f}m gpu_mem={gpu_mem:.1f}MB"
                )
                self.logger.info(msg)

            global_step = epoch * iters + i
            self.logger.log_scalars("train_iter", {k: float(v.item()) for k, v in loss_dict.items()}, global_step)
            self.logger.log_scalars("train_iter", {"loss_total": float(loss.item() * accum), "lr": self.optimizer.param_groups[0]["lr"]}, global_step)

        epoch_time = time.time() - epoch_start
        avg = {k: v / iters for k, v in meter.items()}
        avg["epoch_time"] = epoch_time
        avg["current_lr"] = self.optimizer.param_groups[0]["lr"]
        avg["best_metric_so_far"] = self.best_metric
        self.logger.info(f"epoch={epoch} summary={avg}")
        self.logger.log_scalars("train_epoch", avg, epoch)

    def validate(self, epoch):
        out_dir = os.path.join(self.output_dir, "eval", f"epoch_{epoch:03d}")
        metrics, per_class = self.evaluator.evaluate(self.model, self.val_loader, self.device, output_dir=out_dir)
        self.logger.log_scalars("val", metrics, epoch)
        return metrics
