from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np

from datasets.builder import build_dataloaders, build_dataset
from models.builder import build_model, load_model_weights
from utils.config import pretty_cfg, snapshot_config
from utils.dist import barrier, get_rank, get_world_size, init_distributed_mode, is_main_process
from utils.env import collect_env_info, save_env_info
from utils.file_io import dump_json, dump_text, ensure_dir
from utils.logger import build_logger
from utils.misc import now_str
from utils.seed import set_seed

from .evaluator import Evaluator
from .inferencer import Inferencer
from .trainer import Trainer


@dataclass
class RuntimeContext:
    cfg: dict
    args: object
    output_dir: str
    logger: object
    device: object
    model: object
    distributed: bool


def _collect_dataset_stats(cfg, output_dir, logger):
    ds_cfg = cfg.get("DATASET", {})
    if not bool(ds_cfg.get("SAVE_STATS", True)):
        return

    splits = list(ds_cfg.get("STATS_SPLITS", ["train", "val", "test"]))
    if not splits:
        return
    class_names = list(ds_cfg.get("CLASSES", []))
    stats_root = os.path.join(output_dir, "dataset_stats")
    ensure_dir(stats_root)

    result = {
        "dataset_type": ds_cfg.get("TYPE", ""),
        "data_root": ds_cfg.get("ROOT", ds_cfg.get("DATA_ROOT", "")),
        "splits": {},
    }

    for split in splits:
        split_name = str(split).strip().lower()
        if split_name not in {"train", "val", "test"}:
            logger.warning(f"Skip unknown DATASET.STATS_SPLITS item={split!r}. Supported: train/val/test.")
            continue
        try:
            dataset = build_dataset(cfg, split=split_name)
        except Exception as e:
            logger.warning(f"Failed to build dataset stats for split={split_name}: {e}")
            continue

        # Avoid random train-time augmentations in stats traversal.
        if hasattr(dataset, "transforms"):
            dataset.transforms = None

        per_class_box_count = {i + 1: 0 for i in range(len(class_names))}
        per_class_image_count = {i + 1: 0 for i in range(len(class_names))}
        empty_images = 0
        box_wh = []
        box_area = []
        num_images = len(dataset)

        for idx in range(num_images):
            _, target = dataset[idx]
            boxes = target.get("boxes")
            labels = target.get("labels")
            if boxes is None or boxes.numel() == 0:
                empty_images += 1
                continue
            labels_np = labels.detach().cpu().numpy().astype(int).tolist()
            boxes_np = boxes.detach().cpu().numpy()
            present = set()
            for b, lbl in zip(boxes_np, labels_np):
                if lbl <= 0 or lbl not in per_class_box_count:
                    continue
                x1, y1, x2, y2 = [float(x) for x in b.tolist()]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                per_class_box_count[lbl] += 1
                present.add(lbl)
                box_wh.append([w, h])
                box_area.append(w * h)
            for lbl in present:
                per_class_image_count[lbl] += 1

        box_area = np.asarray(box_area, dtype=np.float64) if box_area else np.asarray([], dtype=np.float64)
        bbox_scale_distribution = {
            "small(<32^2)": int(np.sum(box_area < (32.0**2))) if box_area.size else 0,
            "medium([32^2,96^2))": int(np.sum((box_area >= (32.0**2)) & (box_area < (96.0**2)))) if box_area.size else 0,
            "large(>=96^2)": int(np.sum(box_area >= (96.0**2))) if box_area.size else 0,
        }

        per_class_rows = []
        for label_id in range(1, len(class_names) + 1):
            per_class_rows.append(
                {
                    "label_id": label_id,
                    "class_name": class_names[label_id - 1],
                    "box_count": int(per_class_box_count[label_id]),
                    "image_count": int(per_class_image_count[label_id]),
                }
            )

        split_stats = {
            "split": split_name,
            "num_images": int(num_images),
            "empty_annotation_images": int(empty_images),
            "num_boxes": int(sum(per_class_box_count.values())),
            "bbox_scale_distribution": bbox_scale_distribution,
            "per_class": per_class_rows,
        }
        result["splits"][split_name] = split_stats

    dump_json(result, os.path.join(stats_root, "dataset_stats.json"))
    # csv
    csv_rows = []
    for split_name, split_stats in result["splits"].items():
        for row in split_stats["per_class"]:
            csv_rows.append(
                {
                    "split": split_name,
                    "label_id": row["label_id"],
                    "class_name": row["class_name"],
                    "box_count": row["box_count"],
                    "image_count": row["image_count"],
                    "num_images": split_stats["num_images"],
                    "empty_annotation_images": split_stats["empty_annotation_images"],
                }
            )
    if csv_rows:
        csv_path = os.path.join(stats_root, "dataset_stats.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)

    # txt
    lines = []
    for split_name, split_stats in result["splits"].items():
        lines.append(f"[{split_name}] num_images={split_stats['num_images']} empty={split_stats['empty_annotation_images']}")
        lines.append(
            "bbox_scale_distribution: "
            + ", ".join(f"{k}={v}" for k, v in split_stats["bbox_scale_distribution"].items())
        )
        for row in split_stats["per_class"]:
            lines.append(
                f"  class={row['class_name']} label={row['label_id']} box_count={row['box_count']} image_count={row['image_count']}"
            )
    dump_text("\n".join(lines) + ("\n" if lines else ""), os.path.join(stats_root, "dataset_stats.txt"))
    logger.info(f"Dataset stats saved to {stats_root}")


def build_optimizer(cfg, model):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    opt_cfg = cfg["OPTIMIZER"]
    name = opt_cfg["NAME"].lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=opt_cfg["LR"], momentum=opt_cfg["MOMENTUM"], weight_decay=opt_cfg["WEIGHT_DECAY"])
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=opt_cfg["LR"],
            weight_decay=opt_cfg["WEIGHT_DECAY"],
            betas=tuple(opt_cfg.get("BETAS", [0.9, 0.999])),
            eps=opt_cfg.get("EPS", 1e-8),
        )
    raise ValueError(f"Unsupported optimizer: {opt_cfg['NAME']}")


def build_scheduler(cfg, optimizer):
    import torch

    sch_cfg = cfg["SCHEDULER"]
    name = sch_cfg["NAME"].lower()
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sch_cfg["MILESTONES"], gamma=sch_cfg["GAMMA"])
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sch_cfg["T_MAX"], eta_min=sch_cfg["ETA_MIN"])
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_cfg["STEP_SIZE"], gamma=sch_cfg["GAMMA"])
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {sch_cfg['NAME']}")


def build_output_dir(cfg):
    existing = cfg["RUNTIME"].get("EXISTING_OUTPUT_DIR", "")
    if existing:
        ensure_dir(existing)
        for d in ["checkpoints", "eval", "infer", cfg["LOG"].get("LOG_DIR_NAME", "tb")]:
            ensure_dir(os.path.join(existing, d))
        return existing

    root = cfg["RUNTIME"]["OUTPUT_ROOT"]
    model_name = cfg["MODEL"]["NAME"]
    exp_name = cfg["RUNTIME"]["EXP_NAME"]
    timestamp = now_str()
    out_dir = os.path.join(root, model_name, exp_name, timestamp)
    ensure_dir(out_dir)
    for d in ["checkpoints", "eval", "infer", cfg["LOG"].get("LOG_DIR_NAME", "tb")]:
        ensure_dir(os.path.join(out_dir, d))
    return out_dir


def log_meta(logger, cfg, args, output_dir):
    if not is_main_process():
        return
    logger.info(f"Start time={now_str()} config={args.config} output_dir={output_dir}")
    env = collect_env_info(cfg["RUNTIME"]["DEVICE"], runtime_info=cfg["RUNTIME"])
    if cfg["LOG"].get("SAVE_ENV_INFO", True):
        try:
            save_env_info(env, os.path.join(output_dir, cfg["LOG"].get("ENV_FILENAME", "env.txt")))
        except Exception as e:
            logger.warning(f"Failed to save env info file: {e}")
    logger.info(f"Env: {env}")
    logger.info(
        f"Dataset type={cfg['DATASET']['TYPE']} num_classes={cfg['DATASET']['NUM_CLASSES']} classes={cfg['DATASET']['CLASSES']}"
    )
    logger.info(
        f"Model={cfg['MODEL']['NAME']} weights={cfg['MODEL'].get('WEIGHTS')} train_batch={cfg['DATALOADER']['TRAIN_BATCH_SIZE']} "
        f"epochs={cfg['TRAIN']['EPOCHS']} optimizer={cfg['OPTIMIZER']['NAME']} scheduler={cfg['SCHEDULER']['NAME']} mode={cfg['RUNTIME']['MODE']}"
    )
    logger.info("Config dump:\n" + pretty_cfg(cfg))


def _configure_torch_runtime(cfg):
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(cfg["RUNTIME"].get("CUDNN_BENCHMARK", False))
        torch.backends.cudnn.deterministic = bool(cfg["RUNTIME"].get("DETERMINISTIC", False))


def _resolve_runtime_device(cfg, logger):
    import torch

    req = str(cfg["RUNTIME"].get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")).lower()
    local_rank = int(cfg["RUNTIME"].get("LOCAL_RANK", 0))

    if req.startswith("cpu"):
        device = torch.device("cpu")
    elif req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("RUNTIME.DEVICE is set to CUDA, but torch.cuda.is_available() is False.")
        device = torch.device(req if ":" in req else f"cuda:{local_rank}")
        if hasattr(torch.cuda, "set_device"):
            torch.cuda.set_device(device)
    elif req.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except Exception as e:
            raise ImportError(
                "RUNTIME.DEVICE is set to NPU but torch_npu is not available. "
                "Please install Ascend torch_npu for your CANN/PyTorch version."
            ) from e
        if not hasattr(torch, "npu"):
            raise RuntimeError("torch_npu imported, but torch.npu backend is unavailable.")
        device = torch.device(req if ":" in req else f"npu:{local_rank}")
        if hasattr(torch.npu, "set_device"):
            torch.npu.set_device(device)
        avail = torch.npu.is_available() if hasattr(torch.npu, "is_available") else True
        if not avail:
            raise RuntimeError("RUNTIME.DEVICE is set to NPU but torch.npu.is_available() is False.")
    else:
        raise ValueError(f"Unsupported RUNTIME.DEVICE={req}. Supported: cpu / cuda / npu.")

    logger.info(f"Runtime device resolved: requested={req}, actual={device}")
    return device


def build_runtime(cfg, args):
    import torch

    distributed = init_distributed_mode(cfg)

    # Build one shared output dir for all ranks in a distributed job.
    if distributed:
        if is_main_process():
            output_dir = build_output_dir(cfg)
        else:
            output_dir = ""
        shared = [output_dir]
        torch.distributed.broadcast_object_list(shared, src=0)
        output_dir = shared[0]
        for d in ["", "checkpoints", "eval", "infer", cfg["LOG"].get("LOG_DIR_NAME", "tb")]:
            ensure_dir(os.path.join(output_dir, d) if d else output_dir)
        barrier()
    else:
        output_dir = build_output_dir(cfg)

    logger = build_logger(cfg, output_dir, is_main_process=is_main_process())
    if cfg["LOG"].get("SAVE_CONFIG_SNAPSHOT", True):
        if is_main_process():
            try:
                snapshot_config(cfg, output_dir)
            except Exception as e:
                logger.warning(f"Failed to write config snapshot: {e}")
    log_meta(logger, cfg, args, output_dir)

    set_seed(cfg["RUNTIME"].get("SEED", 42))
    _configure_torch_runtime(cfg)
    device = _resolve_runtime_device(cfg, logger)

    model = build_model(cfg).to(device)
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        local_rank = int(cfg["RUNTIME"].get("LOCAL_RANK", 0))
        if device.type in {"cuda", "npu"}:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        else:
            model = DDP(model, find_unused_parameters=False)
    if cfg["LOG"].get("LOG_MODEL_STRUCTURE", False):
        logger.info(f"Model structure:\n{model}")

    if cfg["RUNTIME"].get("RESUME") and cfg["RUNTIME"].get("WEIGHTS"):
        logger.info("Both RUNTIME.RESUME and RUNTIME.WEIGHTS are set. Resume state will take precedence once Trainer resumes.")
    if cfg["RUNTIME"].get("MODE") == "infer" and cfg["RUNTIME"].get("RESUME"):
        logger.warning("RUNTIME.RESUME is ignored in infer mode. Please use RUNTIME.WEIGHTS/--weights for inference.")

    # weights: model-only load for train/test/infer
    if cfg["RUNTIME"].get("WEIGHTS"):
        if not os.path.isfile(cfg["RUNTIME"]["WEIGHTS"]):
            raise FileNotFoundError(f"RUNTIME.WEIGHTS file not found: {cfg['RUNTIME']['WEIGHTS']}")
        load_info = load_model_weights(model, cfg["RUNTIME"]["WEIGHTS"], strict=False)
        logger.info(
            f"Loaded model weights from {cfg['RUNTIME']['WEIGHTS']} (strict=False). "
            f"missing={len(load_info['missing_keys'])}, unexpected={len(load_info['unexpected_keys'])}"
        )

    logger.info(
        f"Distributed runtime: enabled={distributed} backend={cfg['RUNTIME'].get('DIST_BACKEND', '')} "
        f"world_size={get_world_size()} rank={get_rank()} local_rank={cfg['RUNTIME'].get('LOCAL_RANK', 0)}"
    )
    return RuntimeContext(
        cfg=cfg,
        args=args,
        output_dir=output_dir,
        logger=logger,
        device=device,
        model=model,
        distributed=distributed,
    )


def run_train(ctx: RuntimeContext):
    if ctx.cfg["RUNTIME"].get("EVAL_BEFORE_TRAIN", False):
        run_test(ctx)
    if is_main_process() and bool(ctx.cfg["DATASET"].get("STATS_BEFORE_TRAIN", True)):
        _collect_dataset_stats(ctx.cfg, ctx.output_dir, ctx.logger)
    barrier()
    _, loaders = build_dataloaders(ctx.cfg, mode="train")
    if len(loaders["train"]) == 0:
        raise RuntimeError("Train dataloader is empty; cannot start training.")
    optimizer = build_optimizer(ctx.cfg, ctx.model)
    scheduler = build_scheduler(ctx.cfg, optimizer)
    warmup_cfg = ctx.cfg["SCHEDULER"].get("WARMUP", {})
    if warmup_cfg.get("ENABLED", False):
        ctx.logger.warning("SCHEDULER.WARMUP is enabled but warmup logic is not implemented yet (TODO).")
    trainer = Trainer(
        ctx.cfg,
        ctx.model,
        optimizer,
        scheduler,
        loaders["train"],
        loaders.get("val"),
        ctx.device,
        ctx.logger,
        ctx.output_dir,
    )
    trainer.train()


def run_test(ctx: RuntimeContext):
    _, loaders = build_dataloaders(ctx.cfg, mode="test")
    evaluator = Evaluator(ctx.cfg, ctx.logger)
    metrics, _ = evaluator.evaluate(ctx.model, loaders["test"], ctx.device, output_dir=os.path.join(ctx.output_dir, "eval", "test"))
    ctx.logger.info(f"Test evaluation completed. metrics={metrics}")


def run_infer(ctx: RuntimeContext):
    if ctx.distributed and get_world_size() > 1:
        if is_main_process():
            ctx.logger.warning("Distributed infer is not supported in this version. Running on rank0 only.")
        if not is_main_process():
            return
    inferencer = Inferencer(ctx.cfg, ctx.logger)
    out_dir = ctx.cfg["INFER"].get("OUTPUT_DIR") or os.path.join(ctx.output_dir, "infer")
    inferencer.run(ctx.model, ctx.device, output_dir=out_dir)
