from __future__ import annotations

import os
from dataclasses import dataclass

from datasets.builder import build_dataloaders
from models.builder import build_model, load_model_weights
from utils.config import pretty_cfg, snapshot_config
from utils.env import collect_env_info
from utils.file_io import ensure_dir
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


def build_optimizer(cfg, model):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    opt_cfg = cfg["OPTIMIZER"]
    if opt_cfg["NAME"].lower() == "sgd":
        return torch.optim.SGD(params, lr=opt_cfg["LR"], momentum=opt_cfg["MOMENTUM"], weight_decay=opt_cfg["WEIGHT_DECAY"])
    if opt_cfg["NAME"].lower() == "adamw":
        return torch.optim.AdamW(params, lr=opt_cfg["LR"], weight_decay=opt_cfg["WEIGHT_DECAY"])
    raise ValueError(f"Unsupported optimizer: {opt_cfg['NAME']}")


def build_scheduler(cfg, optimizer):
    import torch

    sch_cfg = cfg["SCHEDULER"]
    name = sch_cfg["NAME"].lower()
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sch_cfg["MILESTONES"], gamma=sch_cfg["GAMMA"])
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sch_cfg["T_MAX"], eta_min=sch_cfg["ETA_MIN"])
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {sch_cfg['NAME']}")


def build_output_dir(cfg):
    root = cfg["RUNTIME"]["OUTPUT_ROOT"]
    model_name = cfg["MODEL"]["NAME"]
    exp_name = cfg["RUNTIME"]["EXP_NAME"]
    timestamp = now_str()
    out_dir = os.path.join(root, model_name, exp_name, timestamp)
    ensure_dir(out_dir)
    for d in ["checkpoints", "eval", "infer", "tb"]:
        ensure_dir(os.path.join(out_dir, d))
    return out_dir


def log_meta(logger, cfg, args, output_dir):
    env = collect_env_info(cfg["RUNTIME"]["DEVICE"])
    logger.info(f"Start time={now_str()} config={args.config} output_dir={output_dir}")
    logger.info(f"Env: {env}")
    logger.info(
        f"Dataset type={cfg['DATASET']['TYPE']} num_classes={cfg['DATASET']['NUM_CLASSES']} classes={cfg['DATASET']['CLASSES']}"
    )
    logger.info(
        f"Model={cfg['MODEL']['NAME']} weights={cfg['MODEL'].get('WEIGHTS')} batch_size={cfg['DATALOADER']['BATCH_SIZE']} "
        f"epochs={cfg['TRAIN']['EPOCHS']} optimizer={cfg['OPTIMIZER']['NAME']} scheduler={cfg['SCHEDULER']['NAME']}"
    )
    logger.info("Config dump:\n" + pretty_cfg(cfg))


def build_runtime(cfg, args):
    import torch

    output_dir = build_output_dir(cfg)
    snapshot_config(args.config, output_dir)
    logger = build_logger(cfg, output_dir)
    log_meta(logger, cfg, args, output_dir)

    set_seed(cfg["RUNTIME"].get("SEED", 42))
    device = torch.device(cfg["RUNTIME"].get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(cfg).to(device)
    if args.mode in ["test", "infer"] and cfg["MODEL"].get("CUSTOM_WEIGHTS"):
        load_model_weights(model, cfg["MODEL"]["CUSTOM_WEIGHTS"], strict=False)

    return RuntimeContext(cfg=cfg, args=args, output_dir=output_dir, logger=logger, device=device, model=model)


def run_train(ctx: RuntimeContext):
    _, loaders = build_dataloaders(ctx.cfg, mode="train")
    optimizer = build_optimizer(ctx.cfg, ctx.model)
    scheduler = build_scheduler(ctx.cfg, optimizer)
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
    evaluator.evaluate(ctx.model, loaders["test"], ctx.device, output_dir=os.path.join(ctx.output_dir, "eval", "test"))


def run_infer(ctx: RuntimeContext):
    inferencer = Inferencer(ctx.cfg, ctx.logger)
    inferencer.run(ctx.model, ctx.device, output_dir=os.path.join(ctx.output_dir, "infer"))
