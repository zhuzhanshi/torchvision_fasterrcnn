from __future__ import annotations

import os
from dataclasses import dataclass

from datasets.builder import build_dataloaders
from models.builder import build_model, load_model_weights
from utils.config import pretty_cfg, snapshot_config
from utils.env import collect_env_info, save_env_info
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
    logger.info(f"Start time={now_str()} config={args.config} output_dir={output_dir}")
    env = collect_env_info(cfg["RUNTIME"]["DEVICE"])
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


def build_runtime(cfg, args):
    import torch

    output_dir = build_output_dir(cfg)
    logger = build_logger(cfg, output_dir)
    if cfg["LOG"].get("SAVE_CONFIG_SNAPSHOT", True):
        try:
            snapshot_config(cfg, output_dir)
        except Exception as e:
            logger.warning(f"Failed to write config snapshot: {e}")
    log_meta(logger, cfg, args, output_dir)

    set_seed(cfg["RUNTIME"].get("SEED", 42))
    _configure_torch_runtime(cfg)
    device = torch.device(cfg["RUNTIME"].get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(cfg).to(device)
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

    return RuntimeContext(cfg=cfg, args=args, output_dir=output_dir, logger=logger, device=device, model=model)


def run_train(ctx: RuntimeContext):
    if ctx.cfg["RUNTIME"].get("EVAL_BEFORE_TRAIN", False):
        run_test(ctx)
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
    inferencer = Inferencer(ctx.cfg, ctx.logger)
    out_dir = ctx.cfg["INFER"].get("OUTPUT_DIR") or os.path.join(ctx.output_dir, "infer")
    inferencer.run(ctx.model, ctx.device, output_dir=out_dir)
