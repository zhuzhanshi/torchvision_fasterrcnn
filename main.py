from __future__ import annotations

import argparse
import os

import torch

from datasets.builder import build_dataloaders
from engine.evaluator import Evaluator
from engine.inferencer import Inferencer
from engine.trainer import Trainer
from models.builder import build_model, load_model_weights
from utils.config import load_config, merge_cli_args, pretty_cfg, snapshot_config
from utils.env import collect_env_info
from utils.file_io import ensure_dir
from utils.logger import build_logger
from utils.misc import now_str
from utils.seed import set_seed


def build_optimizer(cfg, model):
    params = [p for p in model.parameters() if p.requires_grad]
    opt_cfg = cfg["OPTIMIZER"]
    if opt_cfg["NAME"].lower() == "sgd":
        return torch.optim.SGD(params, lr=opt_cfg["LR"], momentum=opt_cfg["MOMENTUM"], weight_decay=opt_cfg["WEIGHT_DECAY"])
    if opt_cfg["NAME"].lower() == "adamw":
        return torch.optim.AdamW(params, lr=opt_cfg["LR"], weight_decay=opt_cfg["WEIGHT_DECAY"])
    raise ValueError(f"Unsupported optimizer: {opt_cfg['NAME']}")


def build_scheduler(cfg, optimizer):
    sch_cfg = cfg["SCHEDULER"]
    name = sch_cfg["NAME"].lower()
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sch_cfg["MILESTONES"], gamma=sch_cfg["GAMMA"])
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sch_cfg["T_MAX"], eta_min=sch_cfg["ETA_MIN"])
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {sch_cfg['NAME']}")


def parse_args(default_mode=None):
    parser = argparse.ArgumentParser("Torchvision Faster R-CNN framework")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default=default_mode or "train", choices=["train", "test", "infer"])

    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--save_vis", type=lambda x: x.lower() in ["1", "true", "yes"], default=None)
    return parser.parse_args()


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


def main(default_mode=None):
    args = parse_args(default_mode)
    cfg = load_config(args.config)
    cfg = merge_cli_args(cfg, args)

    out_dir = build_output_dir(cfg)
    snapshot_config(args.config, out_dir)
    logger = build_logger(cfg, out_dir)
    log_meta(logger, cfg, args, out_dir)

    set_seed(cfg["RUNTIME"].get("SEED", 42))
    device = torch.device(cfg["RUNTIME"].get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(cfg).to(device)
    if args.mode in ["test", "infer"] and cfg["MODEL"].get("CUSTOM_WEIGHTS"):
        load_model_weights(model, cfg["MODEL"]["CUSTOM_WEIGHTS"], strict=False)

    if args.mode == "train":
        _, loaders = build_dataloaders(cfg, mode="train")
        optimizer = build_optimizer(cfg, model)
        scheduler = build_scheduler(cfg, optimizer)
        trainer = Trainer(cfg, model, optimizer, scheduler, loaders["train"], loaders.get("val"), device, logger, out_dir)
        trainer.train()
    elif args.mode == "test":
        _, loaders = build_dataloaders(cfg, mode="test")
        evaluator = Evaluator(cfg, logger)
        evaluator.evaluate(model, loaders["test"], device, output_dir=os.path.join(out_dir, "eval", "test"))
    elif args.mode == "infer":
        inferencer = Inferencer(cfg, logger)
        inferencer.run(model, device, output_dir=os.path.join(out_dir, "infer"))

    logger.close()


if __name__ == "__main__":
    main()
