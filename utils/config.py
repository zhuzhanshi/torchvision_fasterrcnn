from __future__ import annotations

import importlib.util
import os
import pprint
from copy import deepcopy


def _load_py_config(path: str):
    spec = importlib.util.spec_from_file_location("cfg_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "get_cfg"):
        return module.get_cfg()
    if hasattr(module, "CFG"):
        return deepcopy(module.CFG)
    raise ValueError(f"Config file {path} must define get_cfg() or CFG")


def deep_update(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = deepcopy(v)
    return dst


def normalize_cfg(cfg: dict):
    # Dataset aliases
    ds = cfg.get("DATASET", {})
    if "ROOT" not in ds and "DATA_ROOT" in ds:
        ds["ROOT"] = ds["DATA_ROOT"]
    if "DATA_ROOT" not in ds and "ROOT" in ds:
        ds["DATA_ROOT"] = ds["ROOT"]
    ds["ALLOW_EMPTY"] = ds.get("ALLOW_EMPTY", not ds.get("FILTER_EMPTY_GT", False))

    # Input aliases
    inp = cfg.get("INPUT", {})
    if "IMAGE_MEAN" not in inp and "MEAN" in inp:
        inp["IMAGE_MEAN"] = deepcopy(inp["MEAN"])
    if "IMAGE_STD" not in inp and "STD" in inp:
        inp["IMAGE_STD"] = deepcopy(inp["STD"])
    inp["MEAN"] = deepcopy(inp.get("IMAGE_MEAN", inp.get("MEAN", [0.485, 0.456, 0.406])))
    inp["STD"] = deepcopy(inp.get("IMAGE_STD", inp.get("STD", [0.229, 0.224, 0.225])))

    # Aug aliases
    aug = cfg.get("AUG", {})
    if "TRAIN" in aug:
        aug["HFLIP_PROB"] = aug["TRAIN"].get("HFLIP_PROB", aug.get("HFLIP_PROB", 0.0))
        aug["VFLIP_PROB"] = aug["TRAIN"].get("VFLIP_PROB", aug.get("VFLIP_PROB", 0.0))
        aug["COLOR_JITTER"] = deepcopy(aug["TRAIN"].get("COLOR_JITTER", aug.get("COLOR_JITTER", {})))
        aug["RANDOM_RESIZE"] = deepcopy(aug["TRAIN"].get("RANDOM_RESIZE", aug.get("RANDOM_RESIZE", {})))

    # Dataloader aliases
    dl = cfg.get("DATALOADER", {})
    if "BATCH_SIZE" in dl:
        bs = dl["BATCH_SIZE"]
        dl.setdefault("TRAIN_BATCH_SIZE", bs)
        dl.setdefault("VAL_BATCH_SIZE", bs)
        dl.setdefault("TEST_BATCH_SIZE", bs)
    else:
        dl["BATCH_SIZE"] = dl.get("TRAIN_BATCH_SIZE", 1)

    # Runtime/Train AMP + resume alias
    runtime = cfg.get("RUNTIME", {})
    if "DEVICE" in runtime and runtime["DEVICE"] is not None:
        runtime["DEVICE"] = str(runtime["DEVICE"]).lower()
    train = cfg.get("TRAIN", {})
    if "USE_AMP" not in runtime and "AMP" in train:
        runtime["USE_AMP"] = bool(train["AMP"])
    train["AMP"] = bool(runtime.get("USE_AMP", train.get("AMP", False)))
    if runtime.get("RESUME") and not train.get("RESUME"):
        train["RESUME"] = runtime["RESUME"]
    if train.get("RESUME") and not runtime.get("RESUME"):
        runtime["RESUME"] = train["RESUME"]
    train["PRINT_FREQ"] = runtime.get("PRINT_FREQ", train.get("PRINT_FREQ", 20))
    train["VAL_EVERY_EPOCH"] = train.get("VALIDATE_EVERY_EPOCH", train.get("VAL_EVERY_EPOCH", 1))
    grad_clip_cfg = train.get("GRAD_CLIP", 0.0)
    if isinstance(grad_clip_cfg, dict):
        train["GRAD_CLIP_NORM"] = float(grad_clip_cfg.get("MAX_NORM", 0.0))
        if train["GRAD_CLIP_NORM"] > 0 and "ENABLE" not in grad_clip_cfg:
            grad_clip_cfg["ENABLE"] = True
        train["GRAD_CLIP"] = grad_clip_cfg
    else:
        train["GRAD_CLIP_NORM"] = float(grad_clip_cfg if grad_clip_cfg is not None else train.get("GRAD_CLIP_NORM", 0.0))

    # Eval aliases
    eval_cfg = cfg.get("EVAL", {})
    eval_cfg["ENABLED"] = eval_cfg.get("ENABLE", eval_cfg.get("ENABLED", True))

    # Weights/resume routing
    model = cfg.get("MODEL", {})
    runtime_weights = runtime.get("WEIGHTS", "")
    if runtime_weights and not model.get("CUSTOM_WEIGHTS"):
        model["CUSTOM_WEIGHTS"] = runtime_weights
    if model.get("CUSTOM_WEIGHTS") and not runtime.get("WEIGHTS"):
        runtime["WEIGHTS"] = model["CUSTOM_WEIGHTS"]

    # NUM_CLASSES convention: foreground only
    if "NUM_CLASSES" not in model:
        model["NUM_CLASSES"] = ds.get("NUM_CLASSES", 1)
    if "NUM_CLASSES" in ds and model.get("NUM_CLASSES") != ds["NUM_CLASSES"]:
        model["NUM_CLASSES"] = ds["NUM_CLASSES"]

    cfg["DATASET"] = ds
    cfg["INPUT"] = inp
    cfg["AUG"] = aug
    cfg["DATALOADER"] = dl
    cfg["RUNTIME"] = runtime
    cfg["TRAIN"] = train
    cfg["EVAL"] = eval_cfg
    cfg["MODEL"] = model
    return cfg


def load_config(path: str):
    cfg = _load_py_config(path)
    return normalize_cfg(cfg)


def merge_cli_args(cfg, args):
    # mode resolved in main, but allow direct override if present
    if getattr(args, "mode", None):
        cfg["RUNTIME"]["MODE"] = args.mode

    if getattr(args, "data_root", None):
        cfg["DATASET"]["ROOT"] = args.data_root
        cfg["DATASET"]["DATA_ROOT"] = args.data_root

    if getattr(args, "output_root", None):
        cfg["RUNTIME"]["OUTPUT_ROOT"] = args.output_root

    if getattr(args, "exp_name", None):
        cfg["RUNTIME"]["EXP_NAME"] = args.exp_name

    if getattr(args, "batch_size", None) is not None:
        cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = args.batch_size
        cfg["DATALOADER"]["VAL_BATCH_SIZE"] = args.batch_size
        cfg["DATALOADER"]["TEST_BATCH_SIZE"] = args.batch_size
        cfg["DATALOADER"]["BATCH_SIZE"] = args.batch_size

    if getattr(args, "epochs", None) is not None:
        cfg["TRAIN"]["EPOCHS"] = args.epochs

    if getattr(args, "lr", None) is not None:
        cfg["OPTIMIZER"]["LR"] = args.lr

    if getattr(args, "resume", None):
        cfg["RUNTIME"]["RESUME"] = args.resume
        cfg["TRAIN"]["RESUME"] = args.resume

    if getattr(args, "weights", None):
        cfg["RUNTIME"]["WEIGHTS"] = args.weights
        cfg["MODEL"]["CUSTOM_WEIGHTS"] = args.weights

    if getattr(args, "device", None):
        cfg["RUNTIME"]["DEVICE"] = args.device

    if getattr(args, "input_path", None):
        cfg["INFER"]["INPUT_PATH"] = args.input_path

    if getattr(args, "save_vis", None) is not None:
        cfg["INFER"]["SAVE_VIS"] = bool(args.save_vis)

    if getattr(args, "num_workers", None) is not None:
        cfg["RUNTIME"]["NUM_WORKERS"] = int(args.num_workers)
        cfg["DATALOADER"]["NUM_WORKERS"] = int(args.num_workers)

    if getattr(args, "amp", None) is not None:
        cfg["RUNTIME"]["USE_AMP"] = bool(args.amp)
        cfg["TRAIN"]["AMP"] = bool(args.amp)

    return normalize_cfg(cfg)


def resolve_mode(cfg, cli_mode=None, default_mode=None):
    if cli_mode:
        return cli_mode
    if default_mode:
        return default_mode
    return cfg.get("RUNTIME", {}).get("MODE", "train")


def snapshot_config(cfg: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, "config_snapshot.py")
    dumped = pprint.pformat(cfg, sort_dicts=False, width=120)
    content = (
        "# Auto-generated snapshot of the final merged config.\n"
        "# Includes CLI overrides and runtime-normalized compatibility fields.\n\n"
        f"CFG = {dumped}\n\n"
        "def get_cfg():\n"
        "    from copy import deepcopy\n"
        "    return deepcopy(CFG)\n"
    )
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    return target


def pretty_cfg(cfg):
    return pprint.pformat(cfg, sort_dicts=False)
