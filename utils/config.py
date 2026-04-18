from __future__ import annotations

import importlib.util
import os
import pprint
import shutil
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


def load_config(path: str):
    return _load_py_config(path)


def merge_cli_args(cfg, args):
    if args.mode:
        cfg["RUNTIME"]["MODE"] = args.mode
    if args.data_root:
        cfg["DATASET"]["DATA_ROOT"] = args.data_root
    if args.output_root:
        cfg["RUNTIME"]["OUTPUT_ROOT"] = args.output_root
    if args.exp_name:
        cfg["RUNTIME"]["EXP_NAME"] = args.exp_name
    if args.batch_size is not None:
        cfg["DATALOADER"]["BATCH_SIZE"] = args.batch_size
    if args.epochs is not None:
        cfg["TRAIN"]["EPOCHS"] = args.epochs
    if args.lr is not None:
        cfg["OPTIMIZER"]["LR"] = args.lr
    if args.resume:
        cfg["TRAIN"]["RESUME"] = args.resume
    if args.weights:
        cfg["MODEL"]["CUSTOM_WEIGHTS"] = args.weights
    if args.device:
        cfg["RUNTIME"]["DEVICE"] = args.device
    if args.input_path:
        cfg["INFER"]["INPUT_PATH"] = args.input_path
    if args.save_vis is not None:
        cfg["INFER"]["SAVE_VIS"] = args.save_vis
    return cfg


def snapshot_config(config_path: str, out_dir: str):
    target = os.path.join(out_dir, "config_snapshot.py")
    shutil.copy2(config_path, target)
    return target


def pretty_cfg(cfg):
    return pprint.pformat(cfg, sort_dicts=False)
