from __future__ import annotations

import argparse

from utils.config import load_config, merge_cli_args, resolve_mode


def _str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args(default_mode=None):
    parser = argparse.ArgumentParser("Torchvision Faster R-CNN framework")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default=None, choices=["train", "test", "infer"])

    parser.add_argument("--data-root", "--data_root", dest="data_root", type=str, default=None)
    parser.add_argument("--output-root", "--output_root", dest="output_root", type=str, default=None)
    parser.add_argument("--exp-name", "--exp_name", dest="exp_name", type=str, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--input-path", "--input_path", dest="input_path", type=str, default=None)
    parser.add_argument("--save-vis", "--save_vis", dest="save_vis", type=_str2bool, default=None)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--amp", type=_str2bool, default=None)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)

    args = parser.parse_args()
    args.default_mode = default_mode
    return args


def run(default_mode=None):
    args = parse_args(default_mode)
    cfg = load_config(args.config)

    resolved_mode = resolve_mode(cfg, cli_mode=args.mode, default_mode=default_mode)
    args.mode = resolved_mode
    cfg["RUNTIME"]["MODE"] = resolved_mode

    cfg = merge_cli_args(cfg, args)

    from engine.runner import build_runtime, run_infer, run_test, run_train

    ctx = build_runtime(cfg, args)

    try:
        if resolved_mode == "train":
            run_train(ctx)
        elif resolved_mode == "test":
            run_test(ctx)
        elif resolved_mode == "infer":
            run_infer(ctx)
        else:
            raise ValueError(f"Unsupported mode: {resolved_mode}")
    finally:
        ctx.logger.close()


def main(default_mode=None):
    run(default_mode=default_mode)


if __name__ == "__main__":
    main()
