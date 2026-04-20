from __future__ import annotations

import argparse

from utils.config import load_config, merge_cli_args


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


def run(default_mode=None):
    args = parse_args(default_mode)
    cfg = merge_cli_args(load_config(args.config), args)
    from engine.runner import build_runtime, run_infer, run_test, run_train

    ctx = build_runtime(cfg, args)

    try:
        if args.mode == "train":
            run_train(ctx)
        elif args.mode == "test":
            run_test(ctx)
        elif args.mode == "infer":
            run_infer(ctx)
    finally:
        ctx.logger.close()


def main(default_mode=None):
    run(default_mode=default_mode)


if __name__ == "__main__":
    main()
