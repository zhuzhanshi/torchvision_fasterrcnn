import os

from .tensorboard_logger import TBLogger
from .txt_logger import build_txt_logger


class ExperimentLogger:
    def __init__(self, txt_logger=None, tb_logger=None):
        self.txt_logger = txt_logger
        self.tb_logger = tb_logger

    def info(self, msg):
        if self.txt_logger:
            self.txt_logger.info(msg)

    def log_scalars(self, tag, values, step):
        if self.tb_logger:
            self.tb_logger.add_scalars(tag, values, step)

    def close(self):
        if self.tb_logger:
            self.tb_logger.close()


def build_logger(cfg, output_dir):
    txt_logger = None
    tb_logger = None
    if cfg["LOG"].get("TXT", True):
        txt_logger = build_txt_logger(os.path.join(output_dir, "train.log"))
    if cfg["LOG"].get("TENSORBOARD", True):
        tb_dir = os.path.join(output_dir, cfg["LOG"].get("LOG_DIR_NAME", "tb"))
        tb_logger = TBLogger(tb_dir)
    return ExperimentLogger(txt_logger, tb_logger)
