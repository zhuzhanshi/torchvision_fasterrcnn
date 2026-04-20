import logging
import os
import sys


def build_txt_logger(log_file: str):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
