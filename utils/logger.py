import logging
import os
import traceback
from datetime import datetime

from .file_io import append_jsonl
from .tensorboard_logger import TBLogger
from .txt_logger import build_txt_logger


class ExperimentLogger:
    def __init__(self, txt_logger=None, tb_logger=None, json_log_path=None, enabled=True):
        self.txt_logger = txt_logger
        self.tb_logger = tb_logger
        self.json_log_path = json_log_path
        self.enabled = enabled
        self._console = logging.getLogger("experiment_console")
        self._console.setLevel(logging.INFO)
        self._console.propagate = False
        if not self._console.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self._console.addHandler(sh)

    def _log(self, level: str, msg: str):
        if not self.enabled:
            return
        logger = self.txt_logger or self._console
        if hasattr(logger, level):
            getattr(logger, level)(msg)
        else:
            logger.info(msg)
        if self.json_log_path:
            try:
                append_jsonl(
                    {
                        "time": datetime.utcnow().isoformat(),
                        "level": level.upper(),
                        "message": str(msg),
                    },
                    self.json_log_path,
                )
            except Exception as e:
                (self.txt_logger or self._console).warning(f"Failed to write JSON log record: {e}")

    def info(self, msg):
        self._log("info", msg)

    def warning(self, msg):
        self._log("warning", msg)

    def error(self, msg):
        self._log("error", msg)

    def exception(self, msg):
        self._log("error", f"{msg}\n{traceback.format_exc()}")

    def log_scalars(self, tag, values, step):
        if not self.enabled:
            return
        if not isinstance(values, dict):
            return
        if self.tb_logger:
            try:
                self.tb_logger.add_scalars(tag, values, step)
            except Exception as e:
                self.warning(f"TensorBoard log failed for tag={tag} step={step}: {e}")
        if self.json_log_path:
            try:
                append_jsonl(
                    {
                        "time": datetime.utcnow().isoformat(),
                        "type": "scalars",
                        "tag": tag,
                        "step": int(step),
                        "values": values,
                    },
                    self.json_log_path,
                )
            except Exception as e:
                self.warning(f"Failed to write scalar JSON log: {e}")

    def close(self):
        if self.tb_logger:
            self.tb_logger.close()


def build_logger(cfg, output_dir, is_main_process=True):
    txt_logger = None
    tb_logger = None
    json_log_path = None

    if is_main_process and cfg["LOG"].get("TXT", True):
        txt_path = os.path.join(output_dir, cfg["LOG"].get("TXT_FILENAME", "train.log"))
        txt_logger = build_txt_logger(txt_path)

    if is_main_process and cfg["LOG"].get("TENSORBOARD", True):
        tb_dir = os.path.join(output_dir, cfg["LOG"].get("LOG_DIR_NAME", "tb"))
        try:
            tb_logger = TBLogger(tb_dir)
        except Exception as e:
            (txt_logger or logging.getLogger("experiment_console")).warning(
                f"TensorBoard disabled due to initialization failure: {e}"
            )
            tb_logger = None

    if is_main_process and cfg["LOG"].get("JSON", False):
        json_log_path = os.path.join(output_dir, cfg["LOG"].get("JSON_LOG_FILENAME", "events.jsonl"))

    return ExperimentLogger(
        txt_logger=txt_logger,
        tb_logger=tb_logger,
        json_log_path=json_log_path,
        enabled=is_main_process,
    )
