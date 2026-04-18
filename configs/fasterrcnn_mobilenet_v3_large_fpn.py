from .base import get_cfg_defaults


def get_cfg():
    cfg = get_cfg_defaults()
    cfg["MODEL"]["NAME"] = "fasterrcnn_mobilenet_v3_large_fpn"
    cfg["RUNTIME"]["EXP_NAME"] = "frcnn_mbv3_fpn"
    cfg["DATALOADER"]["BATCH_SIZE"] = 4
    return cfg
