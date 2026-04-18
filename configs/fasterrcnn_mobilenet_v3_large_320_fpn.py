from configs.base import get_cfg_defaults


def get_cfg():
    cfg = get_cfg_defaults()
    cfg["MODEL"]["NAME"] = "fasterrcnn_mobilenet_v3_large_320_fpn"
    cfg["RUNTIME"]["EXP_NAME"] = "frcnn_mbv3_320_fpn"
    cfg["INPUT"]["MIN_SIZE"] = 320
    cfg["INPUT"]["MAX_SIZE"] = 640
    cfg["DATALOADER"]["BATCH_SIZE"] = 6
    return cfg
