from configs.base import get_cfg_defaults


def get_cfg():
    cfg = get_cfg_defaults()
    cfg["MODEL"]["NAME"] = "fasterrcnn_resnet50_fpn"
    cfg["RUNTIME"]["EXP_NAME"] = "frcnn_r50_fpn"
    return cfg
