from configs.base import get_cfg_defaults


def get_cfg():
    cfg = get_cfg_defaults()
    cfg["RUNTIME"]["EXP_NAME"] = "frcnn_r50_fpn_v2"
    cfg["MODEL"]["NAME"] = "fasterrcnn_resnet50_fpn_v2"
    cfg["MODEL"]["WEIGHTS"] = "DEFAULT"
    cfg["MODEL"]["WEIGHTS_BACKBONE"] = "DEFAULT"
    cfg["MODEL"]["TRAINABLE_BACKBONE_LAYERS"] = 3
    return cfg
