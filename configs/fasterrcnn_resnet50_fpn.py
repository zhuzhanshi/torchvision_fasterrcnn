from configs.base import get_cfg_defaults


def get_cfg():
    cfg = get_cfg_defaults()
    cfg["RUNTIME"]["EXP_NAME"] = "frcnn_r50_fpn"
    cfg["MODEL"]["NAME"] = "fasterrcnn_resnet50_fpn"
    cfg["MODEL"]["PRETRAINED"] = True
    cfg["MODEL"]["WEIGHTS"] = "DEFAULT"
    cfg["MODEL"]["WEIGHTS_BACKBONE"] = "DEFAULT"
    cfg["MODEL"]["TRAINABLE_BACKBONE_LAYERS"] = 3
    return cfg
