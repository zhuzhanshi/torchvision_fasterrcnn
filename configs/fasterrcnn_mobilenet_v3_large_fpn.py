from configs.base import get_cfg_defaults


def get_cfg():
    cfg = get_cfg_defaults()
    cfg["RUNTIME"]["EXP_NAME"] = "frcnn_mbv3_fpn"
    cfg["MODEL"]["NAME"] = "fasterrcnn_mobilenet_v3_large_fpn"
    cfg["MODEL"]["WEIGHTS"] = "DEFAULT"
    cfg["MODEL"]["WEIGHTS_BACKBONE"] = "DEFAULT"
    cfg["MODEL"]["TRAINABLE_BACKBONE_LAYERS"] = 6
    cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = 4
    cfg["DATALOADER"]["VAL_BATCH_SIZE"] = 4
    cfg["DATALOADER"]["TEST_BATCH_SIZE"] = 4
    cfg["DATALOADER"]["BATCH_SIZE"] = 4
    return cfg
