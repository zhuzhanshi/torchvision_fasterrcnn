from configs.base import get_cfg_defaults


def get_cfg():
    cfg = get_cfg_defaults()
    cfg["RUNTIME"]["EXP_NAME"] = "frcnn_mbv3_320_fpn"
    cfg["MODEL"]["NAME"] = "fasterrcnn_mobilenet_v3_large_320_fpn"
    cfg["MODEL"]["PRETRAINED"] = True
    cfg["MODEL"]["WEIGHTS"] = "DEFAULT"
    cfg["MODEL"]["WEIGHTS_BACKBONE"] = "DEFAULT"
    cfg["MODEL"]["TRAINABLE_BACKBONE_LAYERS"] = 6
    cfg["INPUT"]["MIN_SIZE"] = 320
    cfg["INPUT"]["MAX_SIZE"] = 640
    cfg["MODEL"]["MIN_SIZE"] = 320
    cfg["MODEL"]["MAX_SIZE"] = 640
    cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = 6
    cfg["DATALOADER"]["VAL_BATCH_SIZE"] = 6
    cfg["DATALOADER"]["TEST_BATCH_SIZE"] = 6
    cfg["DATALOADER"]["BATCH_SIZE"] = 6
    return cfg
