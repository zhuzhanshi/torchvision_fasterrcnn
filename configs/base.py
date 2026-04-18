from copy import deepcopy


CFG = {
    "RUNTIME": {
        "MODE": "train",
        "SEED": 42,
        "DEVICE": "cuda",
        "OUTPUT_ROOT": "outputs",
        "EXP_NAME": "default_exp",
        "WORKERS": 4,
        "PIN_MEMORY": True,
    },
    "DATASET": {
        "TYPE": "voc",  # voc | coco
        "DATA_ROOT": "data",
        "TRAIN_SPLIT": "train",
        "VAL_SPLIT": "val",
        "TEST_SPLIT": "test",
        "CLASSES": ["class1"],
        "NUM_CLASSES": 1,  # foreground classes only
        "ALLOW_EMPTY": False,
    },
    "INPUT": {
        "MIN_SIZE": 800,
        "MAX_SIZE": 1333,
        "MEAN": [0.485, 0.456, 0.406],
        "STD": [0.229, 0.224, 0.225],
    },
    "AUG": {
        "HFLIP_PROB": 0.5,
        "VFLIP_PROB": 0.0,
        "COLOR_JITTER": {
            "ENABLED": False,
            "BRIGHTNESS": 0.2,
            "CONTRAST": 0.2,
            "SATURATION": 0.2,
            "HUE": 0.02,
        },
        "RANDOM_RESIZE": {
            "ENABLED": True,
            "SCALES": [640, 672, 704, 736, 768, 800],
            "MAX_SIZE": 1333,
        },
    },
    "DATALOADER": {
        "BATCH_SIZE": 2,
        "SHUFFLE": True,
        "NUM_WORKERS": 4,
        "PIN_MEMORY": True,
        "DROP_LAST": False,
    },
    "MODEL": {
        "NAME": "fasterrcnn_resnet50_fpn",
        "WEIGHTS": "DEFAULT",
        "WEIGHTS_BACKBONE": "DEFAULT",
        "TRAINABLE_BACKBONE_LAYERS": 3,
        "FREEZE_BACKBONE": False,
        "CUSTOM_WEIGHTS": "",
        "RPN": {
            "ANCHOR_SIZES": [[32], [64], [128], [256], [512]],
            "ASPECT_RATIOS": [[0.5, 1.0, 2.0]] * 5,
            "PRE_NMS_TOP_N_TRAIN": 2000,
            "PRE_NMS_TOP_N_TEST": 1000,
            "POST_NMS_TOP_N_TRAIN": 2000,
            "POST_NMS_TOP_N_TEST": 1000,
            "NMS_THRESH": 0.7,
            "FG_IOU_THRESH": 0.7,
            "BG_IOU_THRESH": 0.3,
            "BATCH_SIZE_PER_IMAGE": 256,
            "POSITIVE_FRACTION": 0.5,
        },
        "ROI_HEADS": {
            "SCORE_THRESH": 0.05,
            "NMS_THRESH": 0.5,
            "DETECTIONS_PER_IMG": 100,
            "FG_IOU_THRESH": 0.5,
            "BG_IOU_THRESH": 0.5,
            "BATCH_SIZE_PER_IMAGE": 512,
            "POSITIVE_FRACTION": 0.25,
        },
    },
    "OPTIMIZER": {
        "NAME": "sgd",
        "LR": 0.005,
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 0.0005,
    },
    "SCHEDULER": {
        "NAME": "multistep",
        "MILESTONES": [8, 11],
        "GAMMA": 0.1,
        "T_MAX": 12,
        "ETA_MIN": 1e-6,
    },
    "TRAIN": {
        "EPOCHS": 12,
        "AMP": True,
        "GRAD_CLIP_NORM": 0.0,
        "ACCUMULATION_STEPS": 1,
        "PRINT_FREQ": 20,
        "VAL_EVERY_EPOCH": 1,
        "SAVE_BEST_METRIC": "mAP",
        "RESUME": "",
    },
    "EVAL": {
        "ENABLED": True,
        "IOU_TYPE": "bbox",
    },
    "INFER": {
        "INPUT_PATH": "",
        "SCORE_THRESH": 0.5,
        "NMS_THRESH": 0.5,
        "MAX_DETS": 100,
        "CLASS_FILTER": [],
        "SAVE_VIS": True,
        "SAVE_JSON": True,
        "SAVE_TXT": False,
    },
    "LOG": {
        "TXT": True,
        "TENSORBOARD": True,
        "LOG_DIR_NAME": "tb",
    },
}


def get_cfg_defaults():
    return deepcopy(CFG)
