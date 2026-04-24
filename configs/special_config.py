from copy import deepcopy

from configs.base import CFG as BASE_CFG


CFG = deepcopy(BASE_CFG)

# This config is migrated from an MMDetection-style Faster R-CNN setup.
# Current project is based on torchvision Faster R-CNN (not MMDetection).
# This file is a semantic approximation, not a structural 1:1 replication.
# DynamicRoIHead / DCN / Shared2FCBBoxHead / DeltaXYWHBBoxCoder and other
# MMDetection advanced modules are not implemented in this framework.
# Priority in this config is to align data/classes/anchors/NMS/input-size/
# optimizer/epochs/infer thresholds/NPU training strategy with current capabilities.
# Approximation notes:
# - torchvision uses min_size/max_size with aspect-ratio-preserving resize;
#   it cannot exactly replicate MMDet keep_ratio=False fixed resize behavior.
# - Pad(size_divisor=32) is not explicitly required in torchvision detection list[Tensor] pipeline.
# - MMDet iter-based warmup is currently not enabled in this config.

# ----------------------------------------------------------------------
# Runtime / distributed / NPU
# ----------------------------------------------------------------------
CFG["RUNTIME"]["EXP_NAME"] = "special_fasterrcnn_resnet50_fpn_38cls_npu8"
CFG["RUNTIME"]["DEVICE"] = "npu"
CFG["RUNTIME"]["USE_AMP"] = False
CFG["RUNTIME"]["NUM_WORKERS"] = 4
CFG["RUNTIME"]["PRINT_FREQ"] = 50

CFG["RUNTIME"]["DISTRIBUTED"] = True
CFG["RUNTIME"]["DIST_BACKEND"] = "hccl"
CFG["RUNTIME"]["DIST_URL"] = "env://"
CFG["RUNTIME"]["WORLD_SIZE"] = 1
CFG["RUNTIME"]["RANK"] = 0
CFG["RUNTIME"]["LOCAL_RANK"] = 0
# NOTE: WORLD_SIZE/RANK/LOCAL_RANK are expected to be overridden by torchrun env vars.

CFG["RUNTIME"]["AUTO_RESUME"] = False
CFG["RUNTIME"]["RESUME"] = ""
CFG["RUNTIME"]["WEIGHTS"] = ""

# ----------------------------------------------------------------------
# Dataset: COCO format
# ----------------------------------------------------------------------
CFG["DATASET"]["TYPE"] = "coco"
CFG["DATASET"]["ROOT"] = "/root/work/works/application/coco"
CFG["DATASET"]["TRAIN_SPLIT"] = "train"
CFG["DATASET"]["VAL_SPLIT"] = "val"
CFG["DATASET"]["TEST_SPLIT"] = "test"

classes = [
    "G17DK",
    "biubiu",
    "cc",
    "celiang",
    "cl_bx",
    "ctfs",
    "diban_fanbian",
    "diban_louhuo",
    "diban_posun",
    "diban_xiufu",
    "fm",
    "fpj",
    "gg",
    "gjdfs",
    "guan",
    "hdgd",
    "hg_bx",
    "hjf",
    "hulan",
    "huowu",
    "hwzj",
    "hwzjyiwu",
    "jiaodeng",
    "jqb_ps",
    "liang",
    "mu_diban_posun",
    "mubang",
    "pb",
    "sbzz",
    "shoulun",
    "small_wood",
    "tie",
    "tiegou",
    "tiepian",
    "tss",
    "wenzi",
    "yiwu",
    "zlkyiwu",
]
CFG["DATASET"]["CLASSES"] = classes
CFG["DATASET"]["NUM_CLASSES"] = 38
CFG["MODEL"]["NUM_CLASSES"] = 38

CFG["DATASET"]["FILTER_EMPTY_GT"] = True
CFG["DATASET"]["MIN_BOX_AREA"] = 1.0
CFG["DATASET"]["IGNORE_DIFFICULT"] = False

CFG["DATASET"]["STATS_BEFORE_TRAIN"] = True
CFG["DATASET"]["SAVE_STATS"] = True
CFG["DATASET"]["STATS_SPLITS"] = ["train", "val", "test"]
CFG["DATASET"]["STATS_BBOX_DISTRIBUTION"] = False

# ----------------------------------------------------------------------
# Model: torchvision Faster R-CNN ResNet50-FPN
# ----------------------------------------------------------------------
CFG["MODEL"]["NAME"] = "fasterrcnn_resnet50_fpn"
CFG["MODEL"]["NUM_CLASSES"] = 38
CFG["MODEL"]["PRETRAINED"] = True
CFG["MODEL"]["WEIGHTS"] = "DEFAULT"
CFG["MODEL"]["WEIGHTS_BACKBONE"] = "DEFAULT"

CFG["MODEL"]["FREEZE_BACKBONE"] = False
CFG["MODEL"]["FREEZE_BACKBONE_AT"] = 1
CFG["MODEL"]["TRAINABLE_BACKBONE_LAYERS"] = 3

CFG["MODEL"]["MIN_SIZE"] = 512
CFG["MODEL"]["MAX_SIZE"] = 640
CFG["INPUT"]["MIN_SIZE"] = 512
CFG["INPUT"]["MAX_SIZE"] = 640

CFG["MODEL"]["IMAGE_MEAN"] = [0.485, 0.456, 0.406]
CFG["MODEL"]["IMAGE_STD"] = [0.229, 0.224, 0.225]
CFG["INPUT"]["IMAGE_MEAN"] = [0.485, 0.456, 0.406]
CFG["INPUT"]["IMAGE_STD"] = [0.229, 0.224, 0.225]

# ----------------------------------------------------------------------
# RPN / anchors
# ----------------------------------------------------------------------
CFG["MODEL"]["RPN"]["USE_CUSTOM"] = True
CFG["MODEL"]["RPN"]["ANCHOR_SIZES"] = ((32,), (64,), (128,), (256,), (512,))
ratios = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
CFG["MODEL"]["RPN"]["ASPECT_RATIOS"] = (ratios, ratios, ratios, ratios, ratios)

CFG["MODEL"]["RPN"]["FG_IOU_THRESH"] = 0.7
CFG["MODEL"]["RPN"]["BG_IOU_THRESH"] = 0.3
CFG["MODEL"]["RPN"]["BATCH_SIZE_PER_IMAGE"] = 256
CFG["MODEL"]["RPN"]["POSITIVE_FRACTION"] = 0.5

CFG["MODEL"]["RPN"]["PRE_NMS_TOP_N_TRAIN"] = 2000
CFG["MODEL"]["RPN"]["POST_NMS_TOP_N_TRAIN"] = 1000
CFG["MODEL"]["RPN"]["PRE_NMS_TOP_N_TEST"] = 500
CFG["MODEL"]["RPN"]["POST_NMS_TOP_N_TEST"] = 500
CFG["MODEL"]["RPN"]["NMS_THRESH"] = 0.85

# ----------------------------------------------------------------------
# ROI heads
# ----------------------------------------------------------------------
CFG["MODEL"]["ROI_HEADS"]["BOX_FG_IOU_THRESH"] = 0.5
CFG["MODEL"]["ROI_HEADS"]["BOX_BG_IOU_THRESH"] = 0.5
CFG["MODEL"]["ROI_HEADS"]["BATCH_SIZE_PER_IMAGE"] = 512
CFG["MODEL"]["ROI_HEADS"]["POSITIVE_FRACTION"] = 0.25

CFG["MODEL"]["ROI_HEADS"]["BOX_SCORE_THRESH"] = 0.3
CFG["MODEL"]["ROI_HEADS"]["BOX_NMS_THRESH"] = 0.5
CFG["MODEL"]["ROI_HEADS"]["BOX_DETECTIONS_PER_IMG"] = 100

# ----------------------------------------------------------------------
# Train pipeline approximation
# ----------------------------------------------------------------------
CFG["AUG"]["TRAIN"]["ENABLE"] = True
CFG["AUG"]["TRAIN"]["HFLIP_PROB"] = 0.5
CFG["AUG"]["TRAIN"]["VFLIP_PROB"] = 0.0
CFG["AUG"]["TRAIN"]["COLOR_JITTER"]["ENABLED"] = False
CFG["AUG"]["TRAIN"]["RANDOM_RESIZE"]["ENABLED"] = False
CFG["AUG"]["TRAIN"]["RANDOM_CROP"]["ENABLED"] = False
CFG["AUG"]["TRAIN"]["RANDOM_ROTATE"]["ENABLED"] = False

# ----------------------------------------------------------------------
# Dataloader (for torchrun --nproc_per_node=8: per-rank batch=8)
# ----------------------------------------------------------------------
CFG["DATALOADER"]["TRAIN_BATCH_SIZE"] = 8
CFG["DATALOADER"]["VAL_BATCH_SIZE"] = 1
CFG["DATALOADER"]["TEST_BATCH_SIZE"] = 1
CFG["DATALOADER"]["SHUFFLE"] = True
CFG["DATALOADER"]["DROP_LAST"] = False
CFG["RUNTIME"]["NUM_WORKERS"] = 4

# ----------------------------------------------------------------------
# Optimizer / grad clip
# ----------------------------------------------------------------------
CFG["OPTIMIZER"]["NAME"] = "AdamW"
CFG["OPTIMIZER"]["LR"] = 0.0002
CFG["OPTIMIZER"]["BETAS"] = (0.9, 0.999)
CFG["OPTIMIZER"]["WEIGHT_DECAY"] = 0.05

CFG["TRAIN"]["GRAD_CLIP"]["ENABLE"] = True
CFG["TRAIN"]["GRAD_CLIP"]["MAX_NORM"] = 35
CFG["TRAIN"]["GRAD_CLIP"]["NORM_TYPE"] = 2

# ----------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------
# MMDet policy=step + step=[22, 27] -> framework multistep (epoch-based)
CFG["SCHEDULER"]["NAME"] = "multistep"
CFG["SCHEDULER"]["MILESTONES"] = [22, 27]
CFG["SCHEDULER"]["GAMMA"] = 0.1

# MMDet iter-based warmup (warmup_iters=3000) is kept as semantic placeholder but disabled.
# If iter-based warmup is implemented in this framework later, these fields can be enabled.
CFG["SCHEDULER"]["WARMUP"]["ENABLED"] = False
CFG["SCHEDULER"]["WARMUP"]["TYPE"] = "linear"
CFG["SCHEDULER"]["WARMUP"]["ITERS"] = 3000
CFG["SCHEDULER"]["WARMUP"]["RATIO"] = 0.001

# ----------------------------------------------------------------------
# Train / eval policy
# ----------------------------------------------------------------------
CFG["TRAIN"]["EPOCHS"] = 30
CFG["TRAIN"]["ACCUMULATION_STEPS"] = 1

CFG["EVAL"]["ENABLE"] = True
CFG["EVAL"]["DURING_TRAIN"] = False
CFG["EVAL"]["AFTER_TRAIN"] = True
CFG["EVAL"]["INTERVAL"] = 1
CFG["EVAL"]["BEST_METRIC"] = "map"
CFG["EVAL"]["SCORE_THRESH"] = 0.05
CFG["EVAL"]["MAX_DETS"] = 100
CFG["EVAL"]["PER_CLASS_AP"] = True
CFG["EVAL"]["MIN_SIZE"] = 512
CFG["EVAL"]["MAX_SIZE"] = 640

# ----------------------------------------------------------------------
# Infer
# ----------------------------------------------------------------------
CFG["INFER"]["SCORE_THRESH"] = 0.3
CFG["INFER"]["NMS_THRESH"] = 0.5
CFG["INFER"]["MAX_DETS"] = 100
CFG["INFER"]["SAVE_VIS"] = True
CFG["INFER"]["SAVE_JSON"] = True
CFG["INFER"]["SAVE_TXT"] = False
CFG["INFER"]["MIN_SIZE"] = 512
CFG["INFER"]["MAX_SIZE"] = 640

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
CFG["LOG"]["TXT"] = True
CFG["LOG"]["TENSORBOARD"] = True
CFG["LOG"]["JSON"] = True
CFG["LOG"]["SAVE_CONFIG_SNAPSHOT"] = True
CFG["LOG"]["SAVE_ENV_INFO"] = True
CFG["LOG"]["LOG_ITER_LOSS"] = True
CFG["LOG"]["LOG_EPOCH_LOSS"] = True
CFG["LOG"]["LOG_LR"] = True
CFG["LOG"]["LOG_MEMORY"] = True
CFG["LOG"]["LOG_TIME"] = True


def get_cfg():
    return deepcopy(CFG)
