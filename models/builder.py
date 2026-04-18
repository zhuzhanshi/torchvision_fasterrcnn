from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .components import make_anchor_generator
from .faster_rcnn import MODEL_FACTORY


def build_anchor_generator(cfg: Dict):
    rpn_cfg = cfg["MODEL"]["RPN"]
    return make_anchor_generator(rpn_cfg["ANCHOR_SIZES"], rpn_cfg["ASPECT_RATIOS"])


def replace_box_predictor(model: nn.Module, num_classes: int):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def apply_freeze_strategy(model: nn.Module, cfg: Dict):
    if not cfg["MODEL"].get("FREEZE_BACKBONE", False):
        return model
    for p in model.backbone.parameters():
        p.requires_grad = False
    return model


def load_model_weights(model: nn.Module, path: str, strict: bool = False):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=strict)
    return model


def _resolve_weights(enum_cls, name):
    if not name:
        return None
    if isinstance(name, str) and name.upper() == "DEFAULT":
        return enum_cls.DEFAULT if enum_cls is not None else None
    if enum_cls is None:
        return None
    if isinstance(name, str) and hasattr(enum_cls, name):
        return getattr(enum_cls, name)
    return None


def build_model(cfg: Dict):
    name = cfg["MODEL"]["NAME"]
    if name not in MODEL_FACTORY:
        raise ValueError(f"Unsupported model: {name}")

    model_fn = MODEL_FACTORY[name]
    annotations = getattr(model_fn, "__annotations__", {})
    weights_cls = annotations.get("weights")
    weights_backbone_cls = annotations.get("weights_backbone")

    kwargs = {
        "weights": _resolve_weights(weights_cls, cfg["MODEL"].get("WEIGHTS", "DEFAULT")),
        "weights_backbone": _resolve_weights(weights_backbone_cls, cfg["MODEL"].get("WEIGHTS_BACKBONE", "DEFAULT")),
        "trainable_backbone_layers": cfg["MODEL"].get("TRAINABLE_BACKBONE_LAYERS", None),
        "rpn_anchor_generator": build_anchor_generator(cfg),
    }

    rpn = cfg["MODEL"].get("RPN", {})
    roi = cfg["MODEL"].get("ROI_HEADS", {})
    kwargs.update(
        {
            "rpn_pre_nms_top_n_train": rpn.get("PRE_NMS_TOP_N_TRAIN", 2000),
            "rpn_pre_nms_top_n_test": rpn.get("PRE_NMS_TOP_N_TEST", 1000),
            "rpn_post_nms_top_n_train": rpn.get("POST_NMS_TOP_N_TRAIN", 2000),
            "rpn_post_nms_top_n_test": rpn.get("POST_NMS_TOP_N_TEST", 1000),
            "rpn_nms_thresh": rpn.get("NMS_THRESH", 0.7),
            "rpn_fg_iou_thresh": rpn.get("FG_IOU_THRESH", 0.7),
            "rpn_bg_iou_thresh": rpn.get("BG_IOU_THRESH", 0.3),
            "rpn_batch_size_per_image": rpn.get("BATCH_SIZE_PER_IMAGE", 256),
            "rpn_positive_fraction": rpn.get("POSITIVE_FRACTION", 0.5),
            "box_score_thresh": roi.get("SCORE_THRESH", 0.05),
            "box_nms_thresh": roi.get("NMS_THRESH", 0.5),
            "box_detections_per_img": roi.get("DETECTIONS_PER_IMG", 100),
            "box_fg_iou_thresh": roi.get("FG_IOU_THRESH", 0.5),
            "box_bg_iou_thresh": roi.get("BG_IOU_THRESH", 0.5),
            "box_batch_size_per_image": roi.get("BATCH_SIZE_PER_IMAGE", 512),
            "box_positive_fraction": roi.get("POSITIVE_FRACTION", 0.25),
        }
    )

    model = model_fn(**kwargs)
    model = replace_box_predictor(model, cfg["DATASET"]["NUM_CLASSES"] + 1)
    model = apply_freeze_strategy(model, cfg)

    custom_weights = cfg["MODEL"].get("CUSTOM_WEIGHTS", "")
    if custom_weights:
        load_model_weights(model, custom_weights, strict=False)

    return model
