from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .components import make_anchor_generator
from .faster_rcnn import MODEL_FACTORY


def _resolve_weights(enum_cls, name):
    if enum_cls is None:
        return None
    if not name:
        return None
    if isinstance(name, str) and name.upper() == "DEFAULT":
        return enum_cls.DEFAULT
    if isinstance(name, str) and hasattr(enum_cls, name):
        return getattr(enum_cls, name)
    return None


def _resolve_official_weights(model_fn, model_cfg):
    annotations = getattr(model_fn, "__annotations__", {})
    weights_cls = annotations.get("weights")
    weights_backbone_cls = annotations.get("weights_backbone")

    weights_name = model_cfg.get("WEIGHTS")
    backbone_name = model_cfg.get("WEIGHTS_BACKBONE")
    pretrained = bool(model_cfg.get("PRETRAINED", True))

    if not pretrained and (weights_name is None or str(weights_name).upper() in {"", "DEFAULT", "NONE"}):
        return None, None
    if not pretrained and backbone_name is None:
        backbone_name = None

    if pretrained and not weights_name:
        weights_name = "DEFAULT"
    if pretrained and backbone_name is None:
        backbone_name = "DEFAULT"

    return _resolve_weights(weights_cls, weights_name), _resolve_weights(weights_backbone_cls, backbone_name)


def _resolve_model_classes(cfg: Dict) -> int:
    ds_num = int(cfg["DATASET"]["NUM_CLASSES"])
    model_num = int(cfg["MODEL"].get("NUM_CLASSES", ds_num))
    if ds_num != model_num:
        raise ValueError(
            f"NUM_CLASSES mismatch: DATASET.NUM_CLASSES={ds_num} vs MODEL.NUM_CLASSES={model_num}. "
            "Both must be foreground-class count without background."
        )
    return model_num


def build_anchor_generator(cfg: Dict):
    rpn_cfg = cfg["MODEL"].get("RPN", {})
    if not rpn_cfg.get("USE_CUSTOM", True):
        return None
    return make_anchor_generator(rpn_cfg["ANCHOR_SIZES"], rpn_cfg["ASPECT_RATIOS"])


def replace_box_predictor(model: nn.Module, num_classes_with_background: int):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_background)
    return model


def apply_freeze_strategy(model: nn.Module, cfg: Dict):
    model_cfg = cfg["MODEL"]
    if model_cfg.get("FREEZE_BACKBONE", False):
        for p in model.backbone.parameters():
            p.requires_grad = False
        return model

    freeze_at = int(model_cfg.get("FREEZE_BACKBONE_AT", 0))
    if freeze_at <= 0:
        return model

    # Basic available strategy for ResNet-style backbones
    body = getattr(model.backbone, "body", None)
    if body is None:
        return model
    freeze_prefixes = []
    if freeze_at >= 1:
        freeze_prefixes.extend(["conv1", "bn1"])
    if freeze_at >= 2:
        freeze_prefixes.append("layer1")
    if freeze_at >= 3:
        freeze_prefixes.append("layer2")
    if freeze_at >= 4:
        freeze_prefixes.append("layer3")
    if freeze_at >= 5:
        freeze_prefixes.append("layer4")

    for name, p in body.named_parameters():
        if any(name.startswith(prefix) for prefix in freeze_prefixes):
            p.requires_grad = False
    return model


def load_model_weights(model: nn.Module, path: str, strict: bool = False):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def _build_constructor_kwargs(cfg: Dict):
    model_cfg = cfg["MODEL"]
    rpn_cfg = model_cfg.get("RPN", {})
    roi_cfg = model_cfg.get("ROI_HEADS", {})

    kwargs = {
        "trainable_backbone_layers": model_cfg.get("TRAINABLE_BACKBONE_LAYERS", None),
        "rpn_anchor_generator": build_anchor_generator(cfg),
        "min_size": model_cfg.get("MIN_SIZE"),
        "max_size": model_cfg.get("MAX_SIZE"),
        "image_mean": model_cfg.get("IMAGE_MEAN"),
        "image_std": model_cfg.get("IMAGE_STD"),
        "rpn_pre_nms_top_n_train": rpn_cfg.get("PRE_NMS_TOP_N_TRAIN", 2000),
        "rpn_pre_nms_top_n_test": rpn_cfg.get("PRE_NMS_TOP_N_TEST", 1000),
        "rpn_post_nms_top_n_train": rpn_cfg.get("POST_NMS_TOP_N_TRAIN", 2000),
        "rpn_post_nms_top_n_test": rpn_cfg.get("POST_NMS_TOP_N_TEST", 1000),
        "rpn_nms_thresh": rpn_cfg.get("NMS_THRESH", 0.7),
        "rpn_fg_iou_thresh": rpn_cfg.get("FG_IOU_THRESH", 0.7),
        "rpn_bg_iou_thresh": rpn_cfg.get("BG_IOU_THRESH", 0.3),
        "rpn_batch_size_per_image": rpn_cfg.get("BATCH_SIZE_PER_IMAGE", 256),
        "rpn_positive_fraction": rpn_cfg.get("POSITIVE_FRACTION", 0.5),
        "rpn_score_thresh": rpn_cfg.get("SCORE_THRESH", 0.0),
        "box_score_thresh": roi_cfg.get("BOX_SCORE_THRESH", roi_cfg.get("SCORE_THRESH", 0.05)),
        "box_nms_thresh": roi_cfg.get("BOX_NMS_THRESH", roi_cfg.get("NMS_THRESH", 0.5)),
        "box_detections_per_img": roi_cfg.get("BOX_DETECTIONS_PER_IMG", roi_cfg.get("DETECTIONS_PER_IMG", 100)),
        "box_fg_iou_thresh": roi_cfg.get("BOX_FG_IOU_THRESH", roi_cfg.get("FG_IOU_THRESH", 0.5)),
        "box_bg_iou_thresh": roi_cfg.get("BOX_BG_IOU_THRESH", roi_cfg.get("BG_IOU_THRESH", 0.5)),
        "box_batch_size_per_image": roi_cfg.get("BATCH_SIZE_PER_IMAGE", 512),
        "box_positive_fraction": roi_cfg.get("POSITIVE_FRACTION", 0.25),
    }
    return kwargs


def build_model(cfg: Dict):
    model_name = cfg["MODEL"]["NAME"]
    if model_name not in MODEL_FACTORY:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported: {', '.join(sorted(MODEL_FACTORY.keys()))}"
        )

    model_fn = MODEL_FACTORY[model_name]
    weights, weights_backbone = _resolve_official_weights(model_fn, cfg["MODEL"])
    kwargs = _build_constructor_kwargs(cfg)
    kwargs.update({"weights": weights, "weights_backbone": weights_backbone})

    model = model_fn(**kwargs)

    fg_num_classes = _resolve_model_classes(cfg)
    if cfg["MODEL"].get("REPLACE_HEAD", True):
        model = replace_box_predictor(model, fg_num_classes + 1)

    model = apply_freeze_strategy(model, cfg)
    return model
