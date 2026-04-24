from __future__ import annotations

import os
import warnings
import inspect
from typing import Dict, Tuple

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
)

from .components import make_anchor_generator
from .faster_rcnn import MODEL_FACTORY


MODEL_WEIGHTS_REGISTRY = {
    "fasterrcnn_resnet50_fpn": {
        "weights": FasterRCNN_ResNet50_FPN_Weights,
        "weights_backbone": ResNet50_Weights,
    },
    "fasterrcnn_resnet50_fpn_v2": {
        "weights": FasterRCNN_ResNet50_FPN_V2_Weights,
        "weights_backbone": ResNet50_Weights,
    },
    "fasterrcnn_mobilenet_v3_large_fpn": {
        "weights": FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        "weights_backbone": MobileNet_V3_Large_Weights,
    },
    "fasterrcnn_mobilenet_v3_large_320_fpn": {
        "weights": FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        "weights_backbone": MobileNet_V3_Large_Weights,
    },
}


def _looks_like_local_path(value) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if os.path.isabs(text) or text.startswith(".") or text.startswith("~"):
        return True
    if os.path.sep in text:
        return True
    lower = text.lower()
    return lower.endswith(".pth") or lower.endswith(".pt") or lower.endswith(".bin")


def _resolve_weights(enum_cls, name, model_name: str, field_name: str):
    if name is None:
        return None
    if not isinstance(name, str):
        raise ValueError(f"{model_name}:{field_name} must be a string or None, got {type(name).__name__}.")

    text = name.strip()
    if text == "" or text.upper() == "NONE":
        return None

    if enum_cls is None:
        raise ValueError(f"{model_name}:{field_name} enum class is None; cannot resolve value={text!r}.")
    if not hasattr(enum_cls, "DEFAULT"):
        raise TypeError(f"{model_name}:{field_name} enum class {enum_cls!r} has no DEFAULT attribute.")

    if text.upper() == "DEFAULT":
        return enum_cls.DEFAULT
    if hasattr(enum_cls, text):
        return getattr(enum_cls, text)

    enum_cls_name = getattr(enum_cls, "__name__", str(enum_cls))
    if hasattr(enum_cls, "__members__"):
        available = list(enum_cls.__members__.keys())
    else:
        available = [k for k in dir(enum_cls) if not k.startswith("_")]
    raise ValueError(
        f"Invalid {model_name}:{field_name}={text!r} for enum_cls={enum_cls_name}. "
        f"Supported values include 'DEFAULT' and one of {available}."
    )


def _resolve_official_weights(model_name: str, model_cfg):
    if model_name not in MODEL_WEIGHTS_REGISTRY:
        raise ValueError(
            f"Missing weights registry entry for model={model_name}. "
            f"Registered models: {', '.join(sorted(MODEL_WEIGHTS_REGISTRY.keys()))}"
        )

    weights_cls = MODEL_WEIGHTS_REGISTRY[model_name]["weights"]
    weights_backbone_cls = MODEL_WEIGHTS_REGISTRY[model_name]["weights_backbone"]

    weights_name = model_cfg.get("WEIGHTS")
    backbone_name = model_cfg.get("WEIGHTS_BACKBONE")
    pretrained = bool(model_cfg.get("PRETRAINED", True))

    local_weights_path = None
    if _looks_like_local_path(weights_name):
        local_weights_path = os.path.expanduser(str(weights_name).strip())
        if not os.path.isfile(local_weights_path):
            raise FileNotFoundError(f"Local MODEL.WEIGHTS file not found: {local_weights_path}")
        return None, None, local_weights_path

    if not pretrained:
        weights = _resolve_weights(weights_cls, weights_name, model_name, "WEIGHTS")
        weights_backbone = _resolve_weights(weights_backbone_cls, backbone_name, model_name, "WEIGHTS_BACKBONE")
        return weights, weights_backbone, local_weights_path

    if weights_name is None or (isinstance(weights_name, str) and weights_name.strip() == ""):
        weights_name = "DEFAULT"
    if backbone_name is None or (isinstance(backbone_name, str) and backbone_name.strip() == ""):
        backbone_name = "DEFAULT"

    weights = _resolve_weights(weights_cls, weights_name, model_name, "WEIGHTS")
    weights_backbone = _resolve_weights(weights_backbone_cls, backbone_name, model_name, "WEIGHTS_BACKBONE")
    return weights, weights_backbone, local_weights_path


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
    weights, weights_backbone, local_weights_path = _resolve_official_weights(model_name, cfg["MODEL"])
    kwargs = _build_constructor_kwargs(cfg)
    # torchvision mobilenet Faster R-CNN wrappers already construct and pass a built-in
    # anchor generator internally; forwarding rpn_anchor_generator again causes:
    # "FasterRCNN() got multiple values for keyword argument 'rpn_anchor_generator'".
    if model_name in {
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
    }:
        kwargs.pop("rpn_anchor_generator", None)

    sig = inspect.signature(model_fn)
    if "weights" in sig.parameters:
        kwargs["weights"] = weights
    else:
        warnings.warn(f"{model_name} does not support constructor arg `weights`; it will be ignored.", stacklevel=2)

    if "weights_backbone" in sig.parameters:
        kwargs["weights_backbone"] = weights_backbone
    else:
        if weights_backbone is not None:
            warnings.warn(
                f"{model_name} does not support constructor arg `weights_backbone`; resolved value will be ignored.",
                stacklevel=2,
            )

    model = model_fn(**kwargs)

    if local_weights_path is not None:
        load_info = load_model_weights(model, local_weights_path, strict=False)
        missing_keys = load_info.get("missing_keys", [])
        unexpected_keys = load_info.get("unexpected_keys", [])
        if missing_keys or unexpected_keys:
            warnings.warn(
                f"Loaded local MODEL.WEIGHTS={local_weights_path} with strict=False. "
                f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}",
                stacklevel=2,
            )

    fg_num_classes = _resolve_model_classes(cfg)
    if cfg["MODEL"].get("REPLACE_HEAD", True):
        model = replace_box_predictor(model, fg_num_classes + 1)

    model = apply_freeze_strategy(model, cfg)
    return model
