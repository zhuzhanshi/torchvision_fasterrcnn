from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)

MODEL_FACTORY = {
    "fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn,
    "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
    "fasterrcnn_mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn,
    "fasterrcnn_mobilenet_v3_large_320_fpn": fasterrcnn_mobilenet_v3_large_320_fpn,
}
