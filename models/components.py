from torchvision.models.detection.anchor_utils import AnchorGenerator


def make_anchor_generator(anchor_sizes, aspect_ratios):
    return AnchorGenerator(sizes=tuple(tuple(s) for s in anchor_sizes), aspect_ratios=tuple(tuple(a) for a in aspect_ratios))
