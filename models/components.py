from __future__ import annotations

from typing import Iterable, Sequence

from torchvision.models.detection.anchor_utils import AnchorGenerator


def to_tuple_of_tuples(values: Sequence[Sequence[float]]):
    return tuple(tuple(v) for v in values)


def make_anchor_generator(anchor_sizes, aspect_ratios):
    return AnchorGenerator(sizes=to_tuple_of_tuples(anchor_sizes), aspect_ratios=to_tuple_of_tuples(aspect_ratios))
