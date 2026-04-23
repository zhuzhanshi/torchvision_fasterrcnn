def detection_collate_fn(batch):
    """Torchvision detection collate.

    Returns:
        images: list[Tensor]
        targets: list[dict]
    """
    images, targets = zip(*batch)
    return list(images), list(targets)
