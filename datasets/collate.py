def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
