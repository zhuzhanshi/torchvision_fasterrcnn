from torch.utils.data import RandomSampler, SequentialSampler


def build_sampler(dataset, shuffle=True):
    return RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
