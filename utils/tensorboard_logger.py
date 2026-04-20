from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalars(self, tag, scalar_dict, step):
        for k, v in scalar_dict.items():
            self.writer.add_scalar(f"{tag}/{k}", v, step)

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()
