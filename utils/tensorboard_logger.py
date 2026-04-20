try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


class TBLogger:
    def __init__(self, log_dir: str):
        if SummaryWriter is None:
            raise ImportError("TensorBoard is not available. Please install tensorboard package.")
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalars(self, tag, scalar_dict, step):
        for k, v in scalar_dict.items():
            self.writer.add_scalar(f"{tag}/{k}", v, step)

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()
