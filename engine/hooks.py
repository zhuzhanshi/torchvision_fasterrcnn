class HookBase:
    def before_epoch(self, trainer, epoch):
        pass

    def after_epoch(self, trainer, epoch):
        pass
