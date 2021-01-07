from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, warmup_epochs, post_warmup_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.warmup_epochs = warmup_epochs
        self.post_warmup_scheduler = post_warmup_scheduler
        self.finished = False if warmup_epochs > 0 else True
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.warmup_epochs:
            if self.post_warmup_scheduler:
                if not self.finished:
                    self.post_warmup_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.post_warmup_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epochs + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.warmup_epochs:
            if self.last_epoch == self.total_epoch:
                self.post_warmup_scheduler.best = metrics
            if self.multiplier == 1.0:
                warmup_lr = [base_lr * (float(self.last_epoch) / self.warmup_epochs) for base_lr in self.base_lrs]
            else:
                warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epochs + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr

        else:
            if epoch is None:
                self.post_warmup_scheduler.step(metrics, None)
            else:
                self.post_warmup_scheduler.step(metrics, epoch - self.warmup_epochs)

    def step(self, epoch=None, metrics=None):
        if type(self.post_warmup_scheduler) != ReduceLROnPlateau:
            if self.finished and self.post_warmup_scheduler:
                self.post_warmup_scheduler.step()
                self._last_lr = self.post_warmup_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step()
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
