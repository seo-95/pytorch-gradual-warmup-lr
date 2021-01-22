from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, warmup_steps, post_warmup_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.wup_steps = warmup_steps
        self.post_wup_scheduler = post_warmup_scheduler
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        assert self._step_count <= self.wup_steps, 'Warmup scheduler called after last warmup step'
        if self.multiplier == 1.0:
            return [base_lr * (float(self._step_count) / self.wup_steps) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self._step_count / self.wup_steps + 1.) for base_lr in self.base_lrs]

    def step(self):
        if self._step_count < self.wup_steps:
            return super(GradualWarmupScheduler, self).step()
        else:
            if self._step_count == self.wup_steps:
                # setup the post warmup scheduler if this is the first post warmup step
                self.post_wup_scheduler.base_lrs    = [base_lr * self.multiplier for base_lr in self.base_lrs]
                self.post_wup_scheduler._step_count = self._step_count
                self.post_wup_scheduler.last_epoch  = self.last_epoch
            self.post_wup_scheduler.step()
            self._step_count    += 1
            self.last_epoch     += 1
