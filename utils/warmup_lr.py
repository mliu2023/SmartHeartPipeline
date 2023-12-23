from torch.optim.lr_scheduler import _LRScheduler
import math

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epoch: target learning rate is reached at warmup_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
        value_flag: whether value is required for after_scheduler step
    """

    def __init__(self, optimizer, warmup_epoch, after_scheduler, value_flag=False):
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.value_flag = value_flag
        self.finished = False if warmup_epoch != 0 else True
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch and self.warmup_epoch != 0:
            if self.last_epoch == self.warmup_epoch - 1:
                self.finished = True
            return [base_lr * ((float(self.last_epoch) + 1) / self.warmup_epoch) for base_lr in self.base_lrs]
        elif self.after_scheduler is not None:
            return [group['lr'] for group in self.after_scheduler.optimizer.param_groups]
        else:
            return self.base_lrs
    
    def step(self, val_loss=math.inf):
        if self.finished:
            if self.value_flag:
                self.after_scheduler.step(val_loss)
            else:
                self.after_scheduler.step()
            self._last_lr = self.after_scheduler._last_lr
            print(f'Next learning rate: {self._last_lr}')
        else:
            super(GradualWarmupScheduler, self).step()
            print(f'Next learning rate: {self._last_lr}')