# this is outdated, can be deleted in the future
from torch.optim.lr_scheduler import _LRScheduler

class InverseSquareRootScheduler(_LRScheduler):
    def __init__(self, optimizer, num_epochs, batch_size = 8):
        self.num_epochs = num_epochs
        self.init_lr = optimizer.param_groups[0]['lr']
        self.batch_size = batch_size
        super(InverseSquareRootScheduler, self).__init__(optimizer)

    def get_lr(self):
        #decay_factor = self.init_lr * self.num_epochs**0.5
        #self.lr = decay_factor * self._step_count**-0.5
        self.lr = self.init_lr * (self.batch_size*self._step_count)**-0.5
        return [self.lr for group in self.optimizer.param_groups]