# warmup learning rate scheduler
class Warmup_LR(object):
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.step(1)

    def warmup_learning_rate(self, current_iteration):
        warmup_lr = self.target_lr*float(current_iteration)/float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, current_iteration):
        if current_iteration <= self.warmup_iteration:
            self.warmup_learning_rate(current_iteration)
        else:
            self.after_scheduler.step(current_iteration-self.warmup_iteration)
    
    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict)