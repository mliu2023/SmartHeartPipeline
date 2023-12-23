import numpy as np
class EarlyStopping:
    def __init__(self, patience=15, delta=0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.min_loss = np.inf
        self.max_score = 0

    def __call__(self, metric):
        if(self.mode == 'min'):
            if metric < self.min_loss:
                self.min_loss = metric
                self.counter = 0
            elif metric > (self.min_loss + self.delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
        elif(self.mode == 'max'):
            if metric > self.max_score:
                self.max_score = metric
                self.counter = 0
            elif metric < (self.max_score - self.delta):
                self.counter += 1
                if(self.counter >= self.patience):
                    return True
            return False