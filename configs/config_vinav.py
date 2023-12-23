import torch
from utils.warmup_lr import GradualWarmupScheduler
from models.SEResNet import SEResNet
from utils.loss import *

if torch.cuda.is_available():
    device = torch.device("cuda:0") # nvidia gpu
elif torch.backends.mps.is_available():
    device = torch.device("mps") # support for M1 macs
else:
    device = torch.device("cpu") # cpu

# add more features here
model = SEResNet(kernel_size=13, num_additional_features=6).to(device)

#### training parameters ####
warmup_epochs = 10
total_epochs = 100
min_length = 4096
window_size = 750
window_stride = 750
lr = 3e-3
weight_decay = 1e-7
batch_size = 20

# for first experiment, try 7500
# get the windows the significant data
# change the window size

#### loss ####
loss = torch.nn.BCEWithLogitsLoss()
val_loss = torch.nn.BCELoss()

#### Weighted Loss ####
pos_weight = negative_over_positive_weights(device)
print(f'Weights for each class: {pos_weight}')
loss.pos_weight = pos_weight
val_loss.pos_weight = pos_weight

#### optimizer and lr scheduler ####
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# possible after_scheduler options

'''
StepLR: uses geometric progression every x steps to lower learning rate over time
from torch.optim.lr_scheduler import StepLR
step_size is the amount of time it says at a certain learning rate (x)
gamma is the factor by which the learning rate decreases 
set the last_epoch to total_epochs-warmup_epochs
use same optimizer for STEPLR

Note: MultiStepLR can do a similar thing, but instead of x steps, lowers rate based on certain point in time

after_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.5, last_epoch = total_epochs-warmup_epochs)
'''

after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs-warmup_epochs)
lr_scheduler = GradualWarmupScheduler(optimizer, warmup_epochs, after_scheduler, value_flag=False)
projector = False