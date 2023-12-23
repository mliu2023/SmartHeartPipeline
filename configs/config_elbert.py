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

model = SEResNet(kernel_size=13, num_additional_features=6).to(device)

#### training parameters ####
warmup_epochs = 10
total_epochs = 100
min_length = 7500
window_size = 7500
window_stride = 3000
lr = 3e-3
weight_decay = 1e-7
batch_size = 8

#### loss ####
loss = torch.nn.BCEWithLogitsLoss()
val_loss = torch.nn.BCELoss()

#### Weighted Loss ####
pos_weight = negative_over_positive_weights(device)
print(f'Weights for each class: {pos_weight}')
loss.pos_weight = pos_weight
val_loss.pos_weight = pos_weight

#### optimizer and lr scheduler ####
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum = .9)
after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs-warmup_epochs)
lr_scheduler = GradualWarmupScheduler(optimizer, warmup_epochs, after_scheduler, value_flag=False)
projector = False