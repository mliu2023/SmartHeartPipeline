import torch
from utils.warmup_lr import GradualWarmupScheduler
from models.SEResNet import SEResNet
from models.SEResNetv2 import SEResNetv2
from models.SEResNetTransformer import SEResNetTransformer
from models.QRNN import QRNN
from utils.loss import *

if torch.cuda.is_available():
    device = torch.device("cuda:0") # nvidia gpu
elif torch.backends.mps.is_available():
    device = torch.device("mps") # support for M1 macs
else:
    device = torch.device("cpu") # cpu

# model = SEResNetv2(layers=[2,2,2,2], kernel_sizes=[9], num_additional_features=18).to(device)
# model = SEResNetTransformer().to(device)
model = QRNN(num_class_embeddings=27).to(device)


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
pos_weight = negative_over_positive_weights(device)
print(f'Weights for each class: {pos_weight}')

# loss1 = torch.nn.BCEWithLogitsLoss()
# loss1.pos_weight = pos_weight
# loss2 = Confusion_Weighted_BCELoss()
# loss2.pos_weight = pos_weight
# loss = Combined_Losses(losses=[loss1,loss2],weights=[0.8,0.2])
loss = torch.nn.BCEWithLogitsLoss()
loss.pos_weight = pos_weight
val_loss = torch.nn.BCELoss()

#### optimizer and lr scheduler ####
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs-warmup_epochs)
lr_scheduler = GradualWarmupScheduler(optimizer, warmup_epochs, after_scheduler, value_flag=False)