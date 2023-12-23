import torch
from utils.warmup_lr import GradualWarmupScheduler
from utils.loss import *
from models.ECGFormer import *
import torch.nn as nn
from calculate_downsample import cnn_output_shape

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


model = ECGFormer(12, 512, [7, 5, 5, 3], [3, 2, 2, 2], activation=nn.GELU(), n_layers=4, attention_heads=8, num_additional_features=6, max_seq_length=10, n_classes=27, device=device).to(device)
print(cnn_output_shape(model.encoder, 250))
warmup_epochs = 10
total_epochs = 100
min_length = 7500
window_size = 7500
window_stride = 3000
lr = 1e-5
weight_decay = 1e-7
batch_size = 8
loss = torch.nn.BCEWithLogitsLoss()
val_loss = torch.nn.BCELoss()

#### Weighted Loss ####
pos_weight = negative_over_positive_weights(device)
loss.pos_weight = pos_weight
val_loss.pos_weight = pos_weight

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs-warmup_epochs)
lr_scheduler = GradualWarmupScheduler(optimizer, warmup_epochs, cosine_scheduler, value_flag=False)

use_rpeaks = True