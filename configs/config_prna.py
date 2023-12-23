import torch
from utils.warmup_lr import GradualWarmupScheduler
from utils.loss import *
from models.Prna_Transformer import CTN

# Transformer parameters
d_model = 256   # embedding size
nhead = 8       # number of heads
d_ff = 2048     # feed forward layer size
num_layers = 8  # number of encoding layers

dropout_rate = 0.2
deepfeat_sz = 64
nb_feats = 0
nb_demo = 3

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
model = CTN(d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz, nb_feats, nb_demo).to(device)
warmup_epochs = 10
total_epochs = 100
min_length = 7500
window_size = 7500
window_stride = 3000
lr = 1e-3
weight_decay = 1e-5
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