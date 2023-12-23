import torch
from utils.warmup_lr import GradualWarmupScheduler
from utils.loss import *
from models.ResNetBERT import ResNet1D
from transformers import BertConfig

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

bert_config = BertConfig(
    num_hidden_layers=6,
    num_attention_heads=8,
)
model = ResNet1D(n_block = 6, base_filters = 64, kernel_size = 13, stride = 3, in_channels = 12, groups=1, increasefilter_gap=2, n_classes=27, bert_config=bert_config).to(device)
warmup_epochs = 10
total_epochs = 100
min_length = 7500
window_size = 7500
window_stride = 3000
lr = 1e-4
weight_decay = 1e-5
batch_size = 4
loss = torch.nn.BCEWithLogitsLoss()
val_loss = torch.nn.BCELoss()

#### Weighted Loss ####
pos_weight = negative_over_positive_weights(device)
loss.pos_weight = pos_weight
val_loss.pos_weight = pos_weight

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs-warmup_epochs)
lr_scheduler = GradualWarmupScheduler(optimizer, warmup_epochs, cosine_scheduler, value_flag=False)