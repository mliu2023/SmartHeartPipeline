import torch
from warmup_lr2 import GradualWarmupScheduler
from models.SEResNet import ResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet().to(device)
warmup_epochs = 10
total_epochs = 100
max_length = 1500
window_size = 1500
window_stride = 750
lr = 1e-5
weight_decay = 1e-5
loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs-warmup_epochs)
lr_scheduler =  GradualWarmupScheduler(optimizer, warmup_epochs, cosine_scheduler, value_flag=False)