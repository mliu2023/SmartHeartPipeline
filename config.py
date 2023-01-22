import torch
from warmup_lr import GradualWarmupScheduler
from models.SEResNet import ResNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet().to(device)
warmup_epochs = 10
total_epochs = 100
min_length = 5000
window_size = 5000
window_stride = 3000
lr = 1e-3
weight_decay = 1e-5
loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs-warmup_epochs)
lr_scheduler = GradualWarmupScheduler(optimizer, warmup_epochs, cosine_scheduler, value_flag=False)