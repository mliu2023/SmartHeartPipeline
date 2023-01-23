import torch
from warmup_lr import GradualWarmupScheduler
from models.SEResNet import ResNet
from models.ResNet1dTransformer import ResNet1D
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet1D(n_block = 12, base_filters = 32, kernel_size = 14, stride = 3, in_channels = 12, groups=1, n_classes=27).cuda()
warmup_epochs = 10
total_epochs = 100
min_length = 7500
window_size = 7500
window_stride = 3000
lr = 1e-3
weight_decay = 1e-5
loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs-warmup_epochs)
lr_scheduler = GradualWarmupScheduler(optimizer, warmup_epochs, cosine_scheduler, value_flag=False)