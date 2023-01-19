import torch
from warmup_lr import Warmup_LR
from models.SEResNet import ResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet().to(device)
warmup_epochs = 10
total_epochs = 100
max_length = 1500
window_size = 1500
window_stride = 750
lr = 1e-3
weight_decay = 1e-5
loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs-warmup_epochs)
lr_scheduler = Warmup_LR(optimizer=optimizer,
                        warmup_iteration=warmup_epochs,
                        target_lr=lr,
                        after_scheduler=cosine_scheduler)