import torch
import torch.nn as nn

class dummyNN(nn.Module):
    def __init__(self, channels=12, length=1500):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 1, 10, 2)
        self.fc1 = nn.Linear(1246, 27)
        #self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x).squeeze(dim=1)
        return x