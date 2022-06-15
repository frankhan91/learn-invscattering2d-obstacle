"""
script for neural network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, n_coefs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2, padding_mode='circular')
        self.pool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(4, 4, 5, padding=2, padding_mode='circular')
        self.fc1 = nn.Linear(4 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_coefs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x