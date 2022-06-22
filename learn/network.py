"""
script for neural network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, data_cfg, train_cfg):
        super().__init__()
        nc = data_cfg["nc"]
        out_channels = train_cfg["out_channels"]
        kernel_size = train_cfg["kernel_size"]
        paddle = train_cfg["paddle"]
        linear_dim = train_cfg["linear_dim"]
        n_tgt = data_cfg["n_tgt"]
        n_dir = data_cfg["n_dir"]
        # TODO: be more flexible to more layers
        outdim1 = n_tgt + 2*paddle - kernel_size + 1
        outdim2 = n_dir + 2*paddle - kernel_size + 1
        outdim1 = int(outdim1/2)
        outdim2 = int(outdim2/2)
        outdim1 = outdim1 + 2*paddle - kernel_size + 1
        outdim2 = outdim2 + 2*paddle - kernel_size + 1
        outdim1 = int(outdim1/2)
        outdim2 = int(outdim2/2)
        self.conv1 = nn.Conv2d(1, out_channels, kernel_size, padding=paddle, padding_mode='circular')
        self.pool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=paddle, padding_mode='circular')
        self.fc1 = nn.Linear(out_channels * outdim1 * outdim2, linear_dim[0])
        self.fc2 = nn.Linear(linear_dim[0], linear_dim[1])
        self.fc3 = nn.Linear(linear_dim[1], 2 * nc + 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x