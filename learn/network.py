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
        n_dir = data_cfg["n_dir"]
        n_tgt = data_cfg["n_tgt"]
        padding_mode = 'circular'
        # will use 'zeros' padding if any of n_dir_train or n_tgt_train > 0
        if train_cfg["n_dir_train"] > 0:
            n_dir = train_cfg["n_dir_train"]
            padding_mode = 'zeros'
        if train_cfg["n_tgt_train"] > 0:
            n_tgt = train_cfg["n_tgt_train"]
            padding_mode = 'zeros'
        # TODO: be more flexible to more layers
        outdim1 = n_dir + 2*paddle - kernel_size + 1
        outdim2 = n_tgt + 2*paddle - kernel_size + 1
        outdim1 = int(outdim1/2)
        outdim2 = int(outdim2/2)
        outdim1 = outdim1 + 2*paddle - kernel_size + 1
        outdim2 = outdim2 + 2*paddle - kernel_size + 1
        outdim1 = int(outdim1/2)
        outdim2 = int(outdim2/2)
        self.conv1 = nn.Conv2d(1, out_channels, kernel_size, padding=paddle, padding_mode=padding_mode)
        self.pool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=paddle, padding_mode=padding_mode)
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
    
class ComplexNet(nn.Module):
    def __init__(self, data_cfg, train_cfg):
        super().__init__()
        nc = data_cfg["nc"]
        out_channels = train_cfg["out_channels"]
        kernel_size = train_cfg["kernel_size"]
        paddle = train_cfg["paddle"]
        linear_dim = train_cfg["linear_dim"]
        n_dir = data_cfg["n_dir"]
        n_tgt = data_cfg["n_tgt"]
        if train_cfg["n_dir_train"] > 0 or train_cfg["n_tgt_train"] > 0:
            raise Exception("Cannot train Fourier net with partial data")
        # TODO: be more flexible to more layers
        outdim1 = n_tgt + 2*paddle - kernel_size + 1
        outdim2 = n_dir + 2*paddle - kernel_size + 1
        outdim1 = int(outdim1/2)
        outdim2 = int(outdim2/2)
        outdim1 = outdim1 + 2*paddle - kernel_size + 1
        outdim2 = outdim2 + 2*paddle - kernel_size + 1
        outdim1 = int(outdim1/2)
        outdim2 = int(outdim2/2)
        self.conv1r = nn.Conv2d(1, out_channels, kernel_size, padding=paddle, padding_mode='circular')
        self.pool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.conv2r = nn.Conv2d(out_channels, out_channels, kernel_size, padding=paddle, padding_mode='circular')
        self.fc1r = nn.Linear(out_channels * outdim1 * outdim2, linear_dim[0])
        self.fc2r = nn.Linear(linear_dim[0], linear_dim[1])
        self.fc3r = nn.Linear(linear_dim[1], 2 * nc + 1)
        self.conv1c = nn.Conv2d(1, out_channels, kernel_size, padding=paddle, padding_mode='circular')
        self.conv2c = nn.Conv2d(out_channels, out_channels, kernel_size, padding=paddle, padding_mode='circular')
        self.fc1c = nn.Linear(out_channels * outdim1 * outdim2, linear_dim[0])
        self.fc2c = nn.Linear(linear_dim[0], linear_dim[1])
        self.fc3c = nn.Linear(linear_dim[1], 2 * nc + 1)

    def forward(self, x_and_y):
        x = x_and_y[:,0:1,:,:] #real part
        y = x_and_y[:,1:2,:,:] #imaginary part
        x = F.relu(self.conv1r(x))
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2r(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1r(x))
        x = F.relu(self.fc2r(x))
        x = self.fc3r(x)
        y = F.relu(self.conv1c(y))
        y = self.pool(y)
        y = self.pool(F.relu(self.conv2c(y)))
        y = torch.flatten(y, 1) # flatten all dimensions except batch
        y = F.relu(self.fc1c(y))
        y = F.relu(self.fc2c(y))
        y = self.fc3c(y)
        return x + y