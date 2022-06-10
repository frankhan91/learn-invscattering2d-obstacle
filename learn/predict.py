'''
predict the coefficients with the trained NN
given the scattered data
'''

import os
import json
import argparse
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

directory = os.getcwd()
print("The current directory is ",directory)

# load the scattered data at target and also json file
# TODO: make it to config later for both data and predictor
data_dir = "./data/star3_kh10_2/" #the / at last is important
data_name = "forward_data"
fname = os.path.join(data_dir, data_name+'.mat')
data = scipy.io.loadmat(fname)
std = np.load("std.npy")

# load the predictor
predictor_dir = "./data/star3_kh10_100/"
predictor_name = "test.pt"
predictor = os.path.join(predictor_dir, predictor_name)
saved_parameter = torch.load(predictor)
nc = 3
n_coefs = 2*nc+1 # TODO: change it
class ConvNet(nn.Module):
    def __init__(self):
        # TODO: construct network from config rather than hardcoded to test different architecture easily
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2, padding_mode='circular')
        self.pool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(4, 4, 5, padding=2, padding_mode='circular')
        self.fc1 = nn.Linear(4 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_coefs)
        # self.std = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

loaded_net = ConvNet()
loaded_net.load_state_dict(torch.load(predictor))
loaded_net = loaded_net

# apply the predictor to obtian an initialization
uscat_all = data["uscat_all"].real / std # the scattered data
n_sample, n_dir, n_theta = np.shape(uscat_all)
uscat_all = torch.from_numpy(uscat_all.reshape(n_sample,1,n_dir,n_theta))
uscat_all = uscat_all.float()
coef_pred = loaded_net(uscat_all)

# save the Fourier coefficients 
scipy.io.savemat(
    os.path.join(data_dir, "{}_pred.mat".format(data_name)),
    {"coef_pred": coef_pred.detach().numpy().astype('float64')}
)