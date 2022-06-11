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

# TODO: add config for both data and predictor
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/star3_kh10_100/data_to_predict.mat", type=str)
    parser.add_argument("--model_path", default="./data/star3_kh10_100/test", type=str)
    args = parser.parse_args()
    f = open(os.path.join(args.model_path, "config.json"))
    cfg = json.load(f)
    nc = cfg['nc']
    global n_coefs
    n_coefs = 2 * nc + 1
    f.close()
    return args

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

def main():
    args = parse_args()
    
    # load data
    data = scipy.io.loadmat(args.data_path)
    f = open(os.path.join(args.model_path, "std.txt"))
    std = f.read()
    std = float(std)
    
    #load predictor
    loaded_net = ConvNet()
    loaded_net.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt")))
    
    # apply the predictor to obtian an initialization
    uscat_all = data["uscat_all"].real / std # the scattered data
    n_sample, n_dir, n_theta = np.shape(uscat_all)
    uscat_all = torch.from_numpy(uscat_all.reshape(n_sample,1,n_dir,n_theta))
    uscat_all = uscat_all.float()
    coef_pred = loaded_net(uscat_all)
    
    idx = args.data_path.rfind("/")
    data_name = args.data_path[idx+1:]
    if data_name[-4:] == ".mat":
        data_name = data_name[:-4]
    idx = args.model_path.rfind("/")
    model_name = args.model_path[idx+1:]
    scipy.io.savemat(
        os.path.join(args.model_path, "{}_predby_{}.mat".format(data_name, model_name)),
        {"coef_pred": coef_pred.detach().numpy().astype('float64')}
    )

if __name__ == '__main__':
    main()