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
    parser.add_argument("--data_dir", default="./data/star3_kh10_100", type=str)
    parser.add_argument("--data_name", default="data_to_predict.mat", type=str)
    # model dir is in data dir
    parser.add_argument("--model_name", default="test", type=str)
    # add config because nc is needed to define ConvNet
    parser.add_argument("--cfgpath", default="./configs/nc3.json", type=str)
    args = parser.parse_args()
    f = open(args.cfgpath)
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
    model_dir = os.path.join(args.data_dir, args.model_name)
    
    # load data
    fname = os.path.join(args.data_dir, args.data_name)
    data = scipy.io.loadmat(fname)
    std = np.load(os.path.join(args.data_dir, "std.npy"))
    
    #load predictor
    predictor_name = args.model_name + ".pt"
    predictor = os.path.join(model_dir, predictor_name)
    loaded_net = ConvNet()
    loaded_net.load_state_dict(torch.load(predictor))
    loaded_net = loaded_net
    
    # apply the predictor to obtian an initialization
    uscat_all = data["uscat_all"].real / std # the scattered data
    n_sample, n_dir, n_theta = np.shape(uscat_all)
    uscat_all = torch.from_numpy(uscat_all.reshape(n_sample,1,n_dir,n_theta))
    uscat_all = uscat_all.float()
    coef_pred = loaded_net(uscat_all)
    
    # allow data_name to have ".mat" or not, both fine
    if args.data_name[-4:] == ".mat":
        d_name = args.data_name[:-4]
    else:
        d_name = args.data_name
    # save the Fourier coefficients
    scipy.io.savemat(
        os.path.join(model_dir, "{}_predby_{}.mat".format(d_name, args.model_name)),
        {"coef_pred": coef_pred.detach().numpy().astype('float64')}
    )

if __name__ == '__main__':
    main()