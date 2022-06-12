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
import torch.utils.data
import network

# TODO: add config for both data and predictor
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/star3_kh10_100/forward_data.mat", type=str)
    parser.add_argument("--model_path", default="./data/star3_kh10_100/test", type=str)
    args = parser.parse_args()

    f = open(os.path.join(args.model_path, "data_config.json"))

    cfg = json.load(f)
    nc = cfg['nc']
    global n_coefs
    n_coefs = 2 * nc + 1
    f.close()
    return args

def main():
    args = parse_args()
    
    # load data
    data = scipy.io.loadmat(args.data_path)
    f = open(os.path.join(args.model_path, "std.txt"))
    std = f.read()
    std = float(std)
    
    #load predictor
    loaded_net = network.ConvNet(n_coefs)
    loaded_net.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt")))
    
    # apply the predictor to obtian an initialization
    uscat_all = data["uscat_all"].real / std # the scattered data
    n_sample, n_dir, n_theta = np.shape(uscat_all)
    uscat_all = torch.from_numpy(uscat_all.reshape(n_sample,1,n_dir,n_theta))
    uscat_all = uscat_all.float()
    coef_pred = loaded_net(uscat_all)
    
    # save predicted coef
    if args.data_path[-4:] == ".mat":
        new_data_path = args.data_path[:-4]
    else:
        new_data_path = args.data_path

    model_name = os.path.basename(os.path.normpath(args.model_path))

    scipy.io.savemat(
        new_data_path + "_predby_" + model_name + ".mat",
        {"coef_pred": coef_pred.detach().numpy().astype('float64')}
    )

if __name__ == '__main__':
    main()