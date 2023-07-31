'''
predict the coefficients with the trained NN
given the scattered data
'''

import os
import json
import argparse
import numpy as np
import scipy.io
import scipy.fftpack as sfft
import torch
import torch.utils.data
import network

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# TODO: add config for both data and predictor
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/star3_kh10_n48_100/temp.mat", type=str)
    parser.add_argument("--model_path", default="./data/star3_kh10_n48_100/test", type=str)
    parser.add_argument("--print_coef", default=False, type=bool)
    args = parser.parse_args()

    f = open(os.path.join(args.model_path, "data_config.json"))
    data_cfg = json.load(f)
    f.close()
    
    g = open(os.path.join(args.model_path, "train_config.json"))
    train_cfg = json.load(g)
    g.close()
    return args, data_cfg, train_cfg

def main():
    args, data_cfg, train_cfg = parse_args()
    network_type = train_cfg["network_type"]
    if train_cfg["data_type"] == "float32": data_type = torch.float32
    elif train_cfg["data_type"] == "float64": data_type = torch.float64
    # load data
    data = scipy.io.loadmat(args.data_path)
    
    #load stats and predictor
    f = open(os.path.join(args.model_path, "mean_std.txt"))
    mean_std = f.read()
    f.close()
    if network_type == 'convnet':
        mean, std = [float(x) for x in mean_std.split('\n')]
        loaded_net = network.ConvNet(data_cfg, train_cfg)
    elif network_type == 'complexnet':
        mean_r, std_r, mean_i, std_i = [float(x) for x in mean_std.split('\n')]
        loaded_net = network.ComplexNet(data_cfg, train_cfg)
    loaded_net.type(data_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_net.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt"), map_location=device))
    
    # apply the predictor to obtian an initialization
    uscat_all = data["uscat_all"]
    if train_cfg["n_dir_train"] > 0:
        uscat_all = uscat_all[:,0:train_cfg["n_dir_train"],:]
    if train_cfg["n_tgt_train"] > 0:
        uscat_all = uscat_all[:,:,0:train_cfg["n_tgt_train"]]
    if train_cfg["network_type"] == 'convnet':
        data_to_predict = (uscat_all.real[:,None,:,:]-mean) / std # the scattered data
    elif train_cfg["network_type"] == 'complexnet':
        uscat_ft = sfft.fft2(uscat_all)
        uscat_ft_shift = sfft.fftshift(uscat_ft,axes=(1,2))
        data_to_predict = uscat_ft_shift
        data_real = (data_to_predict.real-mean_r) / std_r
        data_imag = (data_to_predict.imag-mean_i) / std_i
        data_to_predict = np.concatenate((data_real[:, None, :, :], data_imag[:, None, :, :]), axis=1)
    data_to_predict = torch.from_numpy(data_to_predict)
    data_to_predict = data_to_predict.type(data_type)
    coef_pred = loaded_net(data_to_predict)
    
    # save predicted coef
    if args.data_path[-4:] == ".mat":
        new_data_path = args.data_path[:-4]
    else:
        new_data_path = args.data_path

    model_name = os.path.basename(os.path.normpath(args.model_path))

    scipy.io.savemat(
        new_data_path + "_predby_" + model_name + ".mat",
        {"coef_pred": coef_pred.detach().numpy().astype('float64'),
         "coef_val": data["coefs_all"],
         "cfg_str": data["cfg_str"][0]}
    )
    
    # only for inverse code with data_type=nn
    if args.print_coef:
        coef_pred_np = coef_pred.detach().numpy() #shape 1 x (2*nc+1)
        print("start to print the coefficients")
        [print(num) for num in coef_pred_np[0]]

if __name__ == '__main__':
    main()