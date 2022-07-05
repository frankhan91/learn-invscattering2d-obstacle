import os
import json
import argparse
import numpy as np
import scipy.io
import scipy.fftpack as sfft
import torch
import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import network

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", default="./data/star3_kh10_n48_100", type=str)
    parser.add_argument("--model_name", default="test", type=str)
    parser.add_argument("--train_cfg_path", default=None, type=str)
    args = parser.parse_args()
    if args.train_cfg_path is None:
        dirname = os.path.basename(args.dirname)
        ncstr = dirname.split('_')[0]
        if ncstr.startswith("star"):
            try:
                nc = int(ncstr[4:])
            except ValueError:
                print("Error: cannot get the default training config path from dirname.")
                raise
        args.train_cfg_path = "./configs/train_nc{}.json".format(nc)
    f = open(args.train_cfg_path)
    train_cfg = json.load(f)
    
    f.close()
    return args, train_cfg

class RealData(torch.utils.data.Dataset):
    def __init__(self, root, std, network_type):
        self.root = root
        self.files = os.listdir(root) # take all files in the root directory
        self.std = std
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = scipy.io.loadmat(os.path.join(self.root, self.files[idx]))
        uscat = data["uscat"] # shape n_tgt x n_dir
        uscat = uscat.real / self.std
        coefs = data["coefs"] # shape (2^nc+1) x 1
        return uscat[None,:,:], coefs[:,0]

class ComplexData(torch.utils.data.Dataset):
    def __init__(self, root, std, network_type):
        self.root = root
        self.files = os.listdir(root) # take all files in the root directory
        self.std = std
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = scipy.io.loadmat(os.path.join(self.root, self.files[idx]))
        uscat = data["uscat"]
        uscat_ft = sfft.fft2(uscat)
        uscat_ft_shift = sfft.fftshift(uscat_ft)
        real_part = uscat_ft_shift.real / self.std
        imaginary_part = uscat_ft_shift.imag / self.std
        coefs = data["coefs"]
        return np.concatenate((real_part[None,:,:], imaginary_part[None,:,:])), coefs[:,0]


def main():
    args, train_cfg = parse_args()
    print("Train data from {}".format(args.dirname))
    print("model name", args.model_name)
    fname = os.path.join(args.dirname, "valid_data.mat")
    network_type = train_cfg["network_type"]
    valid_data = scipy.io.loadmat(fname)
    data_cfg = json.loads(valid_data["cfg_str"][0])
    coef_val = valid_data["coefs_val"]
    uscat_val = valid_data["uscat_val"]
    
    if network_type == 'convnet':
        tgt_valid = uscat_val.real
        print("The mean value is", np.mean(tgt_valid))
        std = np.std(tgt_valid)
        tgt_valid = tgt_valid[:, None, :, :] / std
        dataset = RealData(os.path.join(args.dirname, "train_data"), std, network_type)
    elif network_type == 'complexnet':
        uscat_ft = sfft.fft2(uscat_val)
        uscat_ft_shift = sfft.fftshift(uscat_ft,axes=(1,2))
        data_real = uscat_ft_shift.real
        data_imag = uscat_ft_shift.imag
        print("The mean values are", np.mean(data_real), np.mean(data_imag))
        std_r = np.std(data_real)
        std_i = np.std(data_imag)
        std = (std_r**2 + std_i**2)**0.5
        data_real = data_real[:, None, :, :] / std
        data_imag = data_imag[:, None, :, :] / std
        tgt_valid = np.concatenate((data_real, data_imag), axis=1)
        dataset = ComplexData(os.path.join(args.dirname, "train_data"), std, network_type)
    
    tgt_valid = torch.tensor(tgt_valid, dtype=torch.float)
    coef_val = torch.tensor(coef_val, dtype=torch.float)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=train_cfg["batch_size"])
    
    loss_fn = nn.MSELoss()
    log_dir=os.path.join(args.dirname, args.model_name)
    writer = SummaryWriter(log_dir)
    epoch = train_cfg["epoch"]
    def train(model, device, train_loader, optimizer, epoch, scheduler):
        for e in range(epoch):
            n_loss = 0
            current_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data = (data.to(device)).type(torch.float)
                target = (target.to(device)).type(torch.float)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                n_loss += 1
                current_loss += loss.item()
            if e % train_cfg["valid_freq"] == 0:
                coef_pred = model(tgt_valid)
                loss_train = current_loss / n_loss
                loss_val = loss_fn(coef_pred, coef_val.to(device)).item()
                print('Train Epoch: {:3}, Train Loss: {:.6f}, Val loss: {:.6f}'.format(
                    e, loss_train, loss_val)
                )
                writer.add_scalar('loss_train', loss_train, e)
                writer.add_scalar('loss_val', loss_val, e)
                writer.add_scalar('log_log_loss_train', np.log(loss_train), np.log(e+1)*1000)
                writer.add_scalar('log_log_loss_val', np.log(loss_val), np.log(e+1)*1000)
            scheduler.step()
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if network_type == 'convnet':
        model = network.ConvNet(data_cfg, train_cfg).to(device)
    elif network_type == 'complexnet':
        model = network.ComplexNet(data_cfg, train_cfg).to(device)
    
    if train_cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg["lr"], momentum=train_cfg["momentum"])
    elif train_cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    
    scheduler = MultiStepLR(optimizer, milestones=train_cfg["milestones"], gamma=train_cfg["gamma"])
    # TODO: add functionality to re-train
    train(model, device, train_loader, optimizer, epoch, scheduler)
    coef_pred = model(tgt_valid)
    writer.close()
    
    scipy.io.savemat(
        os.path.join(args.dirname, "valid_predby_{}.mat".format(args.model_name)),
        {
            "coef_val": coef_val.numpy().astype('float64'),
            "coef_pred": coef_pred.detach().cpu().numpy().astype('float64'),
            "cfg_str": valid_data["cfg_str"][0]
        }
    )
    model_dir = os.path.join(args.dirname, args.model_name)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    f = open(os.path.join(model_dir, "std.txt"), 'w')
    f.writelines(f"{std}\n")
    f.close()

    g = open(os.path.join(model_dir, "data_config.json"), 'w')
    json.dump(data_cfg, g)
    g.close()
    
    h = open(os.path.join(model_dir, "train_config.json"), 'w')
    json.dump(train_cfg, h)
    h.close()
    
if __name__ == '__main__':
    main()