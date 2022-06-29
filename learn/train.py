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
    parser.add_argument("--model_name", default="Fourier", type=str)

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

def main():
    args, train_cfg = parse_args()
    print("Train data from {}".format(args.dirname))
    fname = os.path.join(args.dirname, "forward_data.mat")
    data = scipy.io.loadmat(fname)
    coefs_all = data["coefs_all"]
    uscat_all = data["uscat_all"]
    data_to_train = uscat_all
    if args.model_name == 'Fourier':
        uscat_ft = sfft.fft2(uscat_all)
        uscat_ft_shift = sfft.fftshift(uscat_ft,axes=(1,2))
        data_to_train = uscat_ft_shift
    
    if train_cfg["network_type"] == 'convnet':
        data_to_train = data_to_train.real
        print("The mean value is", np.mean(data_to_train))
        std = np.std(data_to_train)
        data_to_train = data_to_train[:, None, :, :] / std
        data_cfg = json.loads(data["cfg_str"][0])
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(data_to_train, dtype=torch.float),
            torch.tensor(coefs_all, dtype=torch.float)
        )
    elif train_cfg["network_type"] == 'complexnet':
        data_real = data_to_train.real
        data_imag = data_to_train.imag
        print("The mean values are", np.mean(data_real), np.mean(data_imag))
        std_r = np.std(data_real)
        std_i = np.std(data_imag)
        std = (std_r**2 + std_i**2)**0.5
        data_real = data_real[:, None, :, :] / std
        data_imag = data_imag[:, None, :, :] / std
        data_cfg = json.loads(data["cfg_str"][0])
    
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(data_real, dtype=torch.float),
            torch.tensor(data_imag, dtype=torch.float),
            torch.tensor(coefs_all, dtype=torch.float)
        )
    ndata = coefs_all.shape[0]
    nval = min(100, int(ndata*0.05))
    ntrain = ndata - nval
    train_set, val_set = torch.utils.data.random_split(dataset, [ntrain, nval], generator=torch.Generator().manual_seed(train_cfg["seed"]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_cfg["batch_size"])
    if train_cfg["network_type"] == 'convnet':
        uscat_val, coef_val = val_set[:]
    elif train_cfg["network_type"] == 'complexnet':
        ft_val_real, ft_val_imag, coef_val = val_set[:]

    loss_fn = nn.MSELoss()
    log_dir=os.path.join(args.dirname, args.model_name)
    writer = SummaryWriter(log_dir)
    if train_cfg["network_type"] == 'convnet':
        def train(model, device, train_loader, optimizer, epoch, scheduler):
            for e in range(epoch):
                n_loss = 0
                current_loss = 0.0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
                    n_loss += 1
                    current_loss += loss.item()
                if e % train_cfg["valid_freq"] == 0:
                    coef_pred = model(uscat_val)
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
    elif train_cfg["network_type"] == 'complexnet':
        # write two train to avoid 'if else' in each iteration
        def train(model, device, train_loader, optimizer, epoch, scheduler):
            for e in range(epoch):
                n_loss = 0
                current_loss = 0.0
                for batch_idx, (data_r, data_c, target) in enumerate(train_loader):
                    data_r, data_c, target = data_r.to(device), data_c.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data_r, data_c)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
                    n_loss += 1
                    current_loss += loss.item()
                if e % train_cfg["valid_freq"] == 0:
                    coef_pred = model(ft_val_real, ft_val_imag)
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
    
    if train_cfg["network_type"] == 'convnet':
        uscat_val = uscat_val.to(device)
        model = network.ConvNet(data_cfg, train_cfg).to(device)
    elif train_cfg["network_type"] == 'complexnet':
        ft_val_real = ft_val_real.to(device)
        ft_val_imag = ft_val_imag.to(device)
        model = network.ComplexNet(data_cfg, train_cfg).to(device)
    # TODO: test performance of ADAM and other learning rates
    if train_cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg["lr"], momentum=train_cfg["momentum"])
    elif train_cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    epoch = train_cfg["epoch"]
    scheduler = MultiStepLR(optimizer, milestones=train_cfg["milestones"], gamma=train_cfg["gamma"])
    # TODO: add functionality to re-train
    train(model, device, train_loader, optimizer, epoch, scheduler)
    if train_cfg["network_type"] == 'convnet':
        coef_pred = model(uscat_val)
    elif train_cfg["network_type"] == 'complexnet':
        coef_pred = model(ft_val_real, ft_val_imag)
    writer.close()

    scipy.io.savemat(
        os.path.join(args.dirname, "valid_predby_{}.mat".format(args.model_name)),
        {
            "coef_val": coef_val.numpy().astype('float64'),
            "coef_pred": coef_pred.detach().cpu().numpy().astype('float64'),
            "cfg_str": data["cfg_str"][0]
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