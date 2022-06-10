import os
import json
import argparse
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", default="./data/star3_kh10_100", type=str)
    parser.add_argument("--expname", default="test", type=str)
    parser.add_argument("--cfgpath", default=None, type=str)
    args = parser.parse_args()
    if args.cfgpath is None:
        dirname = os.path.basename(args.dirname)
        ncstr = dirname.split('_')[0]
        if ncstr.startswith("star"):
            try:
                nc = int(ncstr[4:])
            except ValueError:
                print("Error: cannot get the default config path from dirname.")
                raise
        args.cfgpath = "./configs/train_nc{}.json".format(nc)
    f = open(args.cfgpath)
    train_cfg = json.load(f)
    f.close()
    return args, train_cfg

def main():
    args, train_cfg = parse_args()
    print("Train data from {}".format(args.dirname))
    fname = os.path.join(args.dirname, "forward_data.mat")
    data = scipy.io.loadmat(fname)
    coefs_all = data["coefs_all"]
    uscat_all = data["uscat_all"].real
    # TODO: need to save std in order to re-evaluate
    print("The mean value is", np.mean(uscat_all))
    std = np.std(uscat_all)
    uscat_all = uscat_all[:, None, :, :] / std
    data_cfg = json.loads(data["cfg_str"][0])

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(uscat_all, dtype=torch.float),
        torch.tensor(coefs_all, dtype=torch.float)
    )
    n_coefs = coefs_all.shape[1]
    ndata = coefs_all.shape[0]
    nval = min(100, int(ndata*0.05))
    ntrain = ndata - nval
    train_set, val_set = torch.utils.data.random_split(dataset, [ntrain, nval], generator=torch.Generator().manual_seed(train_cfg["seed"]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_cfg["batch_size"])
    uscat_val, coef_val = val_set[:]

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

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    loss_fn = nn.MSELoss()
    log_dir=os.path.join(args.dirname, args.expname)
    writer = SummaryWriter(log_dir)

    def train(model, device, train_loader, optimizer, epoch):
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
                loss_val = loss_fn(coef_pred, coef_val).item()
                print('Train Epoch: {:3}, Train Loss: {:.6f}, Val loss: {:.6f}'.format(
                    e, loss_train, loss_val)
                )
                writer.add_scalar('loss_train', loss_train, e)
                writer.add_scalar('loss_val', loss_val, e)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNet().to(device)
    # TODO: test performance of ADAM and other learning rates
    if train_cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg["lr"], momentum=train_cfg["momentum"])
    epoch = train_cfg["epoch"]
    # TODO: add functionality to re-train
    train(model, device, train_loader, optimizer, epoch)
    coef_pred = model(uscat_val)
    writer.close()

    scipy.io.savemat(
        os.path.join(args.dirname, "{}_pred.mat".format(args.expname)),
        {
            "coef_val": coef_val.numpy().astype('float64'),
            "coef_pred": coef_pred.detach().numpy().astype('float64'),
            "cfg_str": data["cfg_str"][0]
        }
    )
    torch.save(model.state_dict(), os.path.join(args.dirname, "{}.pt".format(args.expname)))
    np.save("std.npy", std)

if __name__ == '__main__':
    main()