import os
os.chdir("../")
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

fname = "../examples/data/star3_kh10_100.mat"
data = scipy.io.loadmat(fname)

coefs_all = data["coefs_all"]
uscat_all = data["uscat_all"].real
uscat_all = uscat_all[:, None, :, :] / np.std(uscat_all)
n_coefs = coefs_all.shape[1]

dataset = torch.utils.data.TensorDataset(
    torch.tensor(uscat_all, dtype=torch.float),
    torch.tensor(coefs_all, dtype=torch.float)
)
train_set, val_set = torch.utils.data.random_split(dataset, [96, 4])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=24)
uscat_val, coef_val = val_set[:]

class ConvNet(nn.Module):
    def __init__(self):
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
        if e % 10 == 0:
            coef_pred = model(uscat_val)
            print('Train Epoch: {:3}, Train Loss: {:.6f}, Val loss: {:.6f}'.format(
                e,
                current_loss / n_loss,
                loss_fn(coef_pred, coef_val).item())
            )
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
epoch = 500
train(model, device, train_loader, optimizer, epoch)

scipy.io.savemat(fname.replace(".mat", "_pred.mat"), {"coef_val": coef_val.numpy().astype('float64'), "coef_pred": coef_pred.detach().numpy().astype('float64')})