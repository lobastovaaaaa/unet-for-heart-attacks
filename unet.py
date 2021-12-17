import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataset as dset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
import torchvision
import matplotlib.pyplot as plt
import numpy as np


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)

        return self.relu(tmp)


class Encoder(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            x = self.pool(x)
            ftrs.append(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.dec_blocks[i](x)
        return x


class UNet(nn.Module):
    def __init__(self, enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = torch.nn.functional.interpolate(out, (256, 256))
        return out


class NiftiDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, x_s, y_s, transform=None):
        self.x_s = x_s
        self.y_s = y_s
        self.transform = transform

        x_list = [*self.x_s]
        y_list = [*self.y_s.float()]

        self.x = torch.cat(x_list, axis=-1)
        self.y = torch.cat(y_list, axis=-1)

    def __repr__(self):
        return f'NiftiDataset of shape: {self.x.shape}'

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x[..., idx]
        y = self.y[..., idx]

        if self.transform:
            augmentation = self.transform(image=x, mask=y)
            x = augmentation['image']
            y = augmentation['mask']
        return x, y

    def __len__(self):
        return self.y.shape[-1]


def get_balanced_subset(dset):
    with_lesion_ind = dset.y.sum(axis=0).sum(axis=0) != 0
    lesion_indices = np.arange(len(with_lesion_ind))[with_lesion_ind]
    return torch.utils.data.Subset(dset, lesion_indices)


def compute_accuracy(model, loader):
    model.eval()
    accuracy = 0
    for (x, y) in loader:
        x = x[:, None, ...]
        y = y[:, None, ...]
        prediction = model(x)
        accuracy += dice_loss(prediction, y)
    return accuracy


def dice_loss(pred, answ):
    accuracy = 0
    for i in range(len(pred)):
        prediction = pred[i][0]
        answer = answ[i][0]
        intersection = (prediction * answer).sum()
        a_sum = torch.sum(prediction)
        b_sum = torch.sum(answer)
        accuracy += (2. * intersection) / (a_sum + b_sum)
    return accuracy / batch_size


def train_model(model, train_loader, val_loader, loss, optimizer, scheduler, num_epochs):
    loss_history = []
    train_history = []
    val_history = []
    for epoch in range(num_epochs):
        model.train()
        loss_accum = 0
        for i_step, (x, y) in enumerate(train_loader):
            x = x[:, None, ...]
            y = y[:, None, ...]
            prediction = model(x)
            loss_value = loss(prediction, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            loss_accum += dice_loss(prediction, y)
        scheduler.step()
        ave_loss = loss_accum / (i_step + 1)
        train_accuracy = dice_loss(prediction, y)
        val_accuracy = compute_accuracy(model, val_loader)
        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f, LR: %f" % (ave_loss,
                                                                                  train_accuracy,
                                                                                  val_accuracy,
                                                                                  step_lr.get_last_lr()[0]))
    return loss_history, train_history, val_history


the_x = torch.load('data.pt')
the_y = torch.load('label.pt')

data_size = the_x.data.shape[0]
test_split = .2
split = int(np.floor(test_split * data_size))
indices = list(range(data_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

the_train_x = the_x[train_indices]
the_train_y = the_y[train_indices]
full_train_dataset = NiftiDataset(the_train_x, the_train_y)
train_dataset = get_balanced_subset(full_train_dataset)

the_test_x = the_x[test_indices]
the_test_y = the_y[test_indices]
full_test_dataset = NiftiDataset(the_test_x, the_test_y)
test_dataset = get_balanced_subset(full_test_dataset)

batch_size = 5
data_size = len(train_dataset)

validation_split = .2
split = int(np.floor(validation_split * data_size))
indices = list(range(data_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         sampler=val_sampler)
sample, label = train_dataset[0]

nn_model = nn.Sequential(UNet())
optimizer = optim.Adam(nn_model.parameters(), lr=1e-2, weight_decay=1e-1)
loss = nn.BCEWithLogitsLoss()
step_lr = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

loss_history, train_history, val_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, step_lr, 5)
