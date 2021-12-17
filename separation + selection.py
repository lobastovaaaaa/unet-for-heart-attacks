import torch
import numpy as np
import matplotlib.pyplot as plt


def get_balanced_subset(dset):
    with_lesion_ind = dset.y.sum(axis=0).sum(axis=0) != 0
    lesion_indices = np.arange(len(with_lesion_ind))[with_lesion_ind]
    return torch.utils.data.Subset(dset, lesion_indices)

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


the_x = torch.load('data.pt')  # данные
the_y = torch.load('label.pt')  # их метки

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
