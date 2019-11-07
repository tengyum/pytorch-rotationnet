import os
import glob
import pickle
import torch
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt
from inlearn.utils import data_utils
from hyper_p_pk import ModelNet40_Hyper


class PickleDataset(Dataset):
    def __init__(self, data_glob, all_cate, transform, single):
        self.data_glob = data_glob
        self.all_cate = all_cate
        self.cam = list(map(int, os.path.splitext(os.path.basename(data_glob))[0].split('_')[:4]))
        self.view_grid = self.cam[:2]
        self.view_num = self.view_grid[0] * self.view_grid[1]
        self.single = single

        if single:
            self.pickle_dirs = [(pk, i) for pk in sorted(glob.glob(data_glob)) for i in range(self.view_num)]
        else:
            self.pickle_dirs = list(map(lambda pk: (pk, 0), sorted(glob.glob(data_glob))))

        self.labels = list(map(self._get_label, self.pickle_dirs))
        self.transform = transform

    def _get_label(self, dir_):
        for s in dir_[0].split('/'):
            if s in self.all_cate:
                return self.all_cate.index(s)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_dir, view = self.pickle_dirs[index]
        with open(img_dir, 'rb') as f:
            x = pickle.load(f)

        h, w = x.shape
        w_view, h_view = self.view_grid
        x = x.reshape(h_view, h // h_view, w_view, w // w_view).transpose(0, 2, 1, 3).reshape(-1, h // h_view, w // w_view)

        if self.single:
            x = x[view]
        else:
            x = x.transpose(1, 2, 0)

        if self.transform:
            x = self.transform(x)

        y = self.labels[index]
        return x, y


def pickle_loader(hyper_p_data, hyper_p_loader):
    data = PickleDataset(**hyper_p_data)
    data_loader = DataLoader(data, **hyper_p_loader)
    return data_loader


def pickle_train_test_loader(hyper_train_data, hyper_train_loader, hyper_test_data, hyper_test_loader):
    train_loader = pickle_loader(hyper_train_data, hyper_train_loader)

    mean, std = data_utils.get_mean_std(train_loader)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_loader.dataset.transform = train_transform
    data_utils.get_mean_std(train_loader)

    test_loader = pickle_loader(hyper_test_data, hyper_test_loader)
    test_loader.dataset.transform = train_transform

    return train_loader, test_loader


def main():
    cam = '4_1_100_100_0.01'
    modelnet40_hyper = ModelNet40_Hyper(cam)
    trainloader, testloader = pickle_train_test_loader(*modelnet40_hyper.get_hyper_train(),
                                                       *modelnet40_hyper.get_hyper_test())


if __name__ == '__main__':
    main()
