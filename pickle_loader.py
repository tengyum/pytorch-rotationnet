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
from inlearn import conf
from inlearn.utils import data_utils
from inlearn.exp.hyper_p_pk import ModelNet40_Hyper, SHREC17_Hyper


class PickleDataset(Dataset):
    def __init__(self, name, data_glob, all_cate, transform, load_type):
        self.name = name
        self.data_glob = data_glob
        self.all_cate = all_cate
        self.cam = list(map(int, os.path.splitext(os.path.basename(data_glob))[0].split('_')[:4]))
        self.view_grid = self.cam[:2]
        self.view_num = self.view_grid[0] * self.view_grid[1]
        self.load_type = load_type

        self.raw_pickle_dirs = sorted(glob.glob(data_glob))
        self.raw_labels = list(map(self._get_label, self.raw_pickle_dirs))

        if load_type == 'single':
            self.pickle_dirs = [(pk, i) for pk in sorted(glob.glob(data_glob)) for i in range(self.view_num)]
            self.labels = [label for label in self.raw_labels for _ in range(self.view_num)]
        else:
            self.pickle_dirs = list(map(lambda pk: (pk, 0), sorted(glob.glob(data_glob))))
            self.labels = self.raw_labels

        self.transform = transform

    def _get_label(self, dir_):
        if self.name == 'modelnet40':
            return self.all_cate.index(dir_.split('/')[-4])
        elif self.name == 'shrec17':
            return self.all_cate.index(dir_.split('/')[-5])
        else:
            raise Exception('invalid dataset name %s' % self.name)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_dir, view = self.pickle_dirs[index]
        with open(img_dir, 'rb') as f:
            x = pickle.load(f)

        if not self.load_type == 'whole':
            h, w = x.shape
            w_view, h_view = self.view_grid
            x = x.reshape(h_view, h // h_view, w_view, w // w_view).transpose(0, 2, 1, 3).reshape(-1, h // h_view, w // w_view)
            if self.load_type == 'single':
                x = x[view]
            elif self.load_type == 'channel':
                x = x.transpose(1, 2, 0)
            else:
                raise Exception('invalid load type %s' % self.load_type)

        if self.transform:
            x = self.transform(x)

        y = self.labels[index]
        return x, y


def record_mean_std(hyper_train_data, hyper_train_loader):
    train_loader = pickle_loader(hyper_train_data, hyper_train_loader)

    mean, std = data_utils.get_mean_std(train_loader)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_loader.dataset.transform = train_transform
    data_utils.get_mean_std(train_loader)

    return mean, std


def read_mean_std(name, cam):
    if name == 'modelnet40':
        f = open(os.path.join(conf.PJ_ROOT, 'data/modelnet40_mean_std.txt'), 'r')
    elif name == 'shrec17':
        f = open(os.path.join(conf.PJ_ROOT, 'data/shrec17_mean_std.txt'), 'r')
    else:
        raise Exception('dataset %s does not exist' % name)

    for l in f:
            cam_, mean, std = l.strip().split()
            if cam_ == cam:
                return float(mean), float(std)
    print('Can not find the mean, std for dataset %s and cam %s' % (name, cam))


def pickle_loader(hyper_p_data, hyper_p_loader):
    data = PickleDataset(**hyper_p_data)
    data_loader = DataLoader(data, **hyper_p_loader)
    return data_loader


def pickle_train_test_loader(data_hyper, args):
    train_loader = pickle_loader(*data_hyper.get_hyper_train(args.batch_size))

    mean, std = read_mean_std(data_hyper.name, data_hyper.cam)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([mean], [std])
    ])
    train_loader.dataset.transform = train_transform

    test_loader = pickle_loader(*data_hyper.get_hyper_test(args.batch_size))
    test_loader.dataset.transform = train_transform

    return train_loader, test_loader


def main():
    # c10000
    cam_settings1 = ['4_1_50_50_0.02', '8_2_25_25_0.02', '20_5_10_10_0.02', '40_10_5_5_0.02', '100_25_2_2_0.02', '200_50_1_1_0.02']

    # c22500
    cam_settings2 = ['4_1_75_75_0.01', '12_3_25_25_0.01', '20_5_15_15_0.01', '60_15_5_5_0.01', '100_25_3_3_0.01', '300_75_1_1_0.01']

    # c40000
    cam_settings3 = ['4_1_100_100_0.01', '8_2_50_50_0.01', '16_4_25_25_0.01', '20_5_20_20_0.01',
                     '40_10_10_10_0.01', '80_20_5_5_0.01', '100_25_4_4_0.01', '200_50_2_2_0.01', '400_100_1_1_0.01']

    cams = cam_settings1 + cam_settings2 + cam_settings3

    for cam in cams:
        mean, std = read_mean_std('modelnet40', cam)
        print(mean, std)


if __name__ == '__main__':
    main()
