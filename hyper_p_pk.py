import os
import functools
from torchvision import transforms

from inlearn import conf
from inlearn.utils import data_utils


class ModelNet40_Hyper:
    def __init__(self, cam):
        self.cam = cam
        self.c = functools.reduce(lambda a, b: int(a) * int(b), self.cam.split('_')[:4])
        self.view_num = functools.reduce(lambda a, b: int(a) * int(b), self.cam.split('_')[:2])
        if os.path.basename(os.getenv('HOME')) == 'mat':
            data_root = '/home/mat/Data'
        else:
            data_root = '/media/tengyu/DataU/Data'

        self.data_root = data_root
        self.train_glob = os.path.join(data_root, 'ModelNet/ModelNet40_c%d/*/train/*/%s.pickle' % (self.c, self.cam))
        self.test_glob = os.path.join(data_root, 'ModelNet/ModelNet40_c%d/*/test/*/%s.pickle' % (self.c, self.cam))

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.all_cate = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
                         'bottle', 'bowl', 'car', 'chair', 'cone',
                         'cup', 'curtain', 'desk', 'door', 'dresser',
                         'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
                         'laptop', 'mantel', 'monitor', 'night_stand', 'person',
                         'piano', 'plant', 'radio', 'range_hood', 'sink',
                         'sofa', 'stairs', 'stool', 'table', 'tent',
                         'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.single = True

    def get_hyper_train(self):
        hyper_train_data = {
            'data_glob': self.train_glob,
            'all_cate': self.all_cate,
            'transform': self.transform,
            'single': self.single
        }

        hyper_train_loader = {
            'batch_size': 128,
            'shuffle': True
        }
        return hyper_train_data, hyper_train_loader

    def get_hyper_test(self):
        hyper_test_data = {
            'data_glob': self.test_glob,
            'all_cate': self.all_cate,
            'transform': self.transform,
            'single': self.single
        }

        hyper_test_loader = {
            'batch_size': 128,
            'shuffle': False
        }
        return hyper_test_data, hyper_test_loader

    def get_hyper_model(self, pre):
        hyper_model = {
            'single_grid': map(int, self.cam.split('_')[:2]),
            'pretrained': pre,
            'train': True,
            'save': True,
            'best_acc': 0,
            'start_epoch': 0,
            'lr': 0.1,
            'in_channels': 1,
            'out_classes': len(self.all_cate),
            'resume': False,
            'ckpt_dir': './ckpt/cvpr/tmp.pth'
        }
        return hyper_model

    def get_hyper_rst(self):
        hyper_rst = {
            'save': False,
            'rst_dir': './rst/cvpr/mvcnn_modelnet40_c%d.csv' % self.c
        }
        return hyper_rst


class SHREC17_Hyper:
    def __init__(self, cam):
        self.cam = cam
        self.c = functools.reduce(lambda a, b: int(a) * int(b), self.cam.split('_')[:4])
        if os.path.basename(os.getenv('HOME')) == 'mat':
            data_root = '/home/mat/Data'
        else:
            data_root = '/media/tengyu/DataU/Data'

        self.data_root = data_root
        self.train_glob = os.path.join(data_root, 'SHREC17/SHREC17_c%d/train/*/*/models/model_normalized/%s.pickle' % (self.c, self.cam))
        self.test_glob = os.path.join(data_root, 'SHREC17/SHREC17_c%d/test/*/*/models/model_normalized/%s.pickle' % (self.c, self.cam))

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.all_cate = ['02691156', '02818832', '02880940', '02954340', '03085013', '03337140', '03636649', '03761084',
                         '03948459', '04099429', '04401088', '02747177', '02828884', '02924116', '02958343', '03207941',
                         '03467517', '03642806', '03790512', '03991062', '04225987', '04460130', '02773838', '02843684',
                         '02933112', '02992529', '03211117', '03513137', '03691459', '03797390', '04004475', '04256520',
                         '04468005', '02801938', '02871439', '02942699', '03001627', '03261776', '03593526', '03710193',
                         '03928116', '04074963', '04330267', '04530566', '02808440', '02876657', '02946921', '03046257',
                         '03325088', '03624134', '03759954', '03938244', '04090263', '04379243', '04554684']
        self.single = True

    def get_hyper_train(self):
        hyper_train_data = {
            'data_glob': self.train_glob,
            'all_cate': self.all_cate,
            'transform': self.transform,
            'single': self.single
        }

        hyper_train_loader = {
            'batch_size': 128,
            'shuffle': True
        }
        return hyper_train_data, hyper_train_loader

    def get_hyper_test(self):
        hyper_test_data = {
            'data_glob': self.test_glob,
            'all_cate': self.all_cate,
            'transform': self.transform,
            'single': self.single
        }

        hyper_test_loader = {
            'batch_size': 128,
            'shuffle': False
        }
        return hyper_test_data, hyper_test_loader

    def get_hyper_model(self, pre):
        hyper_model = {
            'single_grid': map(int, self.cam.split('_')[:2]),
            'pretrained': pre,
            'train': True,
            'save': True,
            'best_acc': 0,
            'start_epoch': 0,
            'lr': 0.1,
            'in_channels': 1,
            'out_classes': len(self.all_cate),
            'resume': False,
            'ckpt_dir': './ckpt/cvpr/tmp.pth'
        }
        return hyper_model

    def get_hyper_rst(self):
        hyper_rst = {
            'save': False,
            'rst_dir': './rst/cvpr/mvcnn_SHREC17_c%d.csv' % self.c
        }
        return hyper_rst
