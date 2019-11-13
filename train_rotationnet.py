import argparse
import os
import glob
import time
import shutil
import itertools

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np

from pickle_loader import pickle_train_test_loader
from argparse import Namespace
from hyper_p_pk import ModelNet40_Hyper


class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        else:
            raise ("Finetuning not supported on this architecture yet")

        # padding 0 to make sure the small views not shrink to 0 size
        for i, m in self.features._modules.items():
            if m.__class__.__name__ == 'MaxPool2d':
                if m.padding == 0:
                    m.padding = 1
        # conv1
        self.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet':
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        nsamp = int(input.size(0) / args.nview)

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)
        target_ = torch.LongTensor(target.size(0) * args.nview)

        # compute output
        output = model(input_var)
        num_classes = int(output.size(1) / args.nview) - 1
        output = output.view(-1, num_classes + 1)

        ###########################################
        # compute scores and decide target labels #
        ###########################################
        output_ = torch.nn.functional.log_softmax(output, dim=1)
        # divide object scores by the scores for "incorrect view label" (see Eq.(5))
        output_ = output_[:, :-1] - torch.t(output_[:, -1].repeat(1, output_.size(1) - 1).view(output_.size(1) - 1, -1))
        # reshape output matrix
        output_ = output_.view(-1, args.nview * args.nview, num_classes)
        output_ = output_.data.cpu().numpy()
        output_ = output_.transpose(1, 2, 0)
        # initialize target labels with "incorrect view label"
        for j in range(target_.size(0)):
            target_[j] = num_classes
        # compute scores for all the candidate poses (see Eq.(5))
        scores = np.zeros((args.vcand.shape[0], num_classes, nsamp))
        for j in range(args.vcand.shape[0]):
            for k in range(args.vcand.shape[1]):
                scores[j] = scores[j] + output_[args.vcand[j][k] * args.nview + k]
        # for each sample #n, determine the best pose that maximizes the score for the target class (see Eq.(2))
        for n in range(nsamp):
            j_max = np.argmax(scores[:, target[n * args.nview], n])
            # assign target labels
            for k in range(args.vcand.shape[1]):
                target_[n * args.nview * args.nview + args.vcand[j_max][k] * args.nview + k] = target[n * args.nview]
        ###########################################

        target_ = target_.cuda()
        target_var = torch.autograd.Variable(target_)

        # compute loss
        loss = criterion(output, target_var)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # log_softmax and reshape output
        num_classes = int(output.size(1) / args.nview) - 1
        output = output.view(-1, num_classes + 1)
        output = torch.nn.functional.log_softmax(output, dim=1)
        output = output[:, :-1] - torch.t(output[:, -1].repeat(1, output.size(1) - 1).view(output.size(1) - 1, -1))
        output = output.view(-1, args.nview * args.nview, num_classes)

        # measure accuracy and record loss
        prec1, prec5 = my_accuracy(output.data, target, args, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0) / args.nview)
        top5.update(prec5.item(), input.size(0) / args.nview)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='rotationnet_checkpoint.pth.tar',
                    filename2='rotationnet_model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename2)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def my_accuracy(output_, target, args, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    target = target[0:-1:args.nview]
    batch_size = target.size(0)

    num_classes = output_.size(2)
    output_ = output_.cpu().numpy()
    output_ = output_.transpose(1, 2, 0)
    scores = np.zeros((args.vcand.shape[0], num_classes, batch_size))
    output = torch.zeros((batch_size, num_classes))
    # compute scores for all the candidate poses (see Eq.(6))
    for j in range(args.vcand.shape[0]):
        for k in range(args.vcand.shape[1]):
            scores[j] = scores[j] + output_[args.vcand[j][k] * args.nview + k]
    # for each sample #n, determine the best pose that maximizes the score (for the top class)
    for n in range(batch_size):
        j_max = int(np.argmax(scores[:, :, n]) / scores.shape[1])
        output[n] = torch.FloatTensor(scores[j_max, :, n])
    output = output.cuda()

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def run_rxp(args, exp_settings, Data_Hyper, run_times):
    for exp_i in range(run_times):
        for cam, pre in exp_settings:
            # load data
            args.pretrained = pre
            data_hyper = Data_Hyper(cam)

            best_prec1 = 0
            args.nview = data_hyper.view_num
            # no rotation case as stated in the RotationNet paper
            args.vcand = np.array([np.roll(range(args.nview), -i) for i in range(args.nview)])

            # make the batch size is a multiplication of nview so every minibatch will have all views of an object
            args.batch_size = 512 // args.nview * args.nview
            if args.batch_size % args.nview != 0:
                print('Error: batch size should be multiplication of the number of views,', args.nview)
                exit()

            # Get number of classes from train directory
            num_classes = len(glob.glob(args.data + '/*'))
            print("num_classes = '{}'".format(num_classes))

            # create model
            if args.pretrained:
                print("=> using pre-trained model '{}'".format(args.arch))
                model = models.__dict__[args.arch](pretrained=True)
            else:
                print("=> creating model '{}'".format(args.arch))
                model = models.__dict__[args.arch]()

            train_loader, test_loader = pickle_train_test_loader(data_hyper, args)

            train_dataset, test_dataset = train_loader.dataset, test_loader.dataset

            # load model
            model = FineTuneModel(model, args.arch, (num_classes + 1) * args.nview)
        
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
        
            # define loss function (criterion) and optimizer
            criterion = nn.CrossEntropyLoss().cuda()
        
            # optimizer = torch.optim.SGD(model.parameters(), args.lr,
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),  # Only finetunable params
                                        args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            scheduler = StepLR(optimizer, step_size=200, gamma=0.1)
        
            # optionally resume from a checkpoint
            if args.resume:
                if os.path.isfile(args.resume):
                    print("=> loading checkpoint '{}'".format(args.resume))
                    checkpoint = torch.load(args.resume)
                    args.start_epoch = checkpoint['epoch']
                    best_prec1 = checkpoint['best_prec1']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(args.resume, checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'".format(args.resume))
        
            cudnn.benchmark = True
        
            if args.evaluate:
                validate(test_loader, model, criterion, args)
                return
        
            for epoch in range(args.start_epoch, args.epochs):
                # train for one epoch
                train(train_loader, model, criterion, optimizer, epoch, args)

                # update learning rate
                scheduler.step(epoch)

                # evaluate on validation set
                prec1 = validate(test_loader, model, criterion, args)
        
                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                fname = 'rotationnet_checkpoint.pth.tar'
                fname2 = 'rotationnet_model_best.pth.tar'
        
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, fname, fname2)
        
            rst_dir = './rst/cvpr/rotationnet_c%d_2.csv' % data_hyper.c
            with open(rst_dir, 'a+') as f:
                line = '%s\t%s\t%s\t%f\n' % (cam,
                                             'rotationnet_resnet18',
                                             args.pretrained,
                                             best_prec1)
                f.write(line)
        
        
def main():
    args = Namespace(
        arch='resnet18',
        data='/media/tengyu/DataU/Data/ModelNet/ModelNet40',
        dist_backend='gloo',
        dist_url='tcp://224.66.41.62:23456',
        epochs=1500,
        evaluate=False,
        lr=0.01,
        momentum=0.9,
        pretrained=True,
        print_freq=10,
        resume='',
        start_epoch=0,
        weight_decay=0.0001,
        workers=4,
    )
    # c10000
    # cam_settings = ['4_1_50_50_0.02', '8_2_25_25_0.02', '20_5_10_10_0.02']
    #                 '40_10_5_5_0.02', '100_25_2_2_0.02', '200_50_1_1_0.02']

    # c22500
    # cam_settings = ['4_1_75_75_0.01', '12_3_25_25_0.01', '20_5_15_15_0.01']
                    # '60_15_5_5_0.01', '100_25_3_3_0.01', '300_75_1_1_0.01']

    # c40000
    # cam_settings = ['4_1_100_100_0.01', '8_2_50_50_0.01', '16_4_25_25_0.01', '20_5_20_20_0.01']
                    # '40_10_10_10_0.01', '80_20_5_5_0.01', '100_25_4_4_0.01', '200_50_2_2_0.01', '400_100_1_1_0.01']

    # cam_settings = ['4_1_50_50_0.02', '8_2_25_25_0.02', '4_1_75_75_0.01', '12_3_25_25_0.01', '4_1_100_100_0.01', '8_2_50_50_0.01']
    cam_settings = ['12_3_25_25_0.01']
    pretraineds = [False]
    exp_settings = list(itertools.product(*[cam_settings, pretraineds]))
    Data_Hyper = ModelNet40_Hyper
    run_times = 1

    run_rxp(args, exp_settings, Data_Hyper, run_times)
    

if __name__ == '__main__':
    main()
