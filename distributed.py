#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/3 21:30
# @Author  : xiezheng
# @Site    : 
# @File    : distributed.py


import csv
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
# import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter

from utils import get_logger, write_settings, output_process, AverageMeter, \
    get_learning_rate, accuracy, save_checkpoint


model_names = sorted(name for name in models.__dict__ if name.islower()
                     and not name.startswith("__") and callable(models.__dict__[name]))
# print(model_names)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/mnt/ssd/imagenet/', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + '(default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')

# parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
# parser.add_argument('--step', default=[30, 60], metavar='step decay', help='lr decay by step')
parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--step', default=[3,4], metavar='step decay', help='lr decay by step')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number ()')
parser.add_argument('-b', '--batch-size', default=1200, type=int, metavar='N',
                    help='mini-batch size (default: 3200), this is the total batch size of all GPUs on the current node '
                         'when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=False, type=bool, help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=False, type=bool, help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')

# distributed
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

parser.add_argument('--gpus', default='3,6,7', metavar='gpus_id', help='N gpus for training')
parser.add_argument('--outpath', metavar='DIR', default='./output_ddp', help='path to output')
parser.add_argument('--lr-scheduler', metavar='LR scheduler', default='steplr', help='LR scheduler', dest='lr_scheduler')
parser.add_argument('--gamma', default=0.1, type=float, metavar='gamma', help='gamma')
# parser.print_help()
# assert False, 'Stop !'


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    args = parser.parse_args()
    # args = parser.parse_args('--pretrained'.split())
    # print(args)
    # assert False, 'Stop !'

    if args.seed is not None:
        # setting seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)    # Add
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training.'
                      'This will turn on the cudnn deterministic setting,'
                      'which can slow down your training considerably!'
                      'You may see unexpected behavior when restarting from checkpoint.')
    else:
        cudnn.benchmark = True
    main_worker(args.local_rank, args=args)


def main_worker(local_rank, args):
    best_acc1 = 0
    best_acc1_index = 0

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.outpath = args.outpath + '_' + args.arch
    output_process(args.outpath)
    logger = get_logger(args.outpath, 'DataParallel')
    writer = SummaryWriter(args.outpath)

    # distributed init
    args.nprocs = torch.cuda.device_count()
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(device=local_rank)

    write_settings(args)
    logger.info(args)

    # create model
    if args.pretrained:
        logger.info("=> using pre-trained model: {}".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        logger.info('=> creating model: {}'.format(args.arch))
        model = models.__dict__[args.arch]()

    model = model.cuda(device=local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / args.nprocs)
    model = DDP(model, device_ids=[local_rank])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device=local_rank)
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'steplr':
        lr_scheduler = MultiStepLR(optimizer, milestones=args.step, gamma=args.gamma)
        logger.info('lr_scheduler: SGD MultiStepLR !!!')
    else:
        assert False, logger.info("invalid lr_scheduler={}".format(args.lr_scheduler))
    # logger.info('lr_scheduler={}'.format(lr_scheduler))

    # dataloader
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                              pin_memory=True, sampler=train_sampler)

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args, logger, writer, epoch=-1, local_rank=local_rank)
        return 0

    total_start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # DDP
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        epoch_start = time.time()
        lr_scheduler.step(epoch)

        # train for every epoch
        train(train_loader, model, criterion, optimizer, epoch, args, logger, writer, local_rank)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, logger, writer, epoch, local_rank)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1_index = epoch
            best_acc1 = acc1

        epoch_end = time.time()
        if args.local_rank == 0:
            logger.info('||==> Epoch=[{:d}/{:d}]\tbest_acc1={:.4f}\tbest_acc1_index={}\ttime_cost={:.4f}s'
                        .format(epoch, args.epochs, best_acc1, best_acc1_index, epoch_end - epoch_start))

            # save model
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best, args.outpath)

    total_end = time.time()
    if args.local_rank == 0:
        logger.info('||==> total_time_cost={:.4f}s'.format(total_end - total_start))
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, args, logger, writer, local_rank):
    batch_times = AverageMeter('Time', ':6.3f')
    data_times  = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')   # 4e表示科学记数法中的4位小数
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to train mode
    model.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time = time.time() - end

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, 1)

        # DDP: data synchronization
        dist.barrier()
        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_data_time = reduce_mean(data_time, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1, images.size(0))
        data_times.update(reduced_data_time)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time = time.time() - end
        reduced_batch_time = reduce_mean(batch_time, args.nprocs)
        batch_times.update(reduced_batch_time)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            logger.info('Train epoch: [{:d}/{:d}][{:d}/{:d}]\tlr={:.6f}\tce_loss={:.4f}\ttop1_acc={:.4f}\tdata_time={:6.3f}s'
                        '\tbatch_time={:6.3f}s'.format(epoch, args.epochs, i, len(train_loader), get_learning_rate(optimizer),
                                                      losses.avg, top1.avg, data_times.avg, batch_times.avg))
        # break

    if args.local_rank == 0:
        # save tensorboard
        writer.add_scalar('lr', get_learning_rate(optimizer), epoch)
        writer.add_scalar('Train_ce_loss', losses.avg, epoch)
        writer.add_scalar('Train_top1_accuracy', top1.avg, epoch)

        logger.info('||==> Train epoch: [{:d}/{:d}]\tlr={:.6f}\tce_loss={:.4f}\ttop1_acc={:.4f}\tbatch_time={:6.3f}s'
                    .format(epoch, args.epochs, get_learning_rate(optimizer), losses.avg, top1.avg, batch_times.avg))


def validate(val_loader, model, criterion, args, logger, writer, epoch, local_rank):
    batch_times = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')   # 4e表示科学记数法中的4位小数
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, 1)

            # DDP: data synchronization
            dist.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1, images.size(0))

            # measure elapsed time
            batch_time = time.time() - end
            reduced_batch_time = reduce_mean(batch_time, args.nprocs)
            batch_times.update(reduced_batch_time)
            end = time.time()

            if args.local_rank == 0 and i % args.print_freq == 0:
                logger.info('Val epoch: [{:d}/{:d}][{:d}/{:d}]\tce_loss={:.4f}\ttop1_acc={:.4f}\tbatch_time={:6.3f}s'
                            .format(epoch, args.epochs, i, len(val_loader), losses.avg, top1.avg, batch_times.avg))
            # break

        if args.local_rank == 0:
            # save tensorboard
            writer.add_scalar('Val_ce_loss', losses.avg, epoch)
            writer.add_scalar('Val_top1_accuracy', top1.avg, epoch)

            logger.info('||==> Val epoch: [{:d}/{:d}]\tce_loss={:.4f}\ttop1_acc={:.4f}\tbatch_time={:6.3f}s'
                        .format(epoch, args.epochs, losses.avg, top1.avg, batch_times.avg))

        return top1.avg


if __name__ == '__main__':
    main()
