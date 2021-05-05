#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/4 17:13
# @Author  : xiezheng
# @Site    : 
# @File    : utils.py


import logging
import os
import sys
import shutil

import torch


def get_logger(save_path, logger_name):
    """
    Initialize logger
    """

    logger = logging.getLogger(logger_name)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')

    # file log
    file_handler = logging.FileHandler(os.path.join(save_path, "experiment.log"))
    file_handler.setFormatter(file_formatter)

    # console log
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def output_process(output_path):
    if os.path.exists(output_path):
        print("{} file exist!".format(output_path))
        action = input("Select Action: d (delete) / q (quit):").lower().strip()
        act = action
        if act == 'd':
            shutil.rmtree(output_path)
        else:
            raise OSError("Directory {} exits!".format(output_path))

    if not os.path.exists(output_path):
        os.makedirs(output_path)


def write_settings(settings):
    """
    Save expriment settings to a file
    :param settings: the instance of option
    """

    with open(os.path.join(settings.outpath, "settings.log"), "w") as f:
        for k, v in settings.__dict__.items():
            f.write(str(k) + ": " + str(v) + "\n")


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr[0]


# Follow project
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# class ProgressMeter(object):
#     def __init__(self, num_batches, meters, prefix="", logger=None):
#         self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
#         self.meters = meters
#         self.prefix = prefix
#         self.logger = logger
#
#     def display(self, batch):
#         entries = [self.prefix + self.batch_fmtstr.format(batch)]
#         entries += [str(meter) for meter in self.meters]
#         self.logger.info('\t'.join(entries))
#
#     def _get_batch_fmtstr(self, num_batches):
#         num_digits = len(str(num_batches // 1))
#         fmt = '{:' + str(num_digits) + 'd}'
#         return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')