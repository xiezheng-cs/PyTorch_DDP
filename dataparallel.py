#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/3 20:33
# @Author  : xiezheng
# @Site    : 
# @File    : dataparallel.py


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
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
# print(model_names)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='./xxx', help='path to dataset')
# parser.print_help()
