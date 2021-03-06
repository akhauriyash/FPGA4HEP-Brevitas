# MIT License

# Copyright (c) 2019 Xilinx

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function
import time, os, matplotlib, argparse, torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from utils.get_features import get_features
from utils.graphing import makeRoc, plot_confusion_matrix, plt_conf_mat
from utils.util import counter, counter_fanout, parse_config
from train import train, test
from models.LFC import LFC


matplotlib.use('agg')

parser = argparse.ArgumentParser(description='FPGANet FPGA4HEP Example')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 1024)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Set CUDA')
parser.add_argument('--name', type=str, default=None, 
                    help='Provide Model Name')
parser.add_argument('--model-save-path', type=str, default='.', 
                    help='Provide Folder Path to save model to')
parser.add_argument('--test', type=bool, default=False,
                    help='Set True if testing')
parser.add_argument('--gpu', type=int, default=None,
                    help='which specific gpu to use')
args = parser.parse_args()

if args.gpu!=None:
    torch.cuda.set_device(args.gpu)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


global maxAcc
maxAcc = 0

if(args.cuda==True):
    print("GPU in use", torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count(), torch.cuda.get_device_name(torch.cuda.current_device()))
kwargs = {'num_workers': 4, 'pin_memory': False} if args.cuda else {}

if(args.name==None):
    name = 'FPGA4HEPmodel'
else:
    name = args.name

yamlConfig = parse_config('./yaml_IP_OP_config.yml')
print("Reading dataset...")
X_train_val, X_test, y_train_val, y_test, labels, train_loader, test_loader, input_shape, output_shape  = get_features(yamlConfig, args.batch_size, args.test_batch_size)

dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

model = LFC(num_classes=5, weight_bit_width=8, act_bit_width=8, in_bit_width=8)
if args.cuda:
    model.cuda()
print(model)

## Optimizers
optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=0.0001)
## Scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(args.epochs*x/100) for x in [10, 20, 30, 40, 60, 80]])
## Criterion
criterion = nn.CrossEntropyLoss()
if args.cuda==True:
    criterion = criterion.cuda()

## Train - Test Loop
if(args.test==False):
    for epoch in range(args.epochs):
        train(epoch, model, train_loader, criterion, optimizer, args)
        maxAcc = test(model, name, maxAcc, test_loader, criterion, optimizer, args)
        scheduler.step()
else:
    print("Loading model...")
    model.load_state_dict(torch.load('./../pretrained_models/' + str(name) + '.pth'))
    maxAcc = test(model, name, maxAcc, test_loader, criterion, optimizer, args)

## Graphing
makeRoc(model, labels, name, test_loader, args)
plt_conf_mat(model, labels, name, test_loader, args)
