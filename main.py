from __future__ import print_function
import time, os, matplotlib, argparse, torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from get_features import get_features
from graphing import makeRoc, plot_confusion_matrix, plt_conf_mat
from util import counter, counter_fanout, parse_config
from train import train, test

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
args = parser.parse_args()


args.cuda = True
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
global maxAcc
maxAcc = 0
print("GPU in use", torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count(), torch.cuda.get_device_name(torch.cuda.current_device()))
kwargs = {'num_workers': 4, 'pin_memory': False} if args.cuda else {}

name = 'FPGA4HEPmodel'

yamlConfig = parse_config('./yaml_IP_OP_config.yml')
print("Reading dataset...")
X_train_val, X_test, y_train_val, y_test, labels, train_loader, test_loader, input_shape, output_shape  = get_features(yamlConfig, args.batch_size, args.test_batch_size)

dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

model = Net()
if args.cuda:
    model.cuda()
print(model)

## Optimizers
optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=0.0001)
## Scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(args.epochs*x/100) for x in [10, 20, 30, 40, 60, 80]])
## Criterion
criterion = nn.CrossEntropyLoss().cuda()

## Train Loop
for epoch in range(0, args.epochs):
    scheduler.step()
    train(epoch)
    test()

makeRoc(model, labels, name, test_loader, args.test, model.outwidth, model.outmaxval, args.folder)
plt_conf_mat(model, labels, name, test_loader, args.test, model.outwidth, model.outmaxval, args.folder)
