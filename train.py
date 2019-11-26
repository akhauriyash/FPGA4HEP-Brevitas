
from __future__ import print_function
import time, sys, os, itertools, h5py, yaml, matplotlib, argparse, torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
from functools import reduce
import numpy as np
import pandas as pd
from optparse import OptionParser
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
matplotlib.use('agg')
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple
from brevitas.nn import QuantHardTanh
import torch
from get_features import get_features
from graphing import makeRoc, plot_confusion_matrix, plt_conf_mat
from fpganet.linear import SparseLinear, DenseQuantLinear
from fpganet.convolution import DepthwiseKernelSparseConv, PointwiseConv, SparseConv


parser = argparse.ArgumentParser(description='FPGANet FPGA4HEP Example')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--maxVal', type=float, default=1.61,
                    help='maxVal')
parser.add_argument('--SGD', action='store_true', default=False, 
            help='Use SGD?')
parser.add_argument('--no-softmax', action='store_false', default=True,
                    help='add softmax at end')
parser.add_argument('--fname', type=str, default='not_assigned',help='filename to save results')
parser.add_argument('--folder', type=str, default='not_assigned',help='folder to save results to')
parser.add_argument('--test', action='store_true', default=False,
                    help='Only import model and test it (Graph ROC etc).')
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

def parse_config(config_file) :
    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config)

yamlConfig = parse_config('./yaml_IP_OP_config.yml')
print("Reading dataset...")
X_train_val, X_test, y_train_val, y_test, labels, train_loader, test_loader, input_shape, output_shape  = get_features(yamlConfig, args.batch_size, args.test_batch_size)

dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

def counter(vector):
    s = vector.shape[0]
    k = []
    v = vector.detach()
    v[v!=0] = 1
    for i in range(s):
        k.append(torch.sum(v[i, :]).item())
    return k

def counter_fanout(vector):
    s = vector.shape[1]
    k = []
    v = vector.detach()
    v[v!=0] = 1
    for i in range(s):
        k.append(torch.sum(v[:, i]).item())
    return k


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.metric  = 0 # Used for learning rate policy 'plateau'
        self.test    = False
        self.maxVal  = 1.61
        self.outmaxval = 3
        self.outwidth= 4
        self.BW      = 1
        self.HL      = [16, 32, 32, 32, 5]
        self.X       = [6, 6, 6, 6]
        self.outquant= QuantHardTanh(bit_width=self.outwidth, min_val=-1*self.outmaxval, 
                                     max_val=self.outmaxval)
        self.linear4 = SparseLinear(self.HL[3], self.HL[4], expandSize=self.X[3], inBW=self.BW,
                                    maxVal=self.maxVal, next_module=self.outquant)
        self.linear3 = SparseLinear(self.HL[2], self.HL[3], expandSize=self.X[2], inBW=self.BW,
                                    maxVal=self.maxVal, next_module=self.linear4)
        self.linear2 = SparseLinear(self.HL[1], self.HL[2], expandSize=self.X[1], inBW=self.BW,
                                    maxVal=self.maxVal, next_module=self.linear3)
        self.linear1 = SparseLinear(self.HL[0], self.HL[1], expandSize=self.X[0], inBW=self.BW,
                                    maxVal=self.maxVal, next_module=self.linear2)
        self.softmx  = nn.LogSoftmax()
    def forward(self, x):
        x = x.view(-1, 16)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.softmx(self.outquant(x))
        return x

model = Net()
if args.cuda:
    model.cuda()
print(model)

## Optimizers
if(args.SGD==True):
    optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=0.0001)
else:
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)


## Schedulers
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(args.epochs*x/100) for x in [10, 20, 30, 40, 60, 80]])
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, cooldown=2,patience=10, min_lr=0.0000001, eps=0.000001)
# scheduler = CyclicCosAnnealingLR(optimizer, milestones=[6, 15, 36, 48, 72, 108, 144, 192, 240, 288], decay_milestones=[36, 72, 144, 288, 576], eta_min=1e-7)

## Criterion
criterion = nn.CrossEntropyLoss().cuda()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, torch.max(target.squeeze(), 1)[1])
        if args.cuda:
            loss = loss.cuda()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
                print('Epoch: ' + str(epoch) + "\t%: " + str(int(100.*batch_idx/len(train_loader))) + "\tLoss: " + str(loss.data.item()))
                for g in optimizer.param_groups:
                        print("LR: {:.6f} ".format(g['lr']))

def test():
    model.eval()
    # final1 = QuantHardTanh(bit_width=model.outwidth, min_val=-1*model.outmaxval, 
    #                                  max_val=model.outmaxval).cuda()
    # final2 = nn.LogSoftmax().cuda()
    global maxAcc
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda() 
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # output = final2(final1(output))
        test_loss += criterion(output, torch.max(target.squeeze(), 1)[1])
        pred = torch.squeeze(output.data.max(1, keepdim=True)[1])
        correct += pred.eq(torch.max(target.squeeze(), 1)[1]).long().sum()
    test_loss /= args.test_batch_size*len(test_loader)
    print("Loss: " + str(test_loss) + "\t%: " + str(100*float(correct) / float(args.test_batch_size*len(test_loader))))
    if(maxAcc<float(100*float(float(correct)/float(args.test_batch_size*len(test_loader))))):
        torch.save(model.state_dict(), name + ".pth")
        maxAcc = float(100*float(float(correct)/float(args.test_batch_size*len(test_loader))))

for epoch in range(0, args.epochs):
    scheduler.step()
    train(epoch)
    test()

makeRoc(model, labels, name, test_loader, args.test, model.outwidth, model.outmaxval, args.folder)
plt_conf_mat(model, labels, name, test_loader, args.test, model.outwidth, model.outmaxval, args.folder)


# saved_model_import = torch.load("./saved_models_corrected/" + name + ".pth")
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in saved_model_import.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)