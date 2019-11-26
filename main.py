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

yamlConfig = parse_config('./yaml_IP_OP_config.yml')
print("Reading dataset...")
X_train_val, X_test, y_train_val, y_test, labels, train_loader, test_loader, input_shape, output_shape  = get_features(yamlConfig, args.batch_size, args.test_batch_size)

dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

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