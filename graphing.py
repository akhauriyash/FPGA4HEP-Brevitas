from __future__ import print_function
import time, sys, os, itertools, h5py, yaml, matplotlib, argparse, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce
from torchvision import datasets, transforms
from torch.autograd import Variable
from brevitas.nn.quant_linear import QuantLinear
from brevitas.nn.quant_activation import QuantReLU
from brevitas.nn.quant_activation import QuantHardTanh
from brevitas.core.quant import QuantType
import numpy as np
seed = 42
np.random.seed(seed)
from optparse import OptionParser
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
matplotlib.use('agg')
import matplotlib.pyplot as plt
from get_features import get_features
from fpganet.linear import SparseLinear, DenseQuantLinear
from fpganet.convolution import DepthwiseKernelSparseConv, PointwiseConv, SparseConv
cuda = True

	
def makeRoc(net, labels, name, test_loader, is_test, outwidth, outmaxval, folder):
    net.eval()
    preds = []
    truth = []
    zum = 0
    # final1 = QuantIdentity(bit_width=outwidth, min_val=-1*outmaxval, max_val=outmaxval).cuda()
    # final2 = nn.LogSoftmax().cuda()
    for data, target in test_loader:
        zum += 1
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        # if is_test:
        #     output = final2(final1(net(data)))
        # else:
        output = net(data)
        tar = torch.squeeze(target)
        preds.append(output)
        truth.append(tar)
    preds = np.array([x for y in preds for x in y])
    truth = np.array([x for y in truth for x in y])
    preds = np.array([x.cpu().detach().numpy() for x in preds])
    truth = np.array([x.cpu().detach().numpy() for x in truth]).astype(int)
    print('in makeRoc()')
    if 'j_index' in labels: labels.remove('j_index')
    predict_test = preds
    labels_val   = truth
    df = pd.DataFrame()
    fpr = {}
    tpr = {}
    auc1 = {}
    plt.figure()
    for i, label in enumerate(labels):
        df[label] = labels_val[:,i]
        df[label + '_pred'] = predict_test[:,i]
        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred'])
        auc1[label] = auc(fpr[label], tpr[label])
        plt.plot(tpr[label],fpr[label],label='%s tagger, AUC = %.1f%%'%(label.replace('j_',''),auc1[label]*100.))
    plt.semilogy()
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.figtext(0.25, 0.90, name, wrap=True, horizontalalignment='right', fontsize=4)
    if folder=="not_assigned":
        plt.savefig('./results/' + name + 'ROC.pdf')
    else:
        ff = str(folder)
        ff = "./" + ff + "/"
        plt.savefig(ff + name + 'ROC.pdf')
    return predict_test 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    cbar = plt.colorbar()
    plt.clim(0,1)
    cbar.set_label(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plt_conf_mat(net, labels, name, test_loader, is_test, outwidth, outmaxval, folder):
    net.eval()
    # final1 = QuantHardTanh(bit_width=net.outwidth, min_val=-1*net.outmaxval, 
    #                                  max_val=net.outmaxval).cuda()
    # final2 = nn.LogSoftmax().cuda()
    preds = []
    truth = []
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        # if is_test:
        #     output = final2(final1(net(data))).max(1)[1]
        # else:
        output = net(data).max(1)[1]
        tar = torch.squeeze(target).max(1)[1]
        preds.append(output)
        truth.append(tar)
    preds = [x for y in preds for x in y]
    truth = [x for y in truth for x in y]
    preds = [x.cpu().detach().numpy() for x in preds]
    truth = [x.cpu().detach().numpy() for x in truth]
    cnf_matrix = confusion_matrix(preds, truth)
    np.set_printoptions(precision=2)
    plt.figure()
    if folder=="not_assigned":
        plot_confusion_matrix(cnf_matrix, classes=[l.replace('j_','') for l in labels],
                                  title='Confusion matrix')
        plt.figtext(0.28, 0.90,'hls4ml', wrap=True, horizontalalignment='right', fontsize=4)
        plt.savefig('./results/' +  str(name) + "confusion_matrix.pdf")
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=[l.replace('j_','') for l in labels], normalize=True,
                                  title='Normalized confusion matrix')
        plt.figtext(0.28, 0.90, name, wrap=True, horizontalalignment='right', fontsize=4)
        plt.savefig('./results/' + str(name) + "confusion_matrix_norm.pdf")
    else:
        ff = str(folder)
        ff = "./" + ff + "/"
        plot_confusion_matrix(cnf_matrix, classes=[l.replace('j_','') for l in labels],
                                  title='Confusion matrix')
        plt.figtext(0.28, 0.90,'hls4ml', wrap=True, horizontalalignment='right', fontsize=4)
        plt.savefig(ff +  str(name) + "confusion_matrix.pdf")
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=[l.replace('j_','') for l in labels], normalize=True,
                                  title='Normalized confusion matrix')
        plt.figtext(0.28, 0.90, name, wrap=True, horizontalalignment='right', fontsize=4)
        plt.savefig(ff + str(name) + "confusion_matrix_norm.pdf")

