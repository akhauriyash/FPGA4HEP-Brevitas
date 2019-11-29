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
import itertools, matplotlib, torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable


seed = 42
np.random.seed(seed)

	
def makeRoc(net, labels, name, test_loader, args):
    net.eval()
    preds = []
    truth = []
    zum = 0
    for data, target in test_loader:
        zum += 1
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
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
    plt.savefig('./' + name + 'ROC.pdf')
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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plt_conf_mat(net, labels, name, test_loader, args):
    net.eval()
    preds = []
    truth = []
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
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
    # plt.ylim(4.5, -0.5)
    plot_confusion_matrix(cnf_matrix, classes=[l.replace('j_','') for l in labels],
                              title='Confusion matrix')
    plt.figtext(0.28, 0.90,'hls4ml', wrap=True, horizontalalignment='right', fontsize=4)
    plt.savefig('./' +  str(name) + "confusion_matrix.pdf")
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[l.replace('j_','') for l in labels], normalize=True,
                              title='Normalized confusion matrix')
    plt.figtext(0.28, 0.90, name, wrap=True, horizontalalignment='right', fontsize=4)
    plt.savefig('./' + str(name) + "confusion_matrix_norm.pdf")