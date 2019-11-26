from functools import reduce
from operator import mul
from torch.nn import Module, ModuleList, BatchNorm1d, Dropout
from common import make_quant_linear, make_quant_hard_tanh, make_quant_relu
import torch


FC_OUT_FEATURES = [32, 64, 64]
INTERMEDIATE_FC_PER_OUT_CH_SCALING = True
LAST_FC_PER_OUT_CH_SCALING = False
IN_DROPOUT = 0.0
HIDDEN_DROPOUT = 0.0

class LFC(Module):
    def __init__(self, num_classes=5, weight_bit_width=None, act_bit_width=None,
                        in_bit_width=None):
        super(LFC, self).__init__()
        self.metric  = 0 # Used for learning rate policy 'plateau'
        self.act0    = make_quant_hard_tanh(act_bit_width)
        self.linear1 = make_quant_linear(16, FC_OUT_FEATURES[0], 
                                         bias=True, 
                                         bit_width=weight_bit_width)
        self.bn1     = BatchNorm1d(FC_OUT_FEATURES[0])
        self.act1    = make_quant_relu(act_bit_width)
        
        self.linear2 = make_quant_linear(FC_OUT_FEATURES[0],
                                         FC_OUT_FEATURES[1], 
                                         bias=True, 
                                         bit_width=weight_bit_width)
        self.bn2     = BatchNorm1d(FC_OUT_FEATURES[1])
        self.act2    = make_quant_relu(act_bit_width)
        
        self.linear3 = make_quant_linear(FC_OUT_FEATURES[1],
                                         FC_OUT_FEATURES[2],
                                         bias=True, 
                                         bit_width=weight_bit_width)
        self.bn3     = BatchNorm1d(FC_OUT_FEATURES[2])
        self.act3    = make_quant_relu(act_bit_width)
        
        self.linear4 = make_quant_linear(FC_OUT_FEATURES[2], 
                                         num_classes, 
                                         bias=True, 
                                         bit_width=weight_bit_width)
        self.bn4     = BatchNorm1d(num_classes)
        self.act4    = torch.nn.LogSoftmax()


    def forward(self, x):
        x = x.view(-1, 16)
        x = self.act0(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.linear4(x)
        x = self.bn4(x)
        x = self.act4(x)
        return x
