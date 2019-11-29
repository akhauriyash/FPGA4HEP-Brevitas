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

from functools import reduce
from operator import mul
from torch.nn import Module, ModuleList, BatchNorm1d, Dropout
from brevitas.core.quant import QuantType
from common import make_quant_linear, make_quant_hard_tanh, make_quant_relu, make_activation
import torch


FC_OUT_FEATURES = [64, 32, 32]
INTERMEDIATE_FC_PER_OUT_CH_SCALING = True
LAST_FC_PER_OUT_CH_SCALING = False
IN_DROPOUT = 0.0
HIDDEN_DROPOUT = 0.0

class LFC(Module):
    def __init__(self, num_classes=5, weight_bit_width=None, act_bit_width=None,
                        in_bit_width=None):
        super(LFC, self).__init__()
        self.num_classes      = num_classes
        self.weight_bit_width = weight_bit_width
        self.act_bit_width    = act_bit_width
        self.in_bit_width     = in_bit_width
        self.qtype   = QuantType.BINARY if weight_bit_width==1 else QuantType.INT
        self.metric  = 0 # Used for learning rate policy 'plateau'
        self.act0    = make_activation(act_bit_width, 'hardtanh')
        self.linear1 = make_quant_linear(16, FC_OUT_FEATURES[0], 
                                        weight_quant_type=self.qtype,
                                         bias=True, 
                                         bit_width=weight_bit_width)
        self.bn1     = BatchNorm1d(FC_OUT_FEATURES[0])
        self.act1    = make_activation(act_bit_width, 'relu')

        self.linear2 = make_quant_linear(FC_OUT_FEATURES[0],
                                         FC_OUT_FEATURES[1], 
                                        weight_quant_type=self.qtype,
                                         bias=True, 
                                         bit_width=weight_bit_width)
        self.bn2     = BatchNorm1d(FC_OUT_FEATURES[1])
        self.act2    = make_activation(act_bit_width, 'relu')
        
        self.linear3 = make_quant_linear(FC_OUT_FEATURES[1],
                                         FC_OUT_FEATURES[2],
                                         weight_quant_type=self.qtype,
                                         bias=True, 
                                         bit_width=weight_bit_width)
        self.bn3     = BatchNorm1d(FC_OUT_FEATURES[2])
        self.act3    = make_activation(act_bit_width, 'relu')
        
        self.linear4 = make_quant_linear(FC_OUT_FEATURES[2], 
                                         num_classes, 
                                         weight_quant_type=self.qtype,
                                         bias=True, 
                                         bit_width=weight_bit_width)
        self.bn4     = BatchNorm1d(num_classes)
        self.act4    = torch.nn.LogSoftmax(dim=1)

        
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
