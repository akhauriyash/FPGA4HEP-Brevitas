

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
