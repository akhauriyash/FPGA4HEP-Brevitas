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

import torch
from torch.autograd import Variable

def train(epoch, model, train_loader, criterion, optimizer, args):
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
        train_string = "Epoch - %d \t Loss - %.2f "
        train_loss_info = train_string % (epoch, loss.data.item())
        if batch_idx % args.log_interval == 0:
                print(train_loss_info)
                for g in optimizer.param_groups:
                        print("LR: {:.7f} ".format(g['lr']))

def test(model, name, maxAcc, test_loader, criterion, optimizer, args):
    model.eval()
    test_loss, correct = 0, 0 
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda() 
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += criterion(output, torch.max(target.squeeze(), 1)[1])
        pred = torch.squeeze(output.data.max(1, keepdim=True)[1])
        correct += pred.eq(torch.max(target.squeeze(), 1)[1]).long().sum()
    test_loss /= args.test_batch_size*len(test_loader)
    step_acc = 100*float(correct) / float(args.test_batch_size*len(test_loader))
    test_string = "Loss - %.3f \t Accuracy - %.2f "
    loss_info = test_string % (test_loss.item(), step_acc)
    print(loss_info)
    if(maxAcc<step_acc and args.test==False):
        print(".... Saving Model .....")
        torch.save(model.state_dict(), name + ".pth")
        maxAcc = step_acc
    return maxAcc
