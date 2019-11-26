
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
