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
