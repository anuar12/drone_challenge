from __future__ import print_function
import numpy as np
import torch
import torchvision as tv
import os
import time

args = {'num_classes': 4,
        'batch_size': 64,
        'lr': 0.1,
        'workers': 1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'start_epoch': 0,
        'epochs': 6,
        'print_freq': 10}

#class CustomDenseNet(torch.nn.Module):
#    def __init__(self):
#        super(CustomDenseNet, self).__init__()
#        densenet = tv.models.densenet121(pretrained=True)
#        densenet.classifier = torch.nn.Linear(1024, args['num_classes'])
#        self.densenet = densenet
#        self.softmax = torch.nn.Softmax(dim=1)
#    
#    def forward(self, x):
#        x = self.densenet(x)
#        x = self.softmax(x)
#        return x
    
class CustomDenseNet(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(CustomDenseNet, self).__init__()
        self.pretrained_model = pretrained_model
        self.last_layer = torch.nn.Linear(1024, args['num_classes'])
        self.pretrained_model.classifier = self.last_layer
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.softmax(x)
        return x

model = CustomDenseNet(tv.models.densenet121(pretrained=True))
model = model.cuda()
#for name, module, in model.named_parameters():
#    print(name)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda(async=True)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 3))
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args['print_freq'] == 0:
            print("Epoch %d [%d/%d], Time: %.2f, Loss: %.3f (%.3f), Accuracy: %.1f (%.1f)" %\
                     (epoch, i, len(train_loader), batch_time.val, losses.val, losses.avg, top1.val, top1.avg))


def validate(val_loader, model, criterion, is_test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda(async=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1 = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        print_freq = 2*args['print_freq'] if not is_test else 1
        if i % print_freq == 0:
            print('Val: %d / %d; Time: %.2fs; Loss: %.2f (%.2f); Accuracy: %.1f (%.1f)' %\
                    (i, len(val_loader), batch_time.val, losses.val, losses.avg, top1.val, top1.avg))
        if is_test: print(target_var.data, output.data)

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args['lr'] * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    print("Weight per class: ", weight_per_class)
    weight_per_class = [2., 20., 10., 2.]
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight   


current_dir = os.getcwd()
data_path = current_dir + '/WeatherImageDataset'
train_dir = data_path + '/train'
val_dir = data_path + '/val'
test_dir = data_path + '/test'
normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
dataset_train = tv.datasets.ImageFolder(train_dir, tv.transforms.Compose([
        tv.transforms.RandomSizedCrop(224),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        normalize]))
weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))    
weights = torch.DoubleTensor(weights) 
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))   

train_loader = torch.utils.data.DataLoader(
    dataset_train, #sampler=sampler, 
    batch_size=args['batch_size'], shuffle=True,
    num_workers=args['workers'], pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    tv.datasets.ImageFolder(val_dir, tv.transforms.Compose([
        tv.transforms.Scale(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args['batch_size'], shuffle=False,
    num_workers=args['workers'], pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    tv.datasets.ImageFolder(test_dir, tv.transforms.Compose([
        tv.transforms.Scale(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(), normalize])))

criterion = torch.nn.CrossEntropyLoss().cuda()

ignored_params = list(map(id, model.last_layer.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                     model.parameters())

optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': model.last_layer.parameters(), 'lr': args['lr']}],\
			    lr=args['lr']*0.25,
                            momentum=args['momentum'],
                            weight_decay=args['weight_decay'])
#optimizer = torch.optim.SGD(model.parameters(), args['lr'],
#                            momentum=args['momentum'],
#                            weight_decay=args['weight_decay'])

# if args.evaluate:
#     validate(val_loader, model, criterion)
best_prec1 = 0
for epoch in range(args['start_epoch'], args['epochs']):
    print("\n\t\t==========  NEW EPOCH  =========")
    adjust_learning_rate(optimizer, epoch)

    train(train_loader, model, criterion, optimizer, epoch)
    prec1 = validate(val_loader, model, criterion)

    is_best = prec1 > best_prec1
    #best_prec1 = max(prec1, best_prec1)

prec_test = validate(test_loader, model, criterion, is_test=True)

