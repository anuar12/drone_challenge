from __future__ import print_function
import numpy as np
import torch
import torchvision as tv
import os
import time

args = {'last_hidden_units': 256,
        'batch_size': 64,
        'lr': 0.1,
        'workers': 1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'start_epoch': 0,
        'epochs': 3,
        'print_freq': 10}

class CustomDenseNet(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(CustomDenseNet, self).__init__()
        self.pretrained_model = pretrained_model
        self.last_layer = torch.nn.Linear(1024, args['last_hidden_units'])
        self.pretrained_model.classifier = self.last_layer
        self.relu = torch.nn.ReLU()
        self.layer_w = torch.nn.Linear(args['last_hidden_units'], 4)
        self.layer_s = torch.nn.Linear(args['last_hidden_units'], 5)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.relu(x)
        x_w = self.layer_w(x)
        x_s = self.layer_s(x)
        x_w = self.softmax(x_w)
        x_s = self.softmax(x_s)
        return x_w, x_s

model = CustomDenseNet(tv.models.densenet121(pretrained=True))
model = model.cuda()
#for name, module, in model.named_parameters():
#    print(name)


def train(train_loader, model, criterion, optimizer, epoch, task):
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

        output_w, output_s = model(input_var)
        if 'weather' == task:
            output = output_w
            loss = criterion(output_w, target_var)
        elif 'setting' == task:
            output = output_s
            loss = criterion(output_s, target_var)
        else: raise ValueError()

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


def validate(val_loader, model, criterion, task, is_test=False):
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

        output_w, output_s = model(input_var)
        if 'weather' == task:
            output = output_w
            loss = criterion(output_w, target_var)
        elif 'setting' == task:
            output = output_s
            loss = criterion(output_s, target_var)
        else: raise ValueError()

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
    print("Overall average accuracy on validation: ", top1.avg)

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
    #weight_per_class = [2., 20., 10., 2.]
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight   


normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def get_data_loader(data_dir):
    
    dataset_folder = tv.datasets.ImageFolder(data_dir, tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            normalize]))
    if 'train' in data_dir:
        weights = make_weights_for_balanced_classes(dataset_folder.imgs,\
                                                    len(dataset_folder.classes)) 
        weights = torch.DoubleTensor(weights) 
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        is_shuffle = False
    else:
        sampler = None
        is_shuffle = True

    dataset_loader = torch.utils.data.DataLoader(
        dataset_folder, sampler=sampler, 
        batch_size=args['batch_size'], shuffle=is_shuffle,
        num_workers=args['workers'], pin_memory=True)

    return dataset_loader

current_dir = os.getcwd()
data_path_w = current_dir + '/WeatherImageDataset'
data_path_s = current_dir + '/imagenet'
train_dir_w = data_path_w + '/train'
val_dir_w = data_path_w + '/val'
train_dir_s = data_path_s + '/train'
val_dir_s = data_path_s + '/val'
test_dir = data_path_w + '/test'

train_loader_w = get_data_loader(train_dir_w)
val_loader_w = get_data_loader(val_dir_w)
train_loader_s = get_data_loader(train_dir_s)
val_loader_s = get_data_loader(val_dir_s)


test_loader = torch.utils.data.DataLoader(
    tv.datasets.ImageFolder(test_dir, tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        normalize])))

criterion = torch.nn.CrossEntropyLoss().cuda()

ignored_params = list(map(id, model.last_layer.parameters()))
ignored_params += list(map(id, model.layer_w.parameters()))
ignored_params += list(map(id, model.layer_s.parameters()))
print(ignored_params)
base_params = filter(lambda p: id(p) not in ignored_params,
                     model.parameters())

optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': model.last_layer.parameters(), 'lr': args['lr']}],\
			    lr=args['lr']*0.25,
                            momentum=args['momentum'],
                            weight_decay=args['weight_decay'])

# if args.evaluate:
#     validate(val_loader, model, criterion)
list_of_tasks = ['weather', 'setting']
best_prec1 = 0
for epoch in range(args['start_epoch'], args['epochs']):
    print("\n\t\t==========  NEW EPOCH  =========")
    adjust_learning_rate(optimizer, epoch)
    for task in list_of_tasks:
        if task == 'weather':
            print("\t\t========== Training For Weather ==========")
            train_loader = train_loader_w
            val_loader = val_loader_w
            for name, param in model.named_parameters():
                if 'layer_w' in name:
                    param.requires_grad = True
                if 'layer_s' in name:
                    param.requires_grad = False
        if task == 'setting':
            print("\t\t========== Training For Setting ==========")
            train_loader = train_loader_s
            val_loader = val_loader_s
            for name, param in model.named_parameters():
                if 'layer_s' in name:
                    param.requires_grad = True
                if 'layer_w' in name:
                    param.requires_grad = False

        train(train_loader, model, criterion, optimizer, epoch, task)
        prec1 = validate(val_loader, model, criterion, task)

        is_best = prec1 > best_prec1
        #best_prec1 = max(prec1, best_prec1)

prec_test = validate(test_loader, model, criterion, 'weather', is_test=True)
prec_test = validate(test_loader, model, criterion, 'setting', is_test=True)

