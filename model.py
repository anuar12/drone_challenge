from __future__ import print_function
import numpy as np
import torch
import torchvision as tv
import os
import sys
import time
import pprint
from imblearn.metrics import classification_report_imbalanced

args = {'last_hidden_units': 256,
        'batch_size': 32,
        'lr': 0.1,
        'workers': 1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'start_epoch': 0,
        'epochs': 200,
        'print_freq': 10}
pprint.pprint(args)

class CustomPreTrained(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(CustomPreTrained, self).__init__()
        self.pretrained_model = pretrained_model
        self.last_layer = torch.nn.Linear(1024, args['last_hidden_units'])
        self.pretrained_model.classifier = self.last_layer
        self.relu = torch.nn.ReLU()
        self.layer_w = torch.nn.Linear(args['last_hidden_units'], 4)
        self.layer_s = torch.nn.Linear(args['last_hidden_units'], 5)
        self.layer_d = torch.nn.Linear(args['last_hidden_units'], 3)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.relu(x)
        x_w = self.layer_w(x)
        x_s = self.layer_s(x)
        x_d = self.layer_d(x)
        x_w = self.softmax(x_w)
        x_s = self.softmax(x_s)
        x_d = self.softmax(x_d)
        return x_w, x_s, x_d

class CustomPreTrained2(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(CustomPreTrained2, self).__init__()
        self.features = pretrained_model.features
        self.relu = torch.nn.ReLU()
        self.avg_pool = torch.nn.avg_pool2d(7, 1)

        self.layer_w_dense1 = torch.nn.Conv2d(1024, args['last_hidden_units'], 7)
        self.layer_s_dense1 = torch.nn.Conv2d(1024, args['last_hidden_units'], 7)
        self.layer_d_dense1 = torch.nn.Conv2d(1024, args['last_hidden_units'], 7)
        self.layer_w_dense = torch.nn.Linear(args['last_hidden_units'], 4)
        self.layer_s_dense = torch.nn.Linear(args['last_hidden_units'], 5)
        self.layer_d_dense = torch.nn.Linear(args['last_hidden_units'], 3)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        features = self.features(x)
        x = self.relu(features)

        x_w_dense1 = self.relu(self.layer_w_dense1(x)).view(features.size(0), -1)
        x_s_dense1 = self.relu(self.layer_s_dense1(x)).view(features.size(0), -1)
        x_d_dense1 = self.relu(self.layer_d_dense1(x)).view(features.size(0), -1)

        x_w = self.layer_w_dense(x_w_dense1)
        x_s = self.layer_s_dense(x_s_dense1)
        x_d = self.layer_d_dense(x_d_dense1)
        x_w = self.softmax(x_w)
        x_s = self.softmax(x_s)
        x_d = self.softmax(x_d)
        return x_w, x_s, x_d

model = CustomPreTrained(tv.models.densenet121(pretrained=True))
#model = CustomPreTrained2(tv.models.squeezenet1_1(pretrained=True))
model = model.cuda()
#for name, module, in model.named_parameters():
#    print(name)


def train(train_loader, model, criterion, optimizer, epoch, task):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    local_ave_loss = []
    all_losses = []
    top1 = AverageMeter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if i > 50: break
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda(async=True)
        target_var = torch.autograd.Variable(target)

        output_w, output_s, output_d = model(input_var)
        if 'weather' == task:
            output = output_w
            loss = criterion(output_w, target_var)
        elif 'setting' == task:
            output = output_s
            loss = criterion(output_s, target_var)
        elif 'daytime' == task:
            output = output_d
            loss = criterion(output_d, target_var)
        else: raise ValueError()

        prec1, prec5 = accuracy(output.data, target, topk=(1, 3))
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        local_ave_loss.append(loss.data[0])
        if i % 10 == 0 and i > 1:
            all_losses.append(sum(local_ave_loss) / 10)
            local_ave_loss = []

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args['print_freq'] == 0:
            print("Epoch %d [%d/%d], Time: %.2f, Loss: %.3f (%.3f), Accuracy: %.1f (%.1f)" %\
                     (epoch, i, len(train_loader), batch_time.val, losses.val, losses.avg, top1.val, top1.avg))
    return all_losses


def validate(val_loader, model, criterion, task, is_test=False, is_report=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    model.eval()
    end = time.time()
    targets = []
    for i, (input, target) in enumerate(val_loader):
        if i > 50: break
        #if is_report: targets += target
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda(async=True)
        target_var = torch.autograd.Variable(target)

        output_w, output_s, output_d = model(input_var)
        if 'weather' == task:
            output = output_w
            loss = criterion(output_w, target_var)
        elif 'setting' == task:
            output = output_s
            loss = criterion(output_s, target_var)
        elif 'daytime' == task:
            output = output_d
            loss = criterion(output_d, target_var)
        else: raise ValueError()

        prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top2.update(prec2[0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        print_freq = 2*args['print_freq'] if not is_test else 1
        if i % print_freq == 0:
            print('Val: %d / %d; Time: %.2fs; Loss: %.2f (%.2f); Acc-1: %.1f (%.1f), Acc-2: %.1f (%.1f)' %\
                    (i, len(val_loader), batch_time.val, losses.val, losses.avg, top1.val, top1.avg, top2.val, top2.avg))
        if is_test: print(target_var.data[0], output.data[0])
    print("Overall average accuracy on validation: %.1f, %.1f" % (top1.avg, top2.avg))

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
    lr = args['lr'] * (0.2 ** (epoch // 100))
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
    weight_per_class = [min(x, 15.) for x in weight_per_class]  # clip the largest value
    print("Weight per class: ", weight_per_class)
    #weight_per_class = [2., 20., 10., 2.]
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight   


normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def get_data_loader(data_dir, is_balanced=False):
    
    dataset_folder = tv.datasets.ImageFolder(data_dir, tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            normalize]))
    if 'train' in data_dir or is_balanced:
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

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    #if is_best:
    #    shutil.copyfile(filename, 'model_best.pth.tar')

current_dir = os.getcwd()
data_path_w = current_dir + '/WeatherImageDataset'
data_path_s = current_dir + '/SettingDataset'
data_path_d = current_dir + '/DaytimeDataset'
train_dir_w = data_path_w + '/train'
val_dir_w = data_path_w + '/val'
train_dir_s = data_path_s + '/train'
val_dir_s = data_path_s + '/val'
train_dir_d = data_path_d + '/train'
val_dir_d = data_path_d + '/val'
test_dir_w = data_path_w + '/test'
test_dir_s = data_path_s + '/test'
test_dir_d = data_path_d + '/test'

train_loader_w = get_data_loader(train_dir_w)
val_loader_w = get_data_loader(val_dir_w)
train_loader_s = get_data_loader(train_dir_s)
val_loader_s = get_data_loader(val_dir_s)
train_loader_d = get_data_loader(train_dir_d)
val_loader_d = get_data_loader(val_dir_d)
val_loader_w_bal = get_data_loader(val_dir_w, is_balanced=True)
val_loader_s_bal = get_data_loader(val_dir_s, is_balanced=True)
val_loader_d_bal = get_data_loader(val_dir_d, is_balanced=True)


test_loader_w = torch.utils.data.DataLoader(
    tv.datasets.ImageFolder(test_dir_w, tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        normalize])))
test_loader_s = torch.utils.data.DataLoader(
    tv.datasets.ImageFolder(test_dir_s, tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        normalize])))
test_loader_d = torch.utils.data.DataLoader(
    tv.datasets.ImageFolder(test_dir_d, tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        normalize])))

criterion = torch.nn.CrossEntropyLoss().cuda()

#ignored_params = list(map(id, model.layer_w_dense1.parameters()))
#ignored_params += list(map(id, model.layer_s_dense1.parameters()))
#ignored_params += list(map(id, model.layer_d_dense1.parameters()))
#ignored_params += list(map(id, model.layer_w_dense.parameters()))
#ignored_params += list(map(id, model.layer_s_dense.parameters()))
#ignored_params += list(map(id, model.layer_d_dense.parameters()))

ignored_params = list(map(id, model.last_layer.parameters()))
ignored_params += list(map(id, model.layer_s.parameters()))
ignored_params += list(map(id, model.layer_d.parameters()))
ignored_params += list(map(id, model.layer_w.parameters()))
print(ignored_params)
base_params = filter(lambda p: id(p) not in ignored_params,
                     model.parameters())
addit_params = filter(lambda p: id(p) in ignored_params,
                     model.parameters())
# TODO
optimizer = torch.optim.SGD([
            {'params': base_params},
	    {'params': model.last_layer.parameters(), 'lr': args['lr'],\
                    'momentum': args['momentum'], 'weight_decay': args['weight_decay']},
	    {'params': model.layer_w.parameters(), 'lr': args['lr'],\
                    'momentum': args['momentum'], 'weight_decay': args['weight_decay']},
	    {'params': model.layer_s.parameters(), 'lr': args['lr'],\
                    'momentum': args['momentum'], 'weight_decay': args['weight_decay']},
            {'params': model.layer_d.parameters(), 'lr': args['lr'],\
                    'momentum': args['momentum'], 'weight_decay': args['weight_decay']}],
			    lr=args['lr']*0.2,
                            momentum=args['momentum'],
                            weight_decay=args['weight_decay'])

#optimizer = torch.optim.SGD([
#            {'params': base_params},
#	    {'params': model.layer_w_dense1.parameters(), 'lr': args['lr'],\
#                    'momentum': args['momentum'], 'weight_decay': args['weight_decay']},
#	    {'params': model.layer_s_dense1.parameters(), 'lr': args['lr'],\
#                    'momentum': args['momentum'], 'weight_decay': args['weight_decay']},
#	    {'params': model.layer_d_dense1.parameters(), 'lr': args['lr'],\
#                    'momentum': args['momentum'], 'weight_decay': args['weight_decay']},
#	    {'params': model.layer_w_dense.parameters(), 'lr': args['lr'],\
#                    'momentum': args['momentum'], 'weight_decay': args['weight_decay']},
#	    {'params': model.layer_s_dense.parameters(), 'lr': args['lr'],\
#                    'momentum': args['momentum'], 'weight_decay': args['weight_decay']},
#            {'params': model.layer_d_dense.parameters(), 'lr': args['lr'],\
#                    'momentum': args['momentum'], 'weight_decay': args['weight_decay']}],
#			    lr=args['lr']*0.2,
#                            momentum=args['momentum'],
#                            weight_decay=args['weight_decay'])

# if args.evaluate:
#     validate(val_loader, model, criterion)
list_of_tasks = ['weather', 'setting', 'daytime']
best_prec1 = 0
all_losses_w = []
all_losses_s = []
all_losses_d = []
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
                if 'layer_d' in name:
                    param.requires_grad = False
        elif task == 'setting':
            print("\t\t========== Training For Setting ==========")
            train_loader = train_loader_s
            val_loader = val_loader_s
            for name, param in model.named_parameters():
                if 'layer_s' in name:
                    param.requires_grad = True
                if 'layer_w' in name:
                    param.requires_grad = False
                if 'layer_d' in name:
                    param.requires_grad = False
        elif task == 'daytime':
            print("\t\t========== Training For Daytime ==========")
            train_loader = train_loader_d
            val_loader = val_loader_d
            for name, param in model.named_parameters():
                if 'layer_d' in name:
                    param.requires_grad = True
                if 'layer_s' in name:
                    param.requires_grad = False
                if 'layer_w' in name:
                    param.requires_grad = False
        else: raise ValueError()

        epoch_losses = train(train_loader, model, criterion, optimizer, epoch, task)
        if task == 'weather':
            all_losses_w.append(epoch_losses)
        elif task == 'setting':
            all_losses_s.append(epoch_losses)
        elif task == 'daytime':
            all_losses_d.append(epoch_losses)

    if epoch % 7 == 0:
        prec1 = validate(val_loader_w, model, criterion, 'weather')
        prec1 = validate(val_loader_s, model, criterion, 'setting')
        prec1 = validate(val_loader_d, model, criterion, 'daytime')

np.save('losses_w', all_losses_w)
np.save('losses_s', all_losses_s)
np.save('losses_d', all_losses_d)

print("\n\tFinal Validation...")
prec1 = validate(val_loader_w, model, criterion, 'weather', is_report=True)
prec1 = validate(val_loader_s, model, criterion, 'setting', is_report=True)
prec1 = validate(val_loader_d, model, criterion, 'daytime', is_report=True)
print("\n\tFinal Balanced Validation...")
prec1 = validate(val_loader_w_bal, model, criterion, 'weather', is_report=True)
prec1 = validate(val_loader_s_bal, model, criterion, 'setting', is_report=True)
prec1 = validate(val_loader_d_bal, model, criterion, 'daytime', is_report=True)
print("\n\t\t========== Testing For Weather ==========")
prec_test = validate(test_loader_w, model, criterion, 'weather', is_test=True)
print("\t\t========== Testing For Setting ==========")
prec_test = validate(test_loader_s, model, criterion, 'setting', is_test=True)
print("\t\t========== Testing For Daytime ==========")
prec_test = validate(test_loader_d, model, criterion, 'setting', is_test=True)

save_checkpoint({
    'epoch': args['epochs'] + 1,
    'arch': 'densenet_conv',
    'state_dict': model.state_dict(),
    'best_prec1': prec1})

