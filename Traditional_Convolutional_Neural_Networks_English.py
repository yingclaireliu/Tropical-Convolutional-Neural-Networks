#coding=utf-8
import numpy as np
import torch
from torch import optim, nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchsummary import summary
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from optparse import OptionParser
from matplotlib import pyplot
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm, trange
import time
import sys
import os
import math
import gzip
# use cpu or gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# writer = SummaryWriter() # define writer, use default path


'''
Classes of various custom layers
-6 tropical convolution layers
'''

# MinPlus-Sum_Conv Layer (MinP-S)
class tropical_min_sum_conv(nn.Module):
# minplus conv layer
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(tropical_min_sum_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        # The shape of the convolution kernel is (the number of convolution kernels (also the number of output channels), 
#         the number of input image channels, the length of the convolution kernel, and the width of the convolution kernel) 
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size))
        #self.bias = nn.Parameter(torch.rand(out_channels))   

    def forward(self, x):
        #  The shape of input 'x' is（the number of input image，channels，length，width）
        n_filters, d_filter, h_filter, w_filter = self.weight.size()
        n_x, d_x, h_x, w_x = x.size()
        h_out = (h_x - h_filter + 2 * self.padding) / self.stride + 1
        w_out = (w_x - w_filter + 2 * self.padding) / self.stride + 1
        # the output shape is（the number of input image，output channels，length，width）
        # result = torch.zeros(x.size(0), self.out_channels, out_features, out_features)
        h_out, w_out = int(h_out), int(w_out)
        X_col = torch.nn.functional.unfold(x.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=self.padding, stride=self.stride).view(n_x, d_filter, -1, h_out*w_out)
        X_col = X_col.permute(1,2,3,0).contiguous().view(X_col.size(1),X_col.size(2),-1)
        W_col = self.weight.view(n_filters, d_filter, -1)
        result = ((W_col.unsqueeze(3)+X_col.unsqueeze(0)).min(2).values).sum(1)
        result = (result.view(n_filters, h_out, w_out, n_x)).permute(3, 0, 1, 2).contiguous()

        return result

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding
        )

# MaxPlus-Sum-Conv Layer (MaxP-S)
class tropical_max_sum_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(tropical_max_sum_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        # The shape of the convolution kernel is (the number of convolution kernels (also the number of output channels), 
#         the number of input image channels, the length of the convolution kernel, and the width of the convolution kernel) 
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size))
        #self.bias = nn.Parameter(torch.rand(out_channels))   

    def forward(self, x):
        #  The shape of input 'x' is（the number of input image，channels，length，width）
        n_filters, d_filter, h_filter, w_filter = self.weight.size()
        n_x, d_x, h_x, w_x = x.size()
        h_out = (h_x - h_filter + 2 * self.padding) / self.stride + 1
        w_out = (w_x - w_filter + 2 * self.padding) / self.stride + 1
        # the output shape is（the number of input image，output channels，length，width）
        # result = torch.zeros(x.size(0), self.out_channels, out_features, out_features)
        h_out, w_out = int(h_out), int(w_out)
        X_col = torch.nn.functional.unfold(x.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=self.padding, stride=self.stride).view(n_x, d_filter, -1, h_out*w_out)
        X_col = X_col.permute(1,2,3,0).contiguous().view(X_col.size(1),X_col.size(2),-1)
        W_col = self.weight.view(n_filters, d_filter, -1)
        result = ((W_col.unsqueeze(3)+X_col.unsqueeze(0)).max(2).values).sum(1)
        result = (result.view(n_filters, h_out, w_out, n_x)).permute(3, 0, 1, 2).contiguous()

        return result

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding
        )

# MinPlus-Max-Conv Layer (MinP-Max)
class tropical_min_max_conv(nn.Module):
# +\min\max
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(tropical_min_max_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        # The shape of the convolution kernel is (the number of convolution kernels (also the number of output channels), 
#         the number of input image channels, the length of the convolution kernel, and the width of the convolution kernel) 
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size))
        #self.bias = nn.Parameter(torch.rand(out_channels))   

    def forward(self, x):
        #  The shape of input 'x' is（the number of input image，channels，length，width）
        n_filters, d_filter, h_filter, w_filter = self.weight.size()
        n_x, d_x, h_x, w_x = x.size()
        h_out = (h_x - h_filter + 2 * self.padding) / self.stride + 1
        w_out = (w_x - w_filter + 2 * self.padding) / self.stride + 1
        # the output shape is（the number of input image，output channels，length，width）
        # result = torch.zeros(x.size(0), self.out_channels, out_features, out_features)
        h_out, w_out = int(h_out), int(w_out)
        X_col = torch.nn.functional.unfold(x.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=self.padding, stride=self.stride).view(n_x, d_filter, -1, h_out*w_out)
        X_col = X_col.permute(1,2,3,0).contiguous().view(X_col.size(1),X_col.size(2),-1)
        W_col = self.weight.view(n_filters, d_filter, -1)
        result = ((W_col.unsqueeze(3)+X_col.unsqueeze(0)).min(2).values).max(1).values
        result = (result.view(n_filters, h_out, w_out, n_x)).permute(3, 0, 1, 2).contiguous()

        return result

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding
        )

# MaxPlus-Min-Conv Layer (MaxP-Min)
class tropical_max_min_conv(nn.Module):
# +\max\min
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(tropical_max_min_conv, self).__init__()
       self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        # The shape of the convolution kernel is (the number of convolution kernels (also the number of output channels), 
#         the number of input image channels, the length of the convolution kernel, and the width of the convolution kernel) 
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size))
        #self.bias = nn.Parameter(torch.rand(out_channels))   

    def forward(self, x):
        #  The shape of input 'x' is（the number of input image，channels，length，width）
        n_filters, d_filter, h_filter, w_filter = self.weight.size()
        n_x, d_x, h_x, w_x = x.size()
        h_out = (h_x - h_filter + 2 * self.padding) / self.stride + 1
        w_out = (w_x - w_filter + 2 * self.padding) / self.stride + 1
        # the output shape is（the number of input image，output channels，length，width）
        # result = torch.zeros(x.size(0), self.out_channels, out_features, out_features)
        h_out, w_out = int(h_out), int(w_out)
        X_col = torch.nn.functional.unfold(x.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=self.padding, stride=self.stride).view(n_x, d_filter, -1, h_out*w_out)
        X_col = X_col.permute(1,2,3,0).contiguous().view(X_col.size(1),X_col.size(2),-1)
        W_col = self.weight.view(n_filters, d_filter, -1)
        result = ((W_col.unsqueeze(3)+X_col.unsqueeze(0)).max(2).values).min(1).values
        result = (result.view(n_filters, h_out, w_out, n_x)).permute(3, 0, 1, 2).contiguous()

        return result

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding
        )

# MinPlus-Min-Conv Layer (MinP-Min)
class tropical_min_min_conv(nn.Module):
# +\min\max
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(tropical_min_min_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        # The shape of the convolution kernel is (the number of convolution kernels (also the number of output channels), 
#         the number of input image channels, the length of the convolution kernel, and the width of the convolution kernel) 
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size))
        #self.bias = nn.Parameter(torch.rand(out_channels))   

    def forward(self, x):
        #  The shape of input 'x' is（the number of input image，channels，length，width）
        n_filters, d_filter, h_filter, w_filter = self.weight.size()
        n_x, d_x, h_x, w_x = x.size()
        h_out = (h_x - h_filter + 2 * self.padding) / self.stride + 1
        w_out = (w_x - w_filter + 2 * self.padding) / self.stride + 1
        # the output shape is（the number of input image，output channels，length，width）
        # result = torch.zeros(x.size(0), self.out_channels, out_features, out_features)
        h_out, w_out = int(h_out), int(w_out)
        X_col = torch.nn.functional.unfold(x.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=self.padding, stride=self.stride).view(n_x, d_filter, -1, h_out*w_out)
        X_col = X_col.permute(1,2,3,0).contiguous().view(X_col.size(1),X_col.size(2),-1)
        W_col = self.weight.view(n_filters, d_filter, -1)
        result = ((W_col.unsqueeze(3)+X_col.unsqueeze(0)).min(2).values).min(1).values
        result = (result.view(n_filters, h_out, w_out, n_x)).permute(3, 0, 1, 2).contiguous()

        return result

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding
        )

# MaxPlus-Max-Conv Layer (MaxP-Max)
class tropical_max_max_conv(nn.Module):
# +\min\max
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(tropical_max_max_conv, self).__init__()
       self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        # The shape of the convolution kernel is (the number of convolution kernels (also the number of output channels), 
        # the number of input image channels, the length of the convolution kernel, and the width of the convolution kernel) 
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size))
        #self.bias = nn.Parameter(torch.rand(out_channels))   

    def forward(self, x):
        #  The shape of input 'x' is（the number of input image，channels，length，width）
        n_filters, d_filter, h_filter, w_filter = self.weight.size()
        n_x, d_x, h_x, w_x = x.size()
        h_out = (h_x - h_filter + 2 * self.padding) / self.stride + 1
        w_out = (w_x - w_filter + 2 * self.padding) / self.stride + 1
        # the output shape is（the number of input image，output channels，length，width）
        # result = torch.zeros(x.size(0), self.out_channels, out_features, out_features)
        h_out, w_out = int(h_out), int(w_out)
        X_col = torch.nn.functional.unfold(x.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=self.padding, stride=self.stride).view(n_x, d_filter, -1, h_out*w_out)
        X_col = X_col.permute(1,2,3,0).contiguous().view(X_col.size(1),X_col.size(2),-1)
        W_col = self.weight.view(n_filters, d_filter, -1)
        result = ((W_col.unsqueeze(3)+X_col.unsqueeze(0)).max(2).values).max(1).values
        result = (result.view(n_filters, h_out, w_out, n_x)).permute(3, 0, 1, 2).contiguous()

        return result

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding
        )

'''
The following are various network definitions 
'''

# Conv + Conv
def net0(opt):
    model = nn.Sequential(
        nn.Conv2d(opt.channels, opt.k1, kernel_size=opt.kernel, stride=opt.stride, padding=1),
        nn.Conv2d(opt.k1, opt.k2, kernel_size=opt.kernel, stride=opt.stride, padding=1),
        nn.Flatten(),
        nn.Linear(196, 10)
    )
    summary(model, (opt.channels,opt.image_height,opt.image_width))
    return model

# MinP_S + MaxP_S
def net1(opt):
     # The required input data size format (n, c, h, w), that is, the number, channel, height, width
     # This example is torch.Size([batchsize, 1, 28, 28]) 
    model = nn.Sequential(
        tropical_min_sum_conv(opt.channels, opt.k1, kernel_size=opt.kernel, stride=opt.stride, padding=1),
        tropical_max_sum_conv(opt.k1, opt.k2, kernel_size=opt.kernel, stride=opt.stride, padding=1),
        nn.Flatten(),
        # 196 needs to be calculated, 10 categories 
        nn.Linear(196,10),
    )
    summary(model, (opt.channels,opt.image_height,opt.image_width))
    return model

# MinP_Max + MaxP_Min
def net2(opt):
    model = nn.Sequential(
        tropical_min_max_conv(opt.channels, opt.k1, kernel_size=opt.kernel, stride=opt.stride, padding=1),
        tropical_max_min_conv(opt.k1, opt.k2, kernel_size=opt.kernel, stride=opt.stride, padding=1),
        nn.Flatten(),
        nn.Linear(196, 10),
    )
    summary(model, (opt.channels,opt.image_height,opt.image_width))
    return model

# MinP_S + Conv
def net3(opt):
    model = nn.Sequential(
        tropical_min_sum_conv(opt.channels, opt.k1, kernel_size=opt.kernel, stride=opt.stride, padding=1),
        nn.Conv2d(opt.k1, opt.k2, kernel_size=opt.kernel, stride=opt.stride, padding=1),
        nn.Flatten(),
        nn.Linear(196, 10),
    )
    summary(model, (opt.channels,opt.image_height,opt.image_width))
    return model

# MinP_Min + MaxP_Max
def net4(opt):
    model = nn.Sequential(
        tropical_min_min_conv(opt.channels, opt.k1, kernel_size=opt.kernel, stride=opt.stride, padding=1),
        tropical_max_max_conv(opt.k1, opt.k2, kernel_size=opt.kernel, stride=opt.stride, padding=1),
        nn.Flatten(),
        nn.Linear(196,10)
    )
    summary(model, (opt.channels,opt.image_height,opt.image_width))
    return model

# MinP_Min + Conv
def net5(opt):
    model = nn.Sequential(
        tropical_min_min_conv(opt.channels, opt.k1, kernel_size=opt.kernel, stride=opt.stride,padding=1),
        nn.Conv2d(opt.k1, opt.k2, kernel_size=opt.kernel, stride=opt.stride, padding=1),
        nn.Flatten(),
        nn.Linear(196, 10),
    )
    summary(model, (opt.channels,opt.image_height,opt.image_width))
    return model

# MinP_Max + Conv
def net6(opt):
    model = nn.Sequential(
        tropical_min_max_conv(opt.channels, opt.k1, kernel_size=opt.kernel, stride=opt.stride,padding=1),
        nn.Conv2d(opt.k1, opt.k2, kernel_size=opt.kernel, stride=opt.stride,padding=1),
        nn.Flatten(),
        nn.Linear(196, 10),
    )
    summary(model, (opt.channels,opt.image_height,opt.image_width))
    return model

'''
The following are functions for loading data, training functions, and saving results 
'''
def load_data():
    if opt.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        # Read the data set, the read format is TensorDataset 
        traintd = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testtd = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif opt.dataset == 'cifar10':
        transform = transforms.Compose(
        [
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor()
        #  ,
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform1 = transforms.Compose(
            [
            transforms.ToTensor()
            # ,
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        traintd = datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True, transform=transform1)
        testtd = datasets.CIFAR10(root='./data/cifar10', train=False,
                                            download=True, transform=transform1)
    else:
        print("Data set doesn't exist.")
        exit(0)
    return traintd, testtd

def train_batch(model, loss_func, xb, yb, opt, optimizer=None):
    '''
    In the training phase, back-propagation and optimizer step are required, 
    and the optimizer is not enabled in the test phase, but the loss is obtained
    '''
    # xb = xb.cuda()
    # yb = yb.cuda()
    pred = model(xb.float())
    if opt.loss == 'MSE':
        loss = loss_func(yb.view(yb.shape[0]).float(), pred.view(yb.shape[0]).float())
    else:
        loss = loss_func(pred, yb.view(yb.shape[0]).long())
    
    # The comparison finds the number of correctly predicted labels in each batch correct
    final_pred = torch.max(pred.data, 1)
    correct = torch.where(final_pred[1] == yb.data)[0]

    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), len(xb), len(correct)

def fit(opt, model, loss_func, optimizer, train_dl, test_dl):
    t = 0
    loss_history = []
    # early stopping's parameters
    patience = opt.patience
    trigger_times = 0
    maxtest_acc = -1.0
    maxtrain_acc = 0
    mintest_loss = 100
    mintrain_loss = 100

    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    for epoch in range(opt.epoch):
        print("\n---------------EPOCH", epoch+1, "/", opt.epoch, "START---------------")
        begin = time.time()

        # The role of model.train()\model.eval() is: enable\disable BatchNormalization and Dropout, set BatchNormalization and Dropout to True\False
        model.train()
        dataiter = iter(train_dl)
        train_loss = 0
        
        scheduler.step()
        # print("Learning Rate:", optimizer.state_dict()['param_groups'][0]['lr'])

        try:
            with tqdm(range(len(train_dl)), ncols=80) as tq:
                for i in tq:
                    tq.set_description("Batch %i"%i)
                    tq.set_postfix(TrainLoss_AVG=train_loss/(i+1))
                    xb, yb = next(dataiter)
                    cost, _, _ = train_batch(model, loss_func, xb, yb, opt, optimizer)
                    train_loss += cost
        
        except KeyboardInterrupt:
            tq.close()
            raise
        tq.close()

        if epoch % opt.save_loss_freq == 0:
            loss_history.append(train_loss / len(train_dl))

        if (epoch % opt.print_freq) == 0 or epoch == (opt.epoch -1):
            model.eval()
            with torch.no_grad():
                losses, nums, correct = zip(*[train_batch(model, loss_func, xb, yb, opt) for xb, yb in test_dl])
                lossest, numst, correctt = zip(*[train_batch(model, loss_func, xb, yb, opt) for xb, yb in train_dl])
            train_acc = np.sum(correctt)/(60000)*100
            test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            test_acc = np.sum(correct)/(10000)*100
            print("\nepoch: ", epoch, "TEST LOSS AVG", test_loss, "\n\nTEST ACC", test_acc, "%")
        
        finished = time.time()
        t += finished - begin
        print("\nSPEND", '%.0f' % (t//3600), "h", '%.0f' % (t%3600//60), "m", '%.0f' % (t%3600%60), "s", "IN TOTAL.")
        print("---------------\tEnd\t---------------\n")

        # early stopping
        if test_acc <= maxtest_acc:
            trigger_times += 1
            if trigger_times >= patience and epoch > opt.fitepoch:
                parameters, time_str = caculate_acc(loss_history, maxtest_acc, maxtrain_acc, mintrain_loss / len(train_dl), mintest_loss)
                save_result(parameters, opt, model, t, time_str)
                return
        else:
            trigger_times = 0
            maxtest_acc = test_acc
            mintest_loss = test_loss
            maxtrain_acc = train_acc
            mintrain_loss = train_loss
            if epoch >= opt.fitepoch:
                torch.save(model.state_dict(), './modelsepoch/'+str(opt.bs)+'params.pkl')
    parameters, time_str = caculate_acc(loss_history, maxtest_acc, maxtrain_acc, mintrain_loss / len(train_dl), mintest_loss)
    save_result(parameters, opt, model, t, time_str)

def caculate_acc(loss_history, test_acc, train_acc, train_loss, test_loss):
    time_str = time.strftime("%Y%m%d") + time.strftime("_%H%M%S")
    # Draw loss curve
    # epoch = []
    # for i in range(len(loss_history)):
    #     epoch.append((i+1))
    # plt.plot(epoch, loss_history, label='loss')
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # name_path = './loss/' + time_str +'.png'
    # plt.savefig(name_path)

    parameters = {
        'loss_history': loss_history,
        'acc_train': train_acc,
        'acc_test': test_acc,
        'loss_train': train_loss,
        'loss_test': test_loss
    }
    return parameters, time_str

def train(traintd, testtd, opt):
    '''
    Initialize the model.
    '''
    if opt.net == 'net0':
        model = net0(opt)
    elif opt.net == 'net1':
        model = net1(opt)
    elif opt.net == 'net2':
        model = net2(opt)
    elif opt.net == 'net3':
        model = net3(opt)
    elif opt.net == 'net4':
        model = net4(opt)
    elif opt.net == 'net5':
        model = net5(opt)
    elif opt.net == 'net6':
        model = net6(opt)
    else:
        print("INPUT ERROR : Net doesn't exist.")
        exit(0)

    # Data format conversion.
    train_dl = DataLoader(traintd, batch_size = opt.bs, num_workers= 4, shuffle=True)
    test_dl = DataLoader(testtd, batch_size = opt.bs, num_workers= 4)
    gpus = [0]
   
    print("Train Set:", len(train_dl)*opt.bs//1000, "k")
    print("Test Set:", len(test_dl)*opt.bs//1000, "k")

    # use gpu
    if opt.cuda:
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        # model = model.cuda()
        # print(next(model.parameters()).device)

    # loss function
    if opt.loss == 'crossentropy':
        loss_func = F.cross_entropy
    elif opt.loss == 'MSE':
        loss_func = nn.MSELoss()
    
    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=5e-4, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)
    
    
    # train part
    fit(opt, model, loss_func, optimizer, train_dl, test_dl)

def save_result(parameters, opt, model, t, time_str):
    result_file = open('tropical_convnoexpand.txt', mode='a')
    result_file.writelines(["\n======================== ", time_str, " =======================\n"])
    result_file.writelines([" ".join(sys.argv), "\n"])
    result_file.writelines(["\n", str(model), "\n"])
    result_file.writelines(["\n", "----------------------------------------------------------------"])
    # result_file.writelines(["\n", "Total params: {0:,}".format(total_params)])
    # result_file.writelines(["\n", "Trainable params: {0:,}".format(trainable_params)])
    # result_file.writelines(["\n", "Non-trainable params: {0:,}".format(total_params - trainable_params)])
    result_file.writelines(["\n", "----------------------------------------------------------------", "\n"])
    result_file.writelines(["\n---OPTIONS---\n", str(opt), "\n"])
    result_file.writelines(["\n---LOSS HISTORY---\n", str(parameters['loss_history'])])
    result_file.writelines(["\n\n---TRAIN ACC---\n", str(parameters['acc_train'])])
    result_file.writelines(["\n\n---TRAIN LOSS---\n", str(parameters['loss_train'])])
    result_file.writelines(["\n\n---TEST ACC---\n", str(parameters['acc_test'])])
    result_file.writelines(["\n\n---TEST LOSS---\n", str(parameters['loss_test'])])
    result_file.writelines(["\n\nSPEND ", '%.0f' % (t//3600), " h ", '%.0f' % (t%3600//60), " m ", '%.0f' % (t%3600%60), " s ", "IN TOTAL.", '\n================================================================\n\n'])    
    result_file.close()
    os.rename('./modelsepoch/'+str(opt.bs)+'params.pkl', './modelsepoch/'+time_str+'.pth')

def options_func():
    optParser = OptionParser()
    optParser.add_option("--net", action = "store", type = "str", dest = 'net', default = 'net1', help = "Choosing which net structure to use [1\2\3].")
    optParser.add_option("--dataset", action = "store", type = "str", dest = 'dataset', default = 'mnist', help = "Choosing  dataset [mnist\fashion-mnist\random\txt\CIFAR10\STL10\].")
    optParser.add_option('--channels', action = "store", type = "int", dest = 'channels', default = 1, help = "Input channels of train data.")
    optParser.add_option('--dim', action = "store", type = "int", dest = 'dim', default = 784, help = "Dimension of train data. mnist-784/STL10-27468/CIFAR10-3072")
    optParser.add_option('--out', action = "store", type = "int", dest = 'out', default = 10, help = "Dimension of out data.")
    optParser.add_option('--k1', action = "store", type = "int", dest = 'k1', default = 4, help = "The count of hidden nodes.")
    optParser.add_option('--k2', action = "store", type = "int", dest = 'k2', default = 4, help = "The count of hidden nodes.")
    optParser.add_option('--k3', action = "store", type = "int", dest = 'k3', default = 196, help = "The count of linner layers param.")
    optParser.add_option('--kernel', action = "store", type = "int", dest = 'kernel', default = 3, help = "Kernel size.")
    optParser.add_option('--stride', action = "store", type = "int", dest = 'stride', default = 2, help = "stride.")
    optParser.add_option('--lr', action = "store", type = "float",dest = 'lr', default = 0.02, help = "The learning rate of training.")
    optParser.add_option("--loss", action = "store", type = "str", dest = 'loss', default = 'crossentropy', help = "Loss Function.")
    optParser.add_option("--epoch", action = "store", type = "int", dest = 'epoch', default = 100, help = "Training epoch.")
    optParser.add_option('--print_freq', action = "store", type = "int", dest = 'print_freq', default = 1, help = "The frequency of printing.")
    optParser.add_option('--save_loss_freq', action = "store", type = "int", dest = 'save_loss_freq', default = 1, help = "The frequency of saving loss.")
    optParser.add_option('--bs', action = "store", type = "int", dest = 'bs', default = 64, help = "batch_size")
    optParser.add_option('--momentum', action = "store", type = "float", dest = 'momentum', default = 0.8, help = "Momentum of optimizer")
    optParser.add_option("--cuda", action = "store", type = "str", dest = 'cuda', default = False, help = "use cuda or not")
    optParser.add_option("--fitepoch", action = "store", type = "int", dest = 'fitepoch', default = 30, help = "the min epoch after using early stopping")
    optParser.add_option("--patience", action = "store", type = "int", dest = 'patience', default = 2, help = "the patience with using early stopping")
    optParser.add_option("--image_height", action = "store", type = "int", dest = 'image_height', default = 28, help = "height of image")
    optParser.add_option("--image_width", action = "store", type = "int", dest = 'image_width', default = 28, help = "width of image")
    options, arguments = optParser.parse_args()
    print('\n------------ Options -------------\n', options, '\n-------------- End ----------------\n')
    
    return options, arguments

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)
    # load parameters
    opt, args = options_func()
    # load data
    traintd, testtd = load_data()
    # train and save result
    train(traintd, testtd, opt)
