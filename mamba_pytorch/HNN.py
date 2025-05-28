import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import collections
from torch.nn import init
import math
import collections
import torchvision
from pylab import *

def dct1_filters(k=3, groups=1, expand_dim=1, level=None, DC=True, l1_norm=True):
    if level is None:
        nf = k ** 2 - int(not DC)
    else:
        if level <= k:
            nf = level * (level + 1) // 2 - int(not DC)
        else:
            r = 2 * k - 1 - level
            nf = k ** 2 - r * (r + 1) // 2 - int(not DC)
    filter_bank = np.zeros((nf, k, k), dtype=np.float32)
    m = 0
    for i in range(k):
        for j in range(k):
            if (not DC and i == 0 and j == 0) or (not level is None and i + j >= level):
                continue
            for x in range(k):
                for y in range(k):
                    filter_bank[m, x, y] = math.cos((math.pi * x * i) / (k-1)) * math.cos(
                        (math.pi * y * j) / (k-1))
            if l1_norm:
                filter_bank[m, :, :] /= np.sum(np.abs(filter_bank[m, :, :]))
            else:
                ai = 1.0 if i > 0 else 1.0 / math.sqrt(2.0)
                aj = 1.0 if j > 0 else 1.0 / math.sqrt(2.0)
                filter_bank[m, :, :] *= (2.0 / k) * ai * aj
            m += 1
    filter_bank = np.tile(np.expand_dims(filter_bank, axis=expand_dim), (groups, 1, 1, 1))
    return torch.FloatTensor(filter_bank)

def dct2_filters(k=3, groups=1, expand_dim=1, level=None, DC=True, l1_norm=True):
    if level is None:
        nf = k ** 2 - int(not DC)
    else:
        if level <= k:
            nf = level * (level + 1) // 2 - int(not DC)
        else:
            r = 2 * k - 1 - level
            nf = k ** 2 - r * (r + 1) // 2 - int(not DC)
    filter_bank = np.zeros((nf, k, k), dtype=np.float32)
    m = 0
    for i in range(k):
        for j in range(k):
            if (not DC and i == 0 and j == 0) or (not level is None and i + j >= level):
                continue
            for x in range(k):
                for y in range(k):
                    filter_bank[m, x, y] = math.cos((math.pi * (x + .5) * i) / k) * math.cos(
                        (math.pi * (y + .5) * j) / k)
            if l1_norm:
                filter_bank[m, :, :] /= np.sum(np.abs(filter_bank[m, :, :]))
            else:
                ai = 1.0 if i > 0 else 1.0 / math.sqrt(2.0)
                aj = 1.0 if j > 0 else 1.0 / math.sqrt(2.0)
                filter_bank[m, :, :] *= (2.0 / k) * ai * aj
            m += 1
    filter_bank = np.tile(np.expand_dims(filter_bank, axis=expand_dim), (groups, 1, 1, 1))
    return torch.FloatTensor(filter_bank)


class Harm2d(nn.Module):

    def __init__(self, ni, no, kernel_size, stride=1, padding=0, bias=True, dilation=1, use_bn=False, level=None,
                 DC=True, groups=1):
        super(Harm2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dct = nn.Parameter(
            dct2_filters(k=kernel_size, groups=ni if use_bn else 1, expand_dim=1 if use_bn else 0, level=level, DC=DC),
            requires_grad=False)
        self.dct1 = nn.Parameter(
            dct1_filters(k=kernel_size, groups=ni if use_bn else 1, expand_dim=1 if use_bn else 0, level=level, DC=DC),
            requires_grad=False)

        nf = self.dct.shape[0] // ni if use_bn else self.dct.shape[1]
        if use_bn:
            self.bn = nn.BatchNorm2d(ni * nf, affine=False)
            self.weight = nn.Parameter(
                nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups * nf, 1, 1), mode='fan_out',
                                        nonlinearity='relu'))
        else:
            self.weight = nn.Parameter(
                nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups, nf, 1, 1), mode='fan_out',
                                        nonlinearity='relu'))
        self.bias = nn.Parameter(nn.init.zeros_(torch.Tensor(no))) if bias else None

    def forward(self, x):
        if not hasattr(self, 'bn'):
            filt = torch.sum(self.weight * self.dct, dim=2)
            x = F.conv2d(x, filt, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation,
                         groups=self.groups)
            return x
        else:
            x = F.conv2d(x, self.dct, stride=self.stride, padding=self.padding, dilation=self.dilation,
                         groups=x.size(1))
            x = self.bn(x)
            x = F.conv2d(x, self.weight, bias=self.bias, padding=0, groups=self.groups)
            return x

def harm3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, groups=1, level=None):
    """3x3 harmonic convolution with padding"""
    return Harm2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                               bias=True, use_bn=True, level=level)

class HConvEtAl(torch.nn.Module):
    def __init__(self, input_channels, flatten=True, level=None):
        super(HConvEtAl, self).__init__()
        self.feature_size = 64
        self.name = "conv4"

        self.layer1 = nn.Sequential(collections.OrderedDict([
          ('hconv',   harm3x3(input_channels, 64, stride=1, level=level)),
          ('bn',      nn.BatchNorm2d(64)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer2 = nn.Sequential(collections.OrderedDict([
          ('hconv',   harm3x3(64, 128, stride=1, level=level)),
          ('bn',      nn.BatchNorm2d(128)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer3 = nn.Sequential(collections.OrderedDict([
          ('hconv',   harm3x3(128, 256, stride=1, level=level)),
          ('bn',      nn.BatchNorm2d(256)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer4 = nn.Sequential(collections.OrderedDict([
          ('hconv',    harm3x3(256, 512, stride=1, level=level)),
          ('bn',      nn.BatchNorm2d(512)),
          ('relu',    nn.ReLU()),
          #('avgpool', nn.AvgPool2d(kernel_size=4))
          ('glbpool', nn.AdaptiveAvgPool2d(1))
        ]))

        self.is_flatten = flatten
        self.flatten = nn.Flatten()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
      #  h = self.layer4(h)
        #print(h.size())
        if(self.is_flatten): h = self.flatten(h)
        return h

class HTESTEtAl(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image classification
    Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=15, pool_size=None):
        super(HTESTEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for the convolutional
        # "and pooling layer" -> actually 3x3 2D convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        # resnet = resnet50(pretrained=True)
        # resnet_feature = list(resnet.children())[:-2]
        self.feature = HConvEtAl(input_channels, flatten=True, level=None)
        self.features_sizes = self._get_sizes()
        self.classifier = nn.Linear(self.features_sizes, n_classes)

    def _get_sizes(self):
        x = torch.zeros((self.input_channels, self.patch_size, self.patch_size))
        x = self.feature(x)
        w, h = x.size()
        size0 = w * h
        return size0

    def forward(self, x):
        # x = x.squeeze()
        x = self.feature(x)
        x = self.classifier(x)
        return x

class ConvEtAl(torch.nn.Module):
    def __init__(self, input_channels, flatten=True):
        super(ConvEtAl, self).__init__()
        self.feature_size = 64
        self.name = "conv4"

        self.layer1 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(64)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.layer2 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(128)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.layer3 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(256)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.layer4 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(512)),
          ('relu',    nn.ReLU()),
          #('avgpool', nn.AvgPool2d(kernel_size=4))
          ('glbpool', nn.AdaptiveAvgPool2d(1))
        ]))

        self.is_flatten = flatten
        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
      #  h = self.layer4(h)
        #print(h.size())
        if(self.is_flatten): h = self.flatten(h)
        # print(h.shape)
        return h

class HTESTEtAl(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image classification
    Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=15, pool_size=None):
        super(HTESTEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for the convolutional
        # "and pooling layer" -> actually 3x3 2D convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        # resnet = resnet50(pretrained=True)
        # resnet_feature = list(resnet.children())[:-2]
        self.feature = ConvEtAl(input_channels, flatten=True)
        self.features_sizes = self._get_sizes()
        self.classifier = nn.Linear(self.features_sizes, n_classes)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
        x = self.feature(x)
        w, h = x.size()
        size0 = w * h
        # print(size0)
        return size0

    def forward(self, x):
        # x = x.squeeze()
        x = self.feature(x)
        x = self.classifier(x)
        # print(x.shape)
        return x

if __name__ == '__main__':
    img_size = 15
    x = torch.rand(10, 200, img_size, img_size)
    model = HTESTEtAl(200, 16, patch_size=15)

    optimal_batch_size = 2
    from thop import profile

    flops, params = profile(model, (x,))
    with torch.autograd.profiler.profile(enabled=True) as prof:
        out = model(x)

    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    repetitions = 100
    total_time = 0
    optimal_batch_size = 2
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * optimal_batch_size) / total_time
    print("FinalThroughput:", Throughput)
    import numpy as np

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(x)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(x)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                         std_syn=std_syn,
                                                                                         mean_fps=mean_fps))
    print(mean_syn)