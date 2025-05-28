import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
def dct_filters_3d(k=3, groups=1, expand_dim=1, level=None, DC=True, l1_norm=True):
    if level is None:
        nf = k**3 - int(not DC)
    else:
        if level <= k:
            nf = (level * (level + 1) * (level + 2)) // 6 - int(not DC)
        else:
            r = 3 * k - 3 - level
            nf = k**3 - r * (r + 1) * (r + 2) // 6 - int(not DC)

    filter_bank = np.zeros((nf, k, k, k), dtype=np.float32)
    m = 0
    for i in range(k):
        for j in range(k):
            for l in range(k):
                if (not DC and i == 0 and j == 0 and l == 0) or (not level is None and i + j + l >= level):
                    continue
                for x in range(k):
                    for y in range(k):
                        for z in range(k):
                            filter_bank[m, x, y, z] = (
                                math.cos((math.pi * (x + 0.5) * i) / k) *
                                math.cos((math.pi * (y + 0.5) * j) / k) *
                                math.cos((math.pi * (z + 0.5) * l) / k)
                            )
                if l1_norm:
                    filter_bank[m, :, :, :] /= np.sum(np.abs(filter_bank[m, :, :, :]))
                else:
                    ai = 1.0 if i > 0 else 1.0 / math.sqrt(2.0)
                    aj = 1.0 if j > 0 else 1.0 / math.sqrt(2.0)
                    ak = 1.0 if l > 0 else 1.0 / math.sqrt(2.0)
                    filter_bank[m, :, :, :] *= (2.0 / k) * ai * aj * ak
                m += 1
    filter_bank = np.tile(np.expand_dims(filter_bank, axis=expand_dim), (groups, 1, 1, 1, 1))
    return torch.FloatTensor(filter_bank)

class Harm3d(nn.Module):
    def __init__(self, ni, no, kernel_size=3, stride=1, padding=0, bias=True, dilation=1, use_bn=False, level=None, DC=True, groups=1):
        super(Harm3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dct = nn.Parameter(dct_filters_3d(k=kernel_size, groups=ni if use_bn else 1, expand_dim=1 if use_bn else 0, level=level, DC=DC), requires_grad=False)
        
        nf = self.dct.shape[0] // ni if use_bn else self.dct.shape[1]
        if use_bn:
            self.bn = nn.BatchNorm3d(ni*nf, affine=False)
            self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups * nf, 1, 1, 1), mode='fan_out', nonlinearity='relu'))
        else:
            self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups, nf, 1, 1, 1), mode='fan_out', nonlinearity='relu'))
        self.bias = nn.Parameter(nn.init.zeros_(torch.Tensor(no))) if bias else None

    def forward(self, x):
        # x = x.permute(0, 2, 1, 3, 4)
        if not hasattr(self, 'bn'):
            filt = torch.sum(self.weight * self.dct, dim=2)
            x = F.conv3d(x, filt, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            # x = x.permute(0, 2, 1, 3, 4)
            return x
        else:
            x = F.conv3d(x, self.dct, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=x.size(1))
            x = self.bn(x)
            x = F.conv3d(x, self.weight, bias=self.bias, padding=0, groups=self.groups)
            # x = x.permute(0, 2, 1, 3, 4)
            return x


class TMamba3D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1, input_size: tuple = (15, 15, 15)):
        super().__init__()
        self.model_name = "TMamba3D"
        self.classes = num_classes
        self.hconv1 = Harm3d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.hconv2 = Harm3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.hconv3 = Harm3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # print(x.shape)
        # x = x.permute(0, 2, 1, 3, 4)
        x = self.hconv1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu1(x)
        # print(x.shape)
        # x = x.permute(0, 2, 1, 3, 4)
        x = self.pool1(x)
        # x = x.permute(0, 2, 1, 3, 4)
        # print(x.shape)

        x = self.hconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.hconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    input_value = np.random.randn(2, 1, 200, 21, 21)
    input_value = torch.from_numpy(input_value).float().cuda()
    print(input_value.dtype)
    model = TMamba3D(200, 2).cuda()
    model.train()
    out = model(input_value)
    from thop import profile
    flops, params = profile(model, (input_value,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    print(out.shape)


if __name__ == '__main__':
    main()