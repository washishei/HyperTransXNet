from torch import nn
import sys
import torch.nn.functional as F
import torch
import collections
from torch.nn import init
import math

# sys.path.append('../kan_convolutional')

from .KANConv import KAN_Convolutional_Layer
from .KANLinear import KANLinear
from .ConvKAN import ConvKAN
class KKAN_Convolutional_Network(nn.Module):
    def __init__(self, input_channels, n_convs, output_channles):
        super().__init__()
        self.channels = input_channels * n_convs
        self.conv1 = KAN_Convolutional_Layer(
            n_convs=n_convs,
            kernel_size=(3, 3),
            padding=(1, 1),
            # device=device
        )

        self.conv2 = nn.Conv2d(
            self.channels,
            output_channles,
            kernel_size=1,
            # device=device
        )
        self.ConvKAN1 = ConvKAN(in_channels=input_channels, out_channels=output_channles, kernel_size=3, stride=1, padding=1, version="Fast")
        self.flat = nn.Flatten()

        self.kan1 = KANLinear(
            625,
            10,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0, 1],
        )

    def forward(self, x):
        # x = self.conv1(x)
        # print(x.shape)
        x = self.ConvKAN1(x)
        return x


class KKAN_Convolutional_Network_New(nn.Module):
    def __init__(self, input_channels, n_convs, output_channles):
        super().__init__()
        self.channels = input_channels * n_convs
        self.conv1 = KAN_Convolutional_Layer(
            n_convs=n_convs,
            kernel_size=(3, 3),
            padding=(1, 1),
            # device=device
        )

        self.conv2 = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size=3,
            padding=1,
            groups=input_channels,
            stride=1,
            bias=True
            # device=device
        )
        self.ConvKAN1 = ConvKAN(in_channels=input_channels, out_channels=output_channles, kernel_size=3, stride=1, padding=1, version="Fast")
        self.flat = nn.Flatten()

        self.kan1 = KANLinear(
            625,
            10,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0, 1],
        )

    def forward(self, x):
        # x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.ConvKAN1(x)
        return x
class KConv2(torch.nn.Module):
    def __init__(self, input_channels, flatten=True):
        super(KConv2, self).__init__()
        self.feature_size = 64
        self.name = "kconv2"

        self.layer1 = nn.Sequential(collections.OrderedDict([
          ('conv',    KKAN_Convolutional_Network(input_channels, 4, 64,)),
          ('bn',      nn.BatchNorm2d(64)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        self.layer2 = nn.Sequential(collections.OrderedDict([
          ('conv',    KKAN_Convolutional_Network(64, 4, 128,)),
          ('bn',      nn.BatchNorm2d(128)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        self.layer3 = nn.Sequential(collections.OrderedDict([
          ('conv',    KKAN_Convolutional_Network(128, 4, 256,),),
          ('bn',      nn.BatchNorm2d(256)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.MaxPool2d(kernel_size=2, stride=2)),
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
        if(self.is_flatten): h = self.flatten(h)
        return h

class KC2DEtAl(nn.Module):
    """
    HYPERSPECTRAL CNN FOR IMAGE CLASSIFICATION & BAND SELECTION, WITH APPLICATION
    TO FACE RECOGNITION
    Vivek Sharma, Ali Diba, Tinne Tuytelaars, Luc Van Gool
    Technical Report, KU Leuven/ETH Zürich
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=15):
        super(KC2DEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.feature = KConv2(input_channels, flatten=True)

        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step
        self.features_size = self._get_final_flattened_size()
        # self.out_finetune = nn.Linear(self.features_size, 1024)
        self.KAN1 = KANLinear(
            self.features_size,
            n_classes,
        )

        #self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.feature(x)
            w, h = x.size()
        return w * h

    def forward(self, x):
        x = x.squeeze()
        x = self.feature(x)
        # print(x.shape)
        # x = self.out_finetune(x)
        x = self.KAN1(x)
        return x

if __name__ == '__main__':
    img_size = 15
    x = torch.rand(64, 9, img_size, img_size)
    model = KC2DEtAl(9, 10, patch_size=15)
    y = model(x)
    print(y.shape)
    model.eval()
    from thop import profile
    flops, params = profile(model, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')