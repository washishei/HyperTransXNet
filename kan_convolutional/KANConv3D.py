import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import warnings
import unfoldNd
from enum import Enum
import collections

from .KANLinear import KANLinear

"""
The parameters stride, padding, dilation can either be:

    + a single int - in which case the same value is used for the depth, height and width dimension
    + a tuple of three ints - in which case, the first int is used for the depth dimension, the second int for the height dimension and the third int for the width dimension
"""


class effConvKAN3D(torch.nn.Module):
    """
    This is an implementation of convolutional layers using Kolmogorov-Arnold Networks (KANs)
    """

    def __init__(self,
                 in_channels,  # number of input channels
                 out_channels,  # number of output channels
                 kernel_size,  # the size of the kernel, MUST be a 3-tuple of type (depth, height, width) kernel_sizes
                 stride=1,  # controls stride for cross-correlation
                 padding=0,  # controls the amount of padding applied to the input
                 dilation=1,  # controls the spacing between the kernel points; also known as the à trous algorithm
                 padding_mode='zeros',  # 'zeros' ONLY

                 # refer to https://github.com/Blealtan/efficient-kan for what these variables expect
                 grid_size=5,
                 spline_order=3,
                 scale_noise=0.1,
                 scale_base=1.0,
                 scale_spline=1.0,
                 enable_standalone_scale_spline=True,
                 base_activation=torch.nn.SiLU,
                 grid_eps=0.02,
                 grid_range=[-1, 1], device=None, dtype=None):

        if not effConvKAN3D._is_3_tuple(kernel_size):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        super(effConvKAN3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        if (self.padding_mode != 'zeros'):
            warnings.warn("Warning! Padding_mode is assumed to be 'zeros'. Instable results may arise")

        self.unfold = unfoldNd.UnfoldNd(self.kernel_size, dilation=self.dilation, padding=self.padding,
                                        stride=self.stride)

        self.linear = KANLinear(self.in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2],
                                self.out_channels,
                                grid_size,
                                spline_order,
                                scale_noise,
                                scale_base,
                                scale_spline,
                                enable_standalone_scale_spline,
                                base_activation,
                                grid_eps,
                                grid_range)

    def forward(self, x):
        """
        forward propagation,
        input assumed to be of form: (N,Cin,Depth,Height,Width)
        """

        assert x.dim() == 5
        batch_size, in_channels, depth, height, width = x.size()
        assert in_channels == self.in_channels

        # Unfold the input tensor to extract flattened sliding blocks from a batched input tensor.
        # Input:  [batch_ size, in_channels, depth, height, width]
        # Output: [batch_size, in_channels * ∏(kernel_size), L], where L is the total number of such blocks.
        blocks = self.unfold(x)

        # Move the flattened sliding blocks into the last dimension such that:
        # Input: [batch_size, in_channels * ∏(kernel_size), L]
        # Output: [batch_size, L, in_channels *  ∏(kernel_size)]
        blocks = blocks.transpose(1, 2)  # left to right dimensions: 0, 1, 2 (we are swapping 1 and 2)

        # reshape such that batch_size and number of blocks (L) are mixed:
        # Input: [batch_size, L, in_channels *  ∏(kernel_size)]
        # Output: [batch_size * L, in_channels *  ∏(kernel_size)]
        blocks = blocks.reshape(-1, self.in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2])

        # Apply KAN Linear layer on all blocks (across all batches)
        # Input: [batch_size * L, in_channels *  ∏(kernel_size)]
        # Output: [batch_size * L, out_channels]
        out = self.linear(blocks)

        # Reshape the output to retrieve batch dimension
        # Input: [batch_size * L, out_channels]
        # Output: [batch_size, L, out_channels]
        out = out.reshape(batch_size, -1, out.shape[-1])

        # Transpose to move the number of blocks to the last dimension
        # Input: [batch_size, L, out_channels]
        # Output: [batch_size, out_channels, L]
        out = out.transpose(1, 2)

        # ensure all other parameters for convolution are tuples to find dimensionality of output feature map
        if (not effConvKAN3D._is_3_tuple(self.stride)):
            self.stride = (self.stride, self.stride, self.stride)
        if (not effConvKAN3D._is_3_tuple(self.padding)):
            self.padding = (self.padding, self.padding, self.padding)
        if (not effConvKAN3D._is_3_tuple(self.dilation)):
            self.dilation = (self.dilation, self.dilation, self.dilation)

        depth_out = ((depth + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[
            0]) + 1
        height_out = ((height + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[
            1]) + 1
        width_out = ((width + 2 * self.padding[2] - self.dilation[2] * (self.kernel_size[2] - 1) - 1) // self.stride[
            2]) + 1

        # Reshape to retrieve depth, height, and width of output feature map
        # Input: [batch_size, out_channels, L]
        # Output: [batch_size, out_channels, depth_out, height_out, width_out]
        out = out.view(batch_size, self.out_channels, depth_out, height_out, width_out)

        return out

    def _is_3_tuple(var):
        """
        Helper function to check if a variable is a 3-tuple
        """
        return isinstance(var, tuple) and len(var) == 3

class KConv3(torch.nn.Module):
    def __init__(self, input_channels, flatten=True):
        super(KConv3, self).__init__()
        self.feature_size = 64
        self.name = "conv3"

        self.layer1 = nn.Sequential(collections.OrderedDict([
          ('conv',    effConvKAN3D(1, 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
          ('bn',      nn.BatchNorm3d(4)),
          ('relu',    nn.ReLU()),
          # ('avgpool', nn.Conv3d(4, 4, (3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))),
          ('avgpool', nn.MaxPool3d(kernel_size=2, stride=2)),
        ]))

        self.layer2 = nn.Sequential(collections.OrderedDict([
          ('conv',    effConvKAN3D(4, 8, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
          ('bn',      nn.BatchNorm3d(8)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.MaxPool3d(kernel_size=2, stride=2)),
        ]))

        self.layer3 = nn.Sequential(collections.OrderedDict([
          ('conv',    effConvKAN3D(8, 16, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
          ('bn',      nn.BatchNorm3d(16)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.MaxPool3d(kernel_size=2, stride=2)),
        ]))

        self.layer4 = nn.Sequential(collections.OrderedDict([
          ('conv',    effConvKAN3D(256, 512, kernel_size=3, stride=1, padding=1)),
          ('bn',      nn.BatchNorm2d(512)),
          ('relu',    nn.ReLU()),
          #('avgpool', nn.AvgPool2d(kernel_size=4))
          ('glbpool', nn.AdaptiveAvgPool3d(1))
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
        # print(h.shape)
        h = self.layer2(h)
        h = self.layer3(h)
      #  h = self.layer4(h)
        #print(h.size())
        if(self.is_flatten): h = self.flatten(h)
        return h

class KC3DEtAl(nn.Module):
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
        super(KC3DEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv = nn.Conv2d(input_channels, 30, 1)
        self.feature = KConv3(30, flatten=True)

        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step
        self.features_size = self._get_final_flattened_size()
        self.out_finetune = nn.Linear(self.features_size, 256)
        self.KAN1 = KANLinear(
            256,
            n_classes,
        )

        #self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, 30, self.patch_size, self.patch_size)
            )
            x = self.feature(x)
            w, h = x.size()
        return w * h

    def forward(self, x):
        x = x.squeeze()
        x = self.conv(x)
        x = x.unsqueeze(dim=1)
        # print(x.shape)
        x = self.feature(x)
        x = self.out_finetune(x)
        x = self.KAN1(x)
        return x

if __name__ == '__main__':
    img_size = 15
    x = torch.rand(64, 1, 9, img_size, img_size)
    model = KC3DEtAl(9, 10, patch_size=15)
    y = model(x)
    print(y.shape)
    model.eval()
    from thop import profile
    flops, params = profile(model, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')