# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
# from collections import OrderedDict
from BasicModule import BasicModule
from SIIS_Kernel import SIIS
from MFNF import AttHarm2d
from SDI import SDI

def conv3x3(in_channels,
            out_channels,
            stride=1,
            padding=1,
            bias=True,
            groups=1):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups
    )


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.AttHarm2d1 = AttHarm2d(self.out_channels, self.out_channels,3)
        # self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels,
            kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.AttHarm2d2 = AttHarm2d(self.out_channels, self.out_channels,3)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.AttHarm2d1(self.conv1(x)))
        # residual = x
        # x = F.relu(self.AttHarm2d1(self.conv2(x)))
        x = F.relu((self.conv2(x)))
        before_pool = x
        # x = x + residual
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module): #add additional parameters:width and height of feature size
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 height,
                 width,
                 merge_mode='concat',
                 up_mode='transpose'
                 ):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.upconv = upconv2x2(
            self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.SDI1 = SDI(self.out_channels,self.out_channels, self.out_channels, height, width)
        # self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels,
            kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.SDI2 = SDI(self.out_channels,self.out_channels, self.out_channels, height, width)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        # residual = self.conv1(x)
        x = F.relu(self.SDI1(self.conv1(x)))
        # x = F.relu(self.SDI2(self.conv2(x)))
        x = F.relu((self.conv2(x)))
        # x = x + residual
        return x


class UNet(BasicModule):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """
    def __init__(self,
                 num_classes,
                 in_channels=3,
                 depth=5,
                 start_filts=64,
                 up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("%s is not a valid mode" % (up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("%s is not a valid mode" % (up_mode))

        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError(
                "up_mode \"upsample\" is incompatible with merge_mode \"add\" at the moment, "
                "because it doesn't make sense to use nearest neighbour to reduce "
                "depth channels (by half).")

        self.model_name = 'UNet'
        self.num_classes = num_classes
        self.in_channels = in_channels
        # print(in_channels)
        self.start_filts = start_filts
        self.depth = depth
        self.down_convs = []
        self.up_convs = []
        self.height =[16,32,64,128,256]
        # create the encoder pathway and add to a list
        outs = 0
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, self.height[i+1], self.height[i+1], up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)
        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        # self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = x.squeeze()
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            # print(x.shape)
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


class UNet_SIIS(BasicModule):
    """ UNet + SIIS
    """
    def __init__(self,
                 num_classes,
                 siis_size=[32, 32], width=1, kw=9, dim=128, arch=1,
                 in_channels=3,
                 depth=5,
                 start_filts=64,
                 up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet_SIIS, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("%s is not a valid mode" % (up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("%s is not a valid mode" % (up_mode))

        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError(
                "up_mode \"upsample\" is incompatible with merge_mode \"add\" at the moment, "
                "because it doesn't make sense to use nearest neighbour to reduce "
                "depth channels (by half).")

        self.model_name = 'UNet_siis'
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.down_convs = []
        self.up_convs = []
        self.height = [16, 32, 64, 128, 256]

        # create the encoder pathway and add to a list
        outs = 0
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        self.siis = SIIS(siis_size, width, kw, dim, arch)  # size=siis_size, dim=dim
        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, self.height[i+1], self.height[i+1], up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)
        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        # self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        # 256 -> 128 -> 64 -> 32 -> 16
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        # 16 -> 32 -> 64 -> 128 -> 256
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            if i == 0:
                before_pool = self.siis(before_pool)
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


def build_model(bands, num_classes=2):
    model = UNet(in_channels=bands, num_classes=num_classes)
    return model
