import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import einsum
from timm.models.layers import DropPath, trunc_normal_
# from .aggregation_zeropad import LocalConvolution
#from vit_pytorch.aggregation_zeropad import LocalConvolution
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import collections
import cv2
import spectral
from PIL import Image
import torch.autograd as autograd

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # print(x.shape)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=False)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=False)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=False)
        self.layernorm = LayerNorm(outplanes)
        self.feedform = FeedForward(outplanes, 4, 0.)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=False):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        x = self.act3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            # residual = residual
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x = residual + x
        x = self.layernorm(x)
        x = self.feedform(x) + x

        if return_x_2:
            return x, x2
        else:
            return x

class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv_project(x)  # [N, C, H, W]
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        return x

class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        x_r = x.transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))

class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """

    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=False)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=False)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=False)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size, dw_stride):
        super(CotLayer, self).__init__()
        expansion = 4
        self.dim = dim
        self.kernel_size = kernel_size
        self.dw_stride = dw_stride

        self.expand_block = FCUUp(inplanes=dim, outplanes=dim, up_stride=dw_stride)

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=False)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(dim * 2, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes),
            nn.Conv2d(pow(kernel_size, 2) * dim // share_planes, dim,  kernel_size=1),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.local_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=False),
            nn.Conv2d(attn_chs, self.radix * dim, 1)
        )

    def forward(self, x, y):
        _, _, H, W = x.shape
        y_r = self.expand_block(y, H, W)
        y_r = self.key_embed(y_r)
        qk = torch.cat([y_r, x], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)

        k = self.conv1x1(y_r)
        k = torch.cat([w, k], dim=1)
        k = self.local_conv(k)
        k = self.bn(k)
        k = self.act(k)

        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)

        return out.contiguous()

class SimCotLayer(nn.Module):
    def __init__(self, dim, kernel_size, dw_stride):
        super(SimCotLayer, self).__init__()
        expansion = 4
        self.dim = dim
        self.kernel_size = kernel_size
        self.dw_stride = dw_stride

        self.expand_block = FCUUp(inplanes=dim, outplanes=dim, up_stride=dw_stride)

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=False)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # self.local_conv = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=False),
            nn.Conv2d(attn_chs, self.radix * dim, 1)
        )

    def forward(self, x, y):
        _, _, H, W = x.shape
        y_r = self.expand_block(y, H, W)
        k = self.key_embed(y_r)
        x = torch.cat([x, k], dim=1)
        x = self.conv1x1(x)
        return x

class TranCotLayer(nn.Module):
    def __init__(self, dim, dw_stride):
        super(TranCotLayer, self).__init__()
        expansion = 4
        self.dim = dim
        drop = 0.

        self.squeeze_block = FCUDown(inplanes=dim, outplanes=dim, dw_stride=dw_stride)
        self.act = nn.ReLU()

        self.mlp = Mlp(in_features=dim*2, hidden_features=dim, out_features=dim, act_layer=nn.ReLU, drop=drop)

        self.embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )

        self.conv1x1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU()
        )

        self.proj_drop = nn.Dropout(drop)

    def forward(self, x, y):
        B, H, W = x.shape
        y_r = self.squeeze_block(y)
        y_r = self.embed(y_r)
        qk = torch.cat([y_r, x], dim=2)
        w = self.mlp(qk)
        w = w.view(B, H, W)
        k = self.conv1x1(y_r)
        x = torch.cat([w, k], dim=2)
        x = self.mlp(x)
        x = self.proj_drop(x)
        return x

class Simoutput(nn.Module):
    def __init__(self, dim, number):
        super(Simoutput, self).__init__()
        drop = 0.
        self.mlp = Mlp(in_features=dim, hidden_features=dim, out_features=number, act_layer=nn.ReLU, drop=drop)
        self.conv1x1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.mlp(x)
        # y = self.conv1x1(x)
        return x
class SimTranCotLayer(nn.Module):
    def __init__(self, dim, dw_stride):
        super(SimTranCotLayer, self).__init__()
        expansion = 4
        self.dim = dim
        drop = 0.

        self.squeeze_block = FCUDown(inplanes=dim, outplanes=dim, dw_stride=dw_stride)
        self.act = nn.ReLU()

        self.mlp = Mlp(in_features=dim*2, hidden_features=dim, out_features=dim, act_layer=nn.ReLU, drop=drop)

        self.embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x, y):
        B, H, W = x.shape
        y_r = self.squeeze_block(y)
        y_r = self.embed(y_r)
        x = torch.cat([x, y_r], dim=2)
        x = self.mlp(x)
        return x

class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out

class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class ACTransformerNA(nn.Module):

    def __init__(self, patch_size=16, in_chans=200, num_classes=1000, base_channel=64, channel_ratio=6, num_med_block=0,
                 embed_dim=384, depth=12, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., flatten=True):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0



        self.cls_token = nn.Parameter(torch.zeros(1, 1, 64))
        # self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, 128))
        # self.cls_token_3 = nn.Parameter(torch.zeros(1, 1, 256))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trans_cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(256, num_classes)
        )

        ### Conv clasifier
        self.is_flatten = flatten
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            # Rearrange('... () () -> ...'),
            nn.Linear(256, num_classes)
        )

        ### final classifier
        self.final_cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(512, num_classes)
        )

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, out_channels=8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )


        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=1600, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]


        ## Share layer
        self.avg_pool_c_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_t_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_c_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_t_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_c_3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_t_3 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 1 stage
        # stage_1_channel = int(base_channel * channel_ratio)
        # trans_dw_stride = patch_size // 4
        # self.layer1 = nn.Sequential(collections.OrderedDict([
        #     ('conv', nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1, bias=False)),
        #     ('bn', nn.BatchNorm2d(64)),
        #     ('relu', nn.ReLU()),
        #     ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        # ]))
        ### Stage 1 Conv
        self.layer1 = ConvBlock(inplanes=64, outplanes=64, res_conv=True, stride=1)
        ### Stage 1 Transformer
        self.trans_patch_conv_1 = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU()),
            # ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))
        self.layer_1 = LayerNorm(64)
        self.trans_1 = Block(dim=64, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )
        # 1 stage combine conv and transformer
        self.trans_conv_1 = CotLayer(64, kernel_size=3, dw_stride=1)
        self.conv_trans_1 = TranCotLayer(64, dw_stride=1)

        # 2 stage
        # self.layer2 = nn.Sequential(collections.OrderedDict([
        #     ('conv', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
        #     ('bn', nn.BatchNorm2d(128)),
        #     ('relu', nn.ReLU()),
        #     ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        # ]))
        ### Stage 2 Conv
        self.layer2 = ConvBlock(inplanes=64, outplanes=128, res_conv=True, stride=1)
        # self.conv_2 = SSConv(embed_dim, embed_dim, kernel_size=3)

        ### Stage 2 Transformer
        self.trans_patch_conv_2 = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(128)),
            ('relu', nn.ReLU()),
            # ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))
        self.layer_2 = LayerNorm(128)
        self.trans_2 = Block(dim=128, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )
        # 2 stage combine conv and transformer
        self.trans_conv_2 = CotLayer(128, kernel_size=3, dw_stride=1)
        self.conv_trans_2 = TranCotLayer(128, dw_stride=1)

        # 3 stage
        ### Stage 3 Conv
        self.layer3 = ConvBlock(inplanes=128, outplanes=256, res_conv=True, stride=1)
        # self.layer3 = nn.Sequential(collections.OrderedDict([
        #     ('conv', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)),
        #     ('bn', nn.BatchNorm2d(256)),
        #     ('relu', nn.ReLU()),
        #     ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        # ]))

        ### Stage 3 Transformer
        self.trans_patch_conv_3 = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(256)),
            ('relu', nn.ReLU()),
            # ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))
        self.layer_3 = LayerNorm(256)
        self.trans_3 = Block(dim=256, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )
        # 3 stage combine conv and transformer
        self.trans_conv_3 = CotLayer(256, kernel_size=3, dw_stride=1)
        self.conv_trans_3 = TranCotLayer(256, dw_stride=1)


        ## final stage
        self.conv_f = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn', nn.BatchNorm2d(512)),
            ('relu', nn.ReLU()),
        ]))

        self.trans_patch_conv_f = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn', nn.BatchNorm2d(512)),
            ('relu', nn.ReLU()),
        ]))
        self.layer_f = LayerNorm(512)
        self.trans_f = Block(dim=512, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward(self, x):
        # x = x.squeeze(dim=1)
        ## Stem stage
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)

        # 1 stage
        # x_base = self.conv1(x)
        x_conv = self.layer1(x)
        x_tran = self.trans_patch_conv_1(x)
        x_tran = self.layer_1(x_tran).flatten(2).transpose(1, 2)
        # x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_tran = self.trans_1(x_tran)

        # 1 combine
        # x_conv_com = x_conv
        # x_conv = self.trans_conv_1(x_conv, x_tran)
        # x_tran = self.conv_trans_1(x_tran, x_conv_com)
        ## 1 pool
        B, C, D1, D2 = x_conv.shape
        x_conv = self.avg_pool_c_1(x_conv)
        x_tran = x_tran.permute(0, 2, 1).reshape(B, C, D1, D2)
        x_tran = self.avg_pool_t_1(x_tran)
        # x_tran = x_tran.permute(0, 2, 3, 1)
        # x_tran = rearrange(x_tran, 'b c h w -> b (c h) w')


        # # 2 stage
        # x_tran = x_tran.permute(0, 2, 1).reshape(B, C, D1, D2)
        x_conv = self.layer2(x_conv)
        x_tran = self.trans_patch_conv_2(x_tran)
        x_tran = self.layer_2(x_tran).flatten(2).transpose(1, 2)
        x_tran = self.trans_2(x_tran)

        # 2 combine
        # B, C, D1, D2 = x_conv.shape
        # x_conv_com = x_conv
        # x_conv = self.trans_conv_2(x_conv, x_tran)
        # x_tran = self.conv_trans_2(x_tran, x_conv_com)
        ## 2 pool
        x_conv = self.avg_pool_c_2(x_conv)
        x_tran = x_tran.permute(0, 2, 1).reshape(B, C, D1, D2)
        x_tran = self.avg_pool_t_2(x_tran)

        # # 3 stage
        # x_tran = x_tran.permute(0, 2, 1).reshape(B, C, D1, D2)
        x_conv = self.layer3(x_conv)
        x_tran = self.trans_patch_conv_3(x_tran)
        x_tran = self.layer_3(x_tran).flatten(2).transpose(1, 2)
        x_tran = self.trans_3(x_tran)

        # 3 combine
        # x_conv_com = x_conv
        # x_conv = self.trans_conv_3(x_conv, x_tran)
        # x_tran = self.conv_trans_3(x_tran, x_conv_com)
        # x_conv_f = x_conv
        # x_tran_f = x_tran
        ## 3 pool
        B, C, D1, D2 = x_conv.shape
        x_conv = self.avg_pool_c_3(x_conv)
        # print(x_conv.shape)
        x_tran = x_tran.permute(0, 2, 1).reshape(B, C, D1, D2)
        x_tran = self.avg_pool_t_3(x_tran)
        x_tran_f = x_tran
        x_conv_f = x_conv

        # conv classification
        if (self.is_flatten): x_conv = self.flatten(x_conv)
        conv_cls = self.classifier(x_conv)

        # trans classificatio
        tran_cls = self.trans_cls_head(x_tran)

        ## final output
        # print(x_conv_f.shape)
        # x_tran_f = x_tran_f.permute(0, 2, 1).reshape(B, C, D1, D2)
        # x_tran_f = self.trans_patch_conv_f(x_tran_f)
        # x_tran_f = self.layer_f(x_tran_f).flatten(2).transpose(1, 2)
        # x_tran_f = self.trans_f(x_tran_f)
        # x_conv_f = self.conv_f(x_conv_f)
        # x_tran_f = x_tran_f.permute(0, 2, 1).reshape(x_conv_f.shape[0], x_conv_f.shape[1], x_conv_f.shape[2],
        #                                              x_conv_f.shape[3])
        # output = torch.cat([x_conv_f, x_tran_f], dim=1)
        output = torch.cat([x_conv_f, x_tran_f], dim=1)
        # print(output.shape)
        final_cls = self.final_cls_head(output)

        return conv_cls, tran_cls, final_cls

if __name__ == '__main__':
    img_size = 15
    x = torch.rand(2, 1, 200, img_size, img_size)
    net = ACTransformer()
    # print(net)
    net.eval()


    from thop import profile
    flops, params = profile(net, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

# net = Conformer()
# input = torch.randn(2, 200, 15, 15)
# output1, output2 = net(input)
# print(output1,output2)