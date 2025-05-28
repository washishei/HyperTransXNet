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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
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
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
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
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
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

        #x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

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
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x.transpose(1, 2).reshape(B, C, H, W)
        #x_r = x.transpose(1, 2).reshape(B, C, H, W)
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
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

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
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(dim * 2, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
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
        self.act = nn.GELU()

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix * dim, 1)
        )

    def forward(self, x, y):
        _, _, H, W = x.shape
        y_r = self.expand_block(y, H, W)
        y_r = self.key_embed(y_r)
        qk = torch.cat([y_r, x], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        #w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, qk_hh, qk_ww)

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
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # self.local_conv = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
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
        self.act = nn.GELU()

        self.mlp = Mlp(in_features=dim*2, hidden_features=dim, out_features=dim, act_layer=nn.GELU, drop=drop)

        self.embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU()
        )

        self.conv1x1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU()
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
        self.mlp = Mlp(in_features=dim, hidden_features=dim, out_features=number, act_layer=nn.GELU, drop=drop)
        self.conv1x1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU()
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
        self.act = nn.GELU()

        self.mlp = Mlp(in_features=dim*2, hidden_features=dim, out_features=dim, act_layer=nn.GELU, drop=drop)

        self.embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
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

class Conformer(nn.Module):

    def __init__(self, patch_size=16, in_chans=200, num_classes=1000, base_channel=64, channel_ratio=6, num_med_block=0,
                 embed_dim=384, depth=6, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
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
        # self.is_flatten = flatten
        # self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(256, num_classes)
        )

        ### final classifier
        # self.final_trans_pool = nn.Linear(embed_dim * 2, embed_dim)
        # self.final_trans_norm = nn.LayerNorm(embed_dim)
        self.final_cls_head = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(1024, num_classes)
        )

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        # self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=3, stride=2, padding=1, bias=False)  # 1 / 2 [112, 112]
        # self.bn1 = nn.BatchNorm2d(64)
        # self.act1 = nn.ReLU(inplace=True)
        self.conv1 = SSConv(in_chans, 64, kernel_size=5)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        # stage_1_channel = int(base_channel * channel_ratio)
        # trans_dw_stride = patch_size // 4
        self.layer1 = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU()),
            # ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))
        # self,conv_1 = SSConv(128, embed_dim, kernel_size=5)
        self.trans_patch_conv_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer_1 = LayerNorm(64)
        self.trans_1 = Block(dim=64, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )
        # 1 stage combine conv and transformer
        self.trans_conv_1 = CotLayer(64, kernel_size=3, dw_stride=1)
        self.conv_trans_1 = TranCotLayer(64, dw_stride=1)

        # 2 stage
        self.layer2 = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(128)),
            ('relu', nn.ReLU()),
            # ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))
        # self.conv_2 = SSConv(embed_dim, embed_dim, kernel_size=3)
        self.trans_patch_conv_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.layer_2 = LayerNorm(128)
        self.trans_2 = Block(dim=128, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )
        # 2 stage combine conv and transformer
        self.trans_conv_2 = CotLayer(128, kernel_size=3, dw_stride=1)
        self.conv_trans_2 = TranCotLayer(128, dw_stride=1)

        # 3 stage
        # self.conv_3 = ConvBlock(inplanes=embed_dim, outplanes=embed_dim, res_conv=True, stride=1)
        self.layer3 = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(256)),
            ('relu', nn.ReLU()),
            # ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))
        # self.conv_3 = SSConv(embed_dim, embed_dim, kernel_size=3)
        self.trans_patch_conv_3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.layer_3 = LayerNorm(256)
        self.trans_3 = Block(dim=256, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )
        # 3 stage combine conv and transformer
        self.trans_conv_3 = CotLayer(256, kernel_size=3, dw_stride=1)
        self.conv_trans_3 = TranCotLayer(256, dw_stride=1)

        ## 4 stage
        # self.layer4 = nn.Sequential(collections.OrderedDict([
        #     ('conv', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)),
        #     ('bn', nn.BatchNorm2d(512)),
        #     ('relu', nn.ReLU()),
        #     # ('avgpool', nn.AvgPool2d(kernel_size=4))
        #     ('glbpool', nn.AdaptiveAvgPool2d(1))
        # ]))
        # # self.conv_4 = ConvBlock(inplanes=embed_dim, outplanes=embed_dim, res_conv=True, stride=1)
        # # self.conv_4 = SSConv(embed_dim, embed_dim, kernel_size=3)
        # self.trans_patch_conv_4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.layer_4 = LayerNorm(512)
        # self.trans_4 = Block(dim=512, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
        #                      )
        # # 4 stage combine conv and transformer
        # self.trans_conv_4 = CotLayer(512, kernel_size=3, dw_stride=1)
        # self.conv_trans_4 = TranCotLayer(512, dw_stride=1)

        ## final stage
        self.conv_f = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(512)),
            ('relu', nn.ReLU()),
            # ('avgpool', nn.AvgPool2d(kernel_size=4))
            ('glbpool', nn.AdaptiveAvgPool2d(1))
        ]))
        # self.conv_f = ConvBlock(inplanes=embed_dim, outplanes=embed_dim, res_conv=True, stride=1)
        self.trans_patch_conv_f = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.layer_f = LayerNorm(512)
        self.trans_f = Block(dim=512, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        # self.trans_conv_3 = SimCotLayer(embed_dim, kernel_size=3, dw_stride=1)
        # self.conv_trans_3 = SimTranCotLayer(embed_dim, dw_stride=1)
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
        x = x.squeeze(dim=1)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # 1 stage
        h = self.layer1(x)
        x_base = self.conv1(x)
        x_t = self.trans_patch_conv_1(x_base)
        # print(x_t.shape)
        x_t = self.layer_1(x_t).flatten(2).transpose(1, 2)
        # x_t = torch.cat([cls_tokens, x_t], dim=1)

        x_t = self.trans_1(x_t)

        # 1 combine
        # print(x.shape)
        # print(x_t.shape)
        h = self.trans_conv_1(h, x_t)
        x_t = self.conv_trans_1(x_t, h)

        # # 2 stage
        h = self.layer2(h)
        B, C, D = x_t.shape
        x_t = x_t.permute(0, 2, 1).reshape(B, D, 15, 15)
        x_t = self.trans_patch_conv_2(x_t)
        x_t = self.layer_2(x_t).flatten(2).transpose(1, 2)
        # x_t = torch.cat([cls_tokens, x_t[:, 1:]], dim=1)
        x_t = self.trans_2(x_t)

        # 2 combine
        h = self.trans_conv_2(h, x_t)
        x_t = self.conv_trans_2(x_t, h)

        # # 3 stage
        h = self.layer3(h)
        B, C, D = x_t.shape
        x_t = x_t.permute(0, 2, 1).reshape(B, D, 15, 15)
        x_t = self.trans_patch_conv_3(x_t)
        x_t = self.layer_3(x_t).flatten(2).transpose(1, 2)
        # x_t = torch.cat([cls_tokens, x_t[:, 1:]], dim=1)
        x_t = self.trans_3(x_t)

        # 3 combine
        h = self.trans_conv_3(h, x_t)
        x_t = self.conv_trans_3(x_t, h)

        # 4 stage
        # x = self.conv_4(x, return_x_2=False)
        # x_raw = x
        # x_t = torch.cat([cls_tokens, x_t], dim=1)
        # x_t = self.trans_4(x_t)

        # 4 combine
        # x = self.trans_conv_4(x, x_t)
        # x_t = self.conv_trans_4(x_t[:, 1:], x_raw)
        h_o = h
        # conv classification
        # if (self.is_flatten): h = self.flatten(h)
        # print(h.shape)
        conv_cls = self.classifier(h)

        # trans classification
        B, C, D = x_t.shape
        x_t = x_t.permute(0, 2, 1).reshape(B, D, 15, 15)
        tran_cls = self.trans_cls_head(x_t)

        ## final output
        B, C, D, W = h_o.shape
        x_t = self.trans_patch_conv_f(x_t)
        x_t = self.layer_f(x_t).flatten(2).transpose(1, 2)
        # x_t = torch.cat([cls_tokens, x_t[:, 1:]], dim=1)
        x_t = self.trans_f(x_t)
        B, C, D = x_t.shape
        x_t = x_t.permute(0, 2, 1).reshape(B, D, 15, 15)
        x_t =self.pool(x_t)
        h_o = self.conv_f(h_o)
        output = torch.cat([h_o, x_t], dim=1)
        final_cls = self.final_cls_head(output)

        return conv_cls, tran_cls, final_cls
from PIL import Image
import matplotlib.pyplot as plt
def show_feature_map(img_src, conv_feature):
    '''可视化卷积层特征图输出
    img_src:源图像文件路径
    conv_feature:得到的卷积输出,[b, c, h, w]
    '''
    #img = Image.open(img_file).convert('RGB')
    img = img_src[0]
    img = img.squeeze(0)
    bands = (55, 41, 12)
    img = img.to('cpu')
    img = img.numpy()
    img = np.transpose(img, [1, 2, 0])
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')
    #height, width = 128, 128
    height, width = 128, 128
    for i in range(3):
        conv_features = conv_feature[i]
        heat = conv_features[0].squeeze(0)  # 降维操作,尺寸变为(2048,7,7)
        heat_mean = torch.mean(heat, dim=0)  # 对各卷积层(2048)求平均值,尺寸变为(7,7)
        heat_mean = heat_mean.to('cpu')
        heatmap = heat_mean.numpy()  # 转换为numpy数组
        heatmap /= np.max(heatmap)  # minmax归一化处理
        heatmap = cv2.resize(heatmap, (height, width))  # 变换heatmap图像尺寸,使之与原图匹配,方便后续可视化
        heatmap = np.uint8(255 * heatmap)  # 像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 颜色变换
        plt.imshow(heatmap)
        axis('off')
        plt.show()
        # heatmap = np.array(Image.fromarray(heatmap).convert('L'))
        # rgb = np.transpose(rgb, [1,2,0])
        point_color = (0, 0, 255)
        superimg = heatmap * 0.4 + np.array(rgb)[:, :, ::-1]  # 图像叠加，注意翻转通道，cv用的是bgr
        #cv2.circle(superimg, (64, 64), 10, point_color, 0)
        cv2.circle(superimg, (128, 128), 10, point_color, 0)
        cv2.imwrite('./superimg_layer{}.jpg'.format(str(i+1)), superimg)  # 保存结果
        # 可视化叠加至源图像的结果
        img_ = np.array(Image.open('./superimg_layer{}.jpg'.format((str(i+1)))).convert('RGB'))
        plt.imshow(img_)
        axis('off')
        plt.show()


class SaveConvFeatures():

    def __init__(self, m):  # module to hook
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.data

    def remove(self):
        self.hook.remove()

if __name__ == '__main__':
    img_size = 15
    x = torch.rand(2, 1, 200, img_size, img_size)
    net = Conformer()
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