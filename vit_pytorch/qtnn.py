import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from .core.quaternion_layers import *
# from core.quaternion_layers import *

from QCNN.generateq import *


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'ViP_S': _cfg(crop_pct=0.9),
    'ViP_M': _cfg(crop_pct=0.9),
    'ViP_L': _cfg(crop_pct=0.875),
}


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        #print(x.shape)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = QuaternionConv(dim, dim, 3, padding=1, stride=1)
        # self.conv_spatial = QuaternionConv(dim, dim, 3, stride=1, padding=3, dilatation=3)
        self.conv_spatial = QuaternionConv(2 * dim, dim, 3, stride=1, padding=1)
        self.conv1 = QuaternionConv(dim, dim, 1, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(torch.cat([attn, attn], 1))
        # attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        attn = u * attn
        return attn + u

class SpatialAttention(nn.Module):
    def __init__(self, d_model, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim

        self.proj_1 = nn.Conv2d(d_model, d_model, 1, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut

        return x


class PermutatorBlock(nn.Module):

    def __init__(self, dim, segment_dim, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=SpatialAttention):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        # self.norm1 = QuaternionBatchNorm2d(dim, gamma_init=1.0, beta_param=True)
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = mlp_fn(dim, segment_dim=segment_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.norm2 = norm_layer(dim)
        # self.norm2 = QuaternionBatchNorm2d(dim, gamma_init=1.0, beta_param=True)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        # print('input shape:', x.shape)
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=16, patch_size=2, in_chans=3, embed_dim=256):
        super().__init__()
        #### for GRSS (10, 9) pu(15,9)step(2,1) XA(46,30) for in(20,16)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2,
                                   padding=1)
        #### for IN
        self.proj2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x):
        x = self.proj(x)  # B, C, H, W
        x = self.proj2(x)
        # B, C, H, W = x.shape
        # x = x.reshape(B, H, W, C)
        return x


class Downsample(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=patch_size, stride=stride,
                                   padding=1)

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        # x = x.permute(0, 2, 3, 1)
        return x


def basic_blocks(dim, index, layers, segment_dim, mlp_ratio=3., qkv_bias=True, qk_scale=None, \
                 attn_drop=0, drop_path_rate=0., skip_lam=1.0, mlp_fn=SpatialAttention, **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PermutatorBlock(dim, segment_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, \
                                      attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))

    blocks = nn.Sequential(*blocks)

    return blocks


class QTNN(nn.Module):
    """ Vision Permutator
    """

    def __init__(self, layers, img_size=15, patch_size=3, in_chans=3, num_classes=1000,
                 embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0,
                 qkv_bias=False, qk_scale=None, stride=[2, 1, 1, 1], drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, mlp_fn=SpatialAttention):
        super().__init__()
        self.num_classes = num_classes
        self.pre_press = generateq(channel=in_chans, k_size=1)
        h_inchans = 4

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=h_inchans,
                                      embed_dim=embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, segment_dim[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer, skip_lam=skip_lam,
                                 mlp_fn=mlp_fn)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size, stride=stride[i]))

        self.network = nn.ModuleList(network)

        self.norm = norm_layer(embed_dims[-1])

        # Classifier head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        # x = rearrange(x, 'b (p1 p2) w -> b p1 p2 w', p1=8, p2=8)
        # B,C,H,W-> B,H,W,C
        # x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        B, C, H, W = x.shape
        # print(x.shape)
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = x.squeeze()
        x = self.pre_press(x)
        x = self.forward_embeddings(x)
        # B, H, W, C -> B, N, C
        x = self.forward_tokens(x)
        # print(x.shape)
        x = self.norm(x)
        return self.head(x.mean(1))


# @register_model
# def vip_s14(pretrained=False, **kwargs):
#     layers = [4, 3, 8, 3]
#     transitions = [False, False, False, False]
#     segment_dim = [16, 16, 16, 16]
#     mlp_ratios = [3, 3, 3, 3]
#     embed_dims = [384, 384, 384, 384]
#     model = VisionPermutator(layers, embed_dims=embed_dims, patch_size=14, transitions=transitions,
#                              segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=WeightedPermuteMLP, **kwargs)
#     model.default_cfg = default_cfgs['ViP_S']
#     return model
#
#
# @register_model
# def vip_s7(pretrained=False, **kwargs):
#     layers = [4, 3, 8, 3]
#     transitions = [True, False, False, False]
#     segment_dim = [32, 16, 16, 16]
#     mlp_ratios = [3, 3, 3, 3]
#     embed_dims = [192, 384, 384, 384]
#     model = VisionPermutator(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
#                              segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=WeightedPermuteMLP, **kwargs)
#     model.default_cfg = default_cfgs['ViP_S']
#     return model
#
#
# @register_model
# def vip_m7(pretrained=False, **kwargs):
#     # 55534632
#     layers = [4, 3, 14, 3]
#     transitions = [False, True, False, False]
#     segment_dim = [32, 32, 16, 16]
#     mlp_ratios = [3, 3, 3, 3]
#     embed_dims = [256, 256, 512, 512]
#     model = VisionPermutator(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
#                              segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=WeightedPermuteMLP, **kwargs)
#     model.default_cfg = default_cfgs['ViP_M']
#     return model
#
#
# @register_model
# def vip_l7(pretrained=False, **kwargs):
#     layers = [8, 8, 16, 4]
#     transitions = [True, False, False, False]
#     segment_dim = [32, 16, 16, 16]
#     mlp_ratios = [3, 3, 3, 3]
#     embed_dims = [256, 512, 512, 512]
#     model = VisionPermutator(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
#                              segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=WeightedPermuteMLP, **kwargs)
#     model.default_cfg = default_cfgs['ViP_L']
#     return model
