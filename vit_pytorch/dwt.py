import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import *
import joblib
import collections
import spectral
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = \
            (out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

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


class WeightedPermuteMLP(nn.Module):
    def __init__(self, dim, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim

        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        S = C // self.segment_dim
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H * S)
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W * S)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class ConvPermute(nn.Module):
    def __init__(self, dim, segment_dim=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        self.mlp_c_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.mlp_c_2a = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=1, groups=dim, bias=qkv_bias)
        self.mlp_c_2b = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=1, padding=(0, 2), dilation=1, groups=dim, bias=qkv_bias)

        self.mlp_h_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.mlp_h_2a = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=1, padding=(1, 0), dilation=1, groups=dim,
                                  bias=qkv_bias)
        self.mlp_h_2b = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=1, padding=(2, 0), dilation=1, groups=dim,
                                  bias=qkv_bias)

        self.mlp_w = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        )

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        #self.proj = nn.Linear(dim, dim)
        self.proj = nn.Conv2d(dim*3, dim, kernel_size=1, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x1 = x.reshape(B, H, W, C).permute(0, 3, 1, 2).reshape(B, C, H, W)

        h1 = self.mlp_c_1(x1)
        h_a = self.mlp_c_2a(h1)
        h_b = self.mlp_c_2b(h_a)
        # h = h_a + h_b
       # h = h.reshape(B, C, H, W).permute(0, 2, 3, 1).reshape(B, H, W, C)

        w1 = self.mlp_h_1(x1)
        w_a = self.mlp_h_2a(w1)
        w_b = self.mlp_h_2b(w_a)
        # w = w_a + w_b
        # w = w.reshape(B, C, H, W).permute(0, 2, 3, 1).reshape(B, H, W, C)

        c = self.mlp_w(x1)

        # a = (h + w + c).flatten(2).mean(2)
        # a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2).permute(0, 1, 4, 2, 3)

        x = [h_b, w_b, c]
        x = torch.cat(x, 1)

        # x = x.reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(B, H, W, C)
        return x

class PermutatorBlock(nn.Module):

    def __init__(self, dim, segment_dim, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=ConvPermute):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, segment_dim=segment_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=15, patch_size=3, in_chans=3, embed_dim=256):
        super().__init__()
        # in_chans = embed_dim
        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
        self.branch1x1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.branch3x3_1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.branch3x3_2a = nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = nn.Conv2d(embed_dim, embed_dim, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = nn.Conv2d(embed_dim, embed_dim, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1))
        self.output = nn.Conv2d(256*6, embed_dim, kernel_size=1, stride=1)

        self.out = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.conv(x)
        u = x.clone()
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl)
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]

        outputs = torch.cat(outputs, 1)
        outputs = self.output(outputs)
        outputs = u * outputs
        outputs = self.out(outputs)
        return outputs


class Downsample(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x


def basic_blocks(dim, index, layers, segment_dim, mlp_ratio=3., qkv_bias=True, qk_scale=None, \
                 attn_drop=0, drop_path_rate=0., skip_lam=1.0, mlp_fn=WeightedPermuteMLP, **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PermutatorBlock(dim, segment_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, \
                                      attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))

    blocks = nn.Sequential(*blocks)

    return blocks


class DWT(nn.Module):
    """ Vision Permutator
    """

    def __init__(self, layers, img_size=15, patch_size=3, in_chans=3, num_classes=1000,
                 embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0,
                 qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, mlp_fn=ConvPermute):

        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
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
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size))

        self.network = nn.ModuleList(network)

        self.norm = nn.Sequential(
            norm_layer(embed_dims[-1]),
            norm_layer(embed_dims[-1]))
        self.mlp_x = nn.Linear(embed_dims[-1] *4, 256, bias=True)
        self.norm_f = nn.Sequential(
            norm_layer(256),
            norm_layer(256))

        # Classifier head
        #self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Linear(256, num_classes) if num_classes > 0 else nn.Identity()
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
        # B,C,H,W-> B,H,W,C
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        local_features =[]
        for idx, block in enumerate(self.network):
            x = block(x)
            local_features.append(x)
        x = torch.cat(local_features, 3)
        x = self.mlp_x(x)
        # x = residual + x
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = x.squeeze()
        x = self.forward_embeddings(x)
        B, H, W, C = x.shape
        x1 = x
        x1 = x1.reshape(B, -1, C)
        # x1 = self.mlp_x(x1)
        # x1 = self.norm_f(x1)
        # B, H, W, C -> B, N, C
        x = self.forward_tokens(x)
        x = self.norm_f(x)
        x = x + x1
        return self.head(x.mean(1))
from PIL import Image

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
    height, width = 256, 256
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

# layers = [4, 3, 8, 3]
# transitions = [False, False, False, False]
# segment_dim = [8, 8, 4, 4]
# mlp_ratios = [3, 3, 3, 3]
# embed_dims = [256, 256, 512, 512]
# net = DWT(layers, img_size=145, in_chans=3, num_classes=10, embed_dims=embed_dims, patch_size=3, transitions=transitions,
#         segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=ConvPermute,)
# inputs = cv2.imread("E:/transformer/in/input.jpg")
# inputs = np.asarray(inputs, dtype="float32")
# inputs = np.asarray(np.copy(inputs), dtype="float32")
# print(inputs.shape)
# inputs = np.copy(inputs)
# print(inputs.shape)
# inputs = inputs.transpose(3, 1, 2)
# inputs = torch.from_numpy(inputs)
# output = net(inputs)
#
# print(net.network[0][-1].mlp)
# hook_ref_1 = SaveConvFeatures(net.network[0][-1])
# hook_ref_2 = SaveConvFeatures(net.network[1][-1])
# hook_ref_3 = SaveConvFeatures(net.network[3][-1])
# hook_ref_4 = SaveConvFeatures(net.network[4][-1])
# conv_features = list()
# conv_features.append(hook_ref_1.features)
# conv_features.append(hook_ref_2)
# conv_features.append(hook_ref_3)
# hook_ref_1.remove()
# hook_ref_2.remove()
# hook_ref_3.remove()
# show_feature_map(inputs, conv_features)
# print(net.network[2])


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
