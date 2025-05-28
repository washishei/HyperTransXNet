import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# from vmamba import *
# from eca_test import EfficientGlobalLocalizationAttention
from vit_pytorch.eca_test import *
from vit_pytorch.vision_mamba import *
# helper methods

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# classes

class PatchEmbed2D_up(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x.permute(0, 3, 1, 2)

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

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding=padding, stride=kv_proj_stride, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head = 64, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TiMN(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        channels,
        s1_emb_dim=64,  # stage 1 - dimension
        s1_emb_kernel=3,  # stage 1 - conv kernel
        s1_emb_stride=1,  # stage 1 - conv stride
        s1_proj_kernel=3,  # stage 1 - attention ds-conv kernel size
        s1_kv_proj_stride=1,  # stage 1 - attention key / value projection stride
        s1_heads=1,  # stage 1 - heads
        s1_depth=1,  # stage 1 - depth
        s1_mlp_mult=4,  # stage 1 - feedforward expansion factor
        s2_emb_dim=128,  # stage 2 - (same as above)
        s2_emb_kernel=3,
        s2_emb_stride=1,
        s2_proj_kernel=3,
        s2_kv_proj_stride=2,
        s2_heads=3,
        s2_depth=2,
        s2_mlp_mult=4,
        s3_emb_dim=256,  # stage 3 - (same as above)
        s3_emb_kernel=3,
        s3_emb_stride=1,
        s3_proj_kernel=3,
        s3_kv_proj_stride=1,
        s3_heads=4,
        s3_depth=6,
        s3_mlp_mult=4,
        s4_emb_dim=256,  # stage 3 - (same as above)
        s4_emb_kernel=3,
        s4_emb_stride=1,
        s4_proj_kernel=3,
        s4_kv_proj_stride=1,
        s4_heads=4,
        s4_depth=10,
        s4_mlp_mult=4,
        dropout=0.
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = channels
        layers = []
        self.stem = EfficientGlobalLocalizationAttention(channel=channels, kernel_size=3)

        for prefix in ('s1', 's2', 's3'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)

            layers.append(nn.Sequential(
                nn.Conv2d(dim, config['emb_dim'], kernel_size = config['emb_kernel'], padding = (config['emb_kernel'] // 2), stride = config['emb_stride']),
                LayerNorm(config['emb_dim']),
                Transformer(dim = config['emb_dim'], proj_kernel = config['proj_kernel'], kv_proj_stride = config['kv_proj_stride'], depth = config['depth'], heads = config['heads'], mlp_mult = config['mlp_mult'], dropout = dropout),
                Vim(
                    dim=config['emb_dim'],  # Dimension of the model
                    heads=config['heads'],  # Number of attention heads
                    dt_rank=32,  # Rank of the dynamic routing tensor
                    dim_inner=config['emb_dim'],  # Inner dimension of the model
                    d_state=config['emb_dim'],  # State dimension of the model
                    num_classes=1000,  # Number of output classes
                    image_size=15,  # Size of the input image
                    patch_size=3,  # Size of the image patch
                    channels=config['emb_dim'],  # Number of input channels
                    dropout=0.1,  # Dropout rate
                    depth=config['depth'],  # Depth of the model
                ),
                PatchEmbed2D_up(patch_size=4, in_chans=config['emb_dim'], embed_dim=config['emb_dim'], norm_layer=nn.LayerNorm)
            ))

            dim = config['emb_dim']
        # self.vamba = VSSLayer(
        #     dim=256,
        #     depth=2,
        #     d_state=16,  # 20240109
        #     drop=0.,
        #     attn_drop=0.,
        #     drop_path=0.1,
        #     norm_layer=nn.LayerNorm,
        #     downsample=None,
        #     use_checkpoint=False,
        # )
        # self.patch_embed = PatchEmbed2D(patch_size=4, in_chans=dim, embed_dim=dim,
        #                                 norm_layer=nn.LayerNorm)
        self.layers = nn.Sequential(
            *layers,
            # nn.AdaptiveAvgPool2d(1),
            # Rearrange('... () () -> ...'),
            # nn.Linear(dim, num_classes)
        )

        self.final_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = x.squeeze()
        x = self.stem(x)
        x = self.layers(x)
        # print(x.shape)
        # print(x.shape)
        # x = self.patch_embed(x)
        # print(x.shape)
        # x = self.vamba(x).permute(0, 3, 1, 2)
        # print(x.shape)
        x = self.final_layer(x)
        return x

if __name__ == '__main__':
    img_size = 15
    x = torch.rand(2, 1, 256, img_size, img_size)
    # device = torch.device("gpu"),
    # x = x.to(device)
    x = x.cuda()
    model = TiMN(num_classes=16, channels=256,)
    model = model.cuda()

    from thop import profile
    flops, params = profile(model, (x,))

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
    print("The training time is: **********", total_time)
