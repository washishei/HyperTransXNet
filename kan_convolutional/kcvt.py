import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .KANConv2D import KKAN_Convolutional_Network, KKAN_Convolutional_Network_New
from .KANLinear import KANLinear
from .KANConv3D import effConvKAN3D

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
    def __init__(self, dim, mult = 2, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Conv2d(dim, dim * mult, 1),
            KKAN_Convolutional_Network(dim, 3, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            # KKAN_Convolutional_Network(dim * mult, 3, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias),
            # KKAN_Convolutional_Network(dim_in, 3, dim_in),
            nn.BatchNorm2d(dim_in),
            # KKAN_Convolutional_Network(dim_in, 3, dim_in),
            # nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
            # KKAN_Convolutional_Network(dim_out, 2, dim_out),
        )
    def forward(self, x):
        # residual = x
        x = self.net(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 4, dim_head = 16, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)
        # self.to_q = nn.Sequential(
        #     DepthWiseConv2d(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False),
        #     KKAN_Convolutional_Network(dim, 3, inner_dim))
        # self.to_kv = nn.Sequential(
        #     DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding=padding, stride=kv_proj_stride, bias=False),
        #     KKAN_Convolutional_Network(dim, 3, inner_dim*2))
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding=padding, stride=kv_proj_stride, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.BatchNorm2d(dim),
            KKAN_Convolutional_Network(dim, 3, dim),
            # nn.Conv2d(dim, dim, 1),
            nn.Dropout(dropout)
        )

        # self.to_h = nn.Sequential(
        #     # nn.Conv2d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=1,
        #     #           bias=True),
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, stride=1,
        #               bias=True),
        #     KKAN_Convolutional_Network(dim, 3, dim),
        # )
        #
        # self.to_w = nn.Sequential(
        #     # nn.Conv2d(dim, dim, kernel_size=(1, 3, 1), padding=(0, 1, 0), stride=1,
        #     #           bias=True),
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, stride=1,
        #               bias=True),
        #     KKAN_Convolutional_Network(dim, 3, dim),
        # )
        #
        # self.to_s = nn.Sequential(
        #     # nn.Conv2d(dim, dim, kernel_size=(1, 1, 3), padding=(0, 0, 1), stride=1,
        #     #           bias=True),
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, stride=1,
        #               bias=True),
        #     KKAN_Convolutional_Network(dim, 3, dim),
        # )
        #
        # self.to_out = nn.Sequential(
        #     nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0, stride=1,
        #               bias=True),
        #     KKAN_Convolutional_Network(dim, 3, dim),
        # )


    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)

        # x_h = self.to_h(x)
        # x_w = self.to_w(x)
        # x_s = self.to_s(x)
        # x_total = torch.cat([x_h, x_w, x_s], dim=1)
        # out = self.to_out(x_total)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64, mlp_mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            # print(x.shape)
        return x

class KCvT(nn.Module):
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
        s1_mlp_mult=1,  # stage 1 - feedforward expansion factor
        s2_emb_dim=64,  # stage 2 - (same as above)
        s2_emb_kernel=3,
        s2_emb_stride=2,
        s2_proj_kernel=3,
        s2_kv_proj_stride=2,
        s2_heads=1,
        s2_depth=1,
        s2_mlp_mult=1,
        s3_emb_dim=128,  # stage 3 - (same as above)
        s3_emb_kernel=3,
        s3_emb_stride=1,
        s3_proj_kernel=3,
        s3_kv_proj_stride=1,
        s3_heads=1,
        s3_depth=1,
        s3_mlp_mult=1,
        s4_emb_dim=128,  # stage 4 - (same as above)
        s4_emb_kernel=3,
        s4_emb_stride=2,
        s4_proj_kernel=3,
        s4_kv_proj_stride=2,
        s4_heads=1,
        s4_depth=1,
        s4_mlp_mult=1,
        dropout=0.
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = channels
        layers = []
        self.stem = nn.Sequential(
            effConvKAN3D(1, 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            # effConvKAN3D(4, 8, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )
        self.stem_project = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, stride=1, bias=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            # KKAN_Convolutional_Network(dim, 2, dim),
            # nn.ReLU(),
        )

        for prefix in ('s1','s2', 's3', 's4'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)

            layers.append(nn.Sequential(
                # nn.Conv2d(dim, config['emb_dim'], kernel_size = config['emb_kernel'], padding = (config['emb_kernel'] // 2), stride = config['emb_stride']),
                KKAN_Convolutional_Network(dim, 1, config['emb_dim']),
                LayerNorm(config['emb_dim']),
                Transformer(dim=config['emb_dim'], proj_kernel = config['proj_kernel'], kv_proj_stride = config['kv_proj_stride'], depth = config['depth'], heads = config['heads'], mlp_mult = config['mlp_mult'], dropout = dropout)
            ))

            dim = config['emb_dim']

        self.layers = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            KANLinear(dim, num_classes)
            # nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # x = self.stem(x)
        # print(x.shape)
        # B, C, S, H, W = x.shape
        # x = x.reshape(B, C*S, H, W)
        # print(x.shape)
        # x = self.stem_project(x)
        # print(x.shape)
        x = x.squeeze()
        # x = self.stem_project(x)
        return self.layers(x)

if __name__ == '__main__':
    img_size = 15
    x = torch.rand(2, 1, 10, img_size, img_size)
    model = KCvT(num_classes=16, channels=10,)
    y = model(x)
    # print(y.shape)

    # from thop import profile
    # flops, params = profile(model, (x,))
    #
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
    # repetitions = 100
    # total_time = 0
    # optimal_batch_size = 2
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #         starter.record()
    #         _ = model(x)
    #         ender.record()
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender) / 1000
    #         total_time += curr_time
    # Throughput = (repetitions * optimal_batch_size) / total_time
    # print("FinalThroughput:", Throughput)
    # print("The training time is: **********", total_time)



