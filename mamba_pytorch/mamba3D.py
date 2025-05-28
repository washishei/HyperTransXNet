import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.cuda.amp import autocast
import collections
import torch.utils.checkpoint as checkpoint
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from torch import einsum
from einops import rearrange, repeat
from timm.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.io as sio
from torch.utils.data import DataLoader, Dataset
import random

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
device = "cuda:0"
# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class PatchEmbed2D(nn.Module):
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
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = harm3x3(in_chans, embed_dim)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

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

class MambaLayer(nn.Module):
    def __init__(self, in_chans,  dim, h):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.ones(3))
        self.patch_embed_1 = nn.Sequential(
            Harm3d(in_chans, in_chans, kernel_size=3, padding=1),
            # nn.BatchNorm3d(in_chans),
            nn.GELU()
        )
        self.patch_embed_2 = nn.Sequential(
            Harm3d(in_chans, in_chans, kernel_size=3, padding=1),
            # nn.BatchNorm3d(in_chans),
            nn.GELU()
        )

        self.patch_embed_3 = nn.Sequential(
            Harm3d(in_chans, in_chans, kernel_size=3, padding=1),
            # nn.BatchNorm3d(in_chans),
            nn.GELU()
        )
        self.conv_mamba = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1),
            # nn.SiLU(),
            nn.SiLU()
        )
        self.conv_final = nn.Sequential(
            nn.Conv2d(in_chans * 2, in_chans, kernel_size=1, stride=1)
        )
        self.mamba_c = VSSLayer(
            dim=in_chans * in_chans,
            depth=2,
        )
        self.mamba_l = VSSLayer(
            dim=h*h,
            depth=2,
        )
        self.h_ln = LayerNorm(in_chans)

    # @autocast(enabled=False)
    def forward(self, x):
        # residual = x
        B, C, S, H, W = x.shape
        x_c = self.patch_embed_1(x)
        x_c = x_c.reshape(B, C*S, H, W).permute(0, 2, 3, 1)
        x_mamba_c = self.mamba_c(x_c)
        x_mamba_c = x_mamba_c.reshape(B, H, W, C*S).permute(0, 3, 1, 2).reshape(B, C, S, H, W)
        # print("x_mamba_c's shape is", x_mamba_c.shape)
        x_l = self.patch_embed_2(x)
        x_l = x_l.reshape(B, C, S, H*W)
        x_mamba_l = self.mamba_l(x_l)
        x_mamba_l = x_mamba_l.reshape(B, C, S, H, W)
        x_residual = self.patch_embed_3(x)
        # print("x_mamba_l's shape is", x_mamba_l.shape)
        out = self.gamma[0] * x_residual + self.gamma[1] * x_mamba_l + self.gamma[2] * x_mamba_c
        out = self.h_ln(out)
        return out

def dct_filters_3d(k=3, groups=1, expand_dim=1, level=None, DC=True, l1_norm=True):
    if level is None:
        nf = k ** 3 - int(not DC)
    else:
        if level <= k:
            nf = (level * (level + 1) * (level + 2)) // 6 - int(not DC)
        else:
            r = 3 * k - 3 - level
            nf = k ** 3 - r * (r + 1) * (r + 2) // 6 - int(not DC)

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
    def __init__(self, ni, no, kernel_size=3, stride=1, padding=0, bias=True, dilation=1, use_bn=False, level=None,
                 DC=True, groups=1):
        super(Harm3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dct = nn.Parameter(
            dct_filters_3d(k=kernel_size, groups=ni if use_bn else 1, expand_dim=1 if use_bn else 0, level=level,
                           DC=DC), requires_grad=False)

        nf = self.dct.shape[0] // ni if use_bn else self.dct.shape[1]
        if use_bn:
            self.bn = nn.BatchNorm3d(ni * nf, affine=False)
            self.weight = nn.Parameter(
                nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups * nf, 1, 1, 1), mode='fan_out',
                                        nonlinearity='relu'))
        else:
            self.weight = nn.Parameter(
                nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups, nf, 1, 1, 1), mode='fan_out',
                                        nonlinearity='relu'))
        self.bias = nn.Parameter(nn.init.zeros_(torch.Tensor(no))) if bias else None

    def forward(self, x):
        if not hasattr(self, 'bn'):
            filt = torch.sum(self.weight * self.dct, dim=2)
            x = F.conv3d(x, filt, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation,
                         groups=self.groups)
            return x
        else:
            x = F.conv3d(x, self.dct, stride=self.stride, padding=self.padding, dilation=self.dilation,
                         groups=x.size(1))
            x = self.bn(x)
            x = F.conv3d(x, self.weight, bias=self.bias, padding=0, groups=self.groups)
            return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Harm3d(in_features, hidden_features, kernel_size=3, padding=1)
        self.act = act_layer()
        self.fc2 = Harm3d(hidden_features, out_features, kernel_size=3, padding=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class MambaLayer3D(nn.Module):
    def __init__(self, in_chans, dim, h):
        super(MambaLayer3D, self).__init__()
        self.mamba3d = MambaLayer(in_chans, dim, h)
        self.norm = nn.InstanceNorm3d(in_chans)
        self.mlp = Mlp(in_chans, 256,)
        self.mlp_norm = nn.InstanceNorm3d(in_chans)


    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba3d(x)
        x = residual + x
        x = self.mlp(self.mlp_norm(x)) + x
        return x


# class VSSLayer(nn.Module):
#     def __init__(self, dim, depth, kernel_size=3, padding=1):
#         super(VSSLayer, self).__init__()
#         layers = []
#         for _ in range(depth):
#             layers.append(Harm3d(dim, dim, kernel_size=kernel_size, padding=padding))
#             layers.append(nn.ReLU())
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.layers(x)



# Define a function to calculate and save the Spearman correlation heatmap for HSI data
def calculate_and_save_spearman_heatmap(hsi_data, output_path="heatmap_output", filename="spearman_heatmap.png", max_samples=5000):
    """
    Adaptive function to calculate and save the Spearman correlation heatmap for HSI data.

    Parameters:
    - hsi_data (numpy.ndarray): HSI data in the shape (height, width, bands) or with more dimensions.
    - output_path (str): Directory path to save the heatmap image.
    - filename (str): Name of the output image file.
    - max_samples (int): Maximum number of samples for correlation calculation, used to limit computation on large data.
    """
    # Ensure hsi_data is 2D (samples, bands) by reshaping if needed
    if len(hsi_data.shape) > 2:
        hsi_data_flattened = hsi_data.reshape(-1, hsi_data.shape[-1])
    else:
        hsi_data_flattened = hsi_data

    # Determine whether to perform sampling based on the number of samples
    num_samples = hsi_data_flattened.shape[0]
    if num_samples > max_samples:
        # Randomly sample a subset of the data for large datasets
        sample_indices = random.sample(range(num_samples), max_samples)
        sampled_data = hsi_data_flattened[sample_indices, :]
    else:
        # Use full data if the sample size is within the limit
        sampled_data = hsi_data_flattened

    # Calculate the Spearman correlation matrix
    spearman_corr = pd.DataFrame(sampled_data).corr(method='spearman')

    # Plot and save the Spearman correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(spearman_corr, cmap='coolwarm', center=0, square=True, cbar_kws={'label': 'Spearman Correlation'})
    plt.title("Spearman Correlation Heatmap")
    plt.xlabel("Spectral Bands")
    plt.ylabel("Spectral Bands")
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

    return spearman_corr

# Example of calling the function
# calculate_and_save_spearman_heatmap(ksc_hsi_data, sample_size=1000, filename="ksc_spearman_heatmap_sampled.png")



class TMamba3DD(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1, input_size: int = 15):
        super().__init__()
        self.model_name = "TMamba3DD"
        self.classes = num_classes
        self.channeles = in_channels
        p1 = self.channeles // 2
        p2 = p1 // 2
        p3 = p2 // 2
        self.patch_size = input_size

        self.gammb = nn.Parameter(torch.ones(2))

        # Layers definitions
        self.hconv1 = Harm3d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.relu1 = nn.GELU()
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 1))

        self.hconv2 = Harm3d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.relu2 = nn.GELU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1))

        self.hconv3 = Harm3d(16, p3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(p3)
        self.relu3 = nn.GELU()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1))

        self.mamba_layer_1 = nn.Sequential(
            MambaLayer3D(p3, 128, 15),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        self.mamba_layer_2 = nn.Sequential(
            MambaLayer3D(p3, 128, 7),
        )

        self.mamba_layer_3 = nn.Sequential(
            MambaLayer3D(p3, 128, 7),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        self.mamba_layer_4 = nn.Sequential(
            MambaLayer3D(p3, 128, 3),
        )

        self.pool_f_3d = nn.Sequential(
            Harm3d(p3, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.pool_f_1d = nn.Sequential(
            Harm3d(p3, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.features_size = self._get_sizes()

        self.fc_1 = nn.Linear(512, 256)

        # Fully connected layer expecting (256) inputs after AdaptiveAvgPool3d
        self.fc = nn.Linear(256, num_classes)




    def _get_sizes(self):
        x = torch.zeros((1, 1, self.channeles, self.patch_size, self.patch_size))
        x = self.hconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
 
        x = self.hconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
 
        x = self.hconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
 
        b, c, s, h, w = x.size()
        size0 = c * s * w * h
        return size0

    def forward(self, x):
#     
#        save_path = "features/"
#        if not os.path.exists(save_path):
#            os.makedirs(save_path)
        x = self.hconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
#        np.save("features/layer1_output.npy", x.detach().cpu().numpy())  # Save Layer 1 output

      
        x = self.hconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
 
        x = self.hconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
#        np.save("features/layer3_output.npy", x.detach().cpu().numpy())  # Save Layer 3 output

        residual = x
        residual = self.pool_f_1d(residual)
        residual = residual.view(residual.size(0), -1)
        
        x = self.mamba_layer_1(x)
        x = self.mamba_layer_2(x)
        x = self.mamba_layer_3(x)
        x = self.mamba_layer_4(x)

        x = self.pool_f_3d(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256)
        out = self.gammb[0] * residual + self.gammb[1] * x

        out = self.fc(out)
        return out


def main():
    input_value = np.random.randn(2, 1, 200, 15, 15)
    input_value = torch.from_numpy(input_value).float().cuda()
    print(input_value.dtype)
    model = TMamba3DD(200, 2).cuda()
    model.train()
    out = model(input_value)
    from thop import profile
    flops, params = profile(model, (input_value,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    print(out.shape)


if __name__ == '__main__':
    main()
