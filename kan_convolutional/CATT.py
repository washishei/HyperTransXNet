import math
import torch
from torch import nn, einsum, autograd
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        d,
        H,
        T,
        bias=True,
        dropout=0.,
    ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        assert d % H == 0

        # key, query, value projections for all heads, but in a batch
        # output is 3X the dimension because it includes key, query and value
        self.c_attn = nn.Linear(d, 3*d, bias=bias)

        # projection of concatenated attention head outputs
        self.c_proj = nn.Linear(d, d, bias=bias)

        # dropout modules
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.H = H
        self.d = d

        # causal mask to ensure that attention is only applied to
        # the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(T, T))
                                    .view(1, 1, T, T))

    def forward(self, x):
        B, T, _ = x.size() # batch size, sequence length, embedding dimensionality
        # print(x.shape)

        # compute query, key, and value vectors for all heads in batch
        # split the output into separate query, key, and value tensors
        q, k, v = self.c_attn(x).split(self.d, dim=2) # [B, T, d]
        # reshape tensor into sequences of smaller token vectors for each head
        k = k.view(B, T, self.H, self.d // self.H).transpose(1, 2) # [B, H, T, d // H]
        q = q.view(B, T, self.H, self.d // self.H).transpose(1, 2)
        v = v.view(B, T, self.H, self.d // self.H).transpose(1, 2)

        # compute the attention matrix, perform masking, and apply dropout
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # [B, H, T, T]
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # compute output vectors for each token
        y = att @ v # [B, H, T, d // H]

        # concatenate outputs from each attention head and linearly project
        y = y.transpose(1, 2).contiguous().view(B, T, self.d)
        y = self.resid_dropout(self.c_proj(y))
        return y

class CausalAttention(nn.Module):
    """Autoregresively masked, multihead self-attention layer.

    Autoregressive masking means that the current pixel can only attend to itself,
    pixels to the left, and pixels above. When mask_center=True, the current pixel does
    not attend to itself.

    This Module generalizes attention to use 2D convolutions instead of fully connected
    layers. As such, the input is expected to be 4D image tensors.
    """

    def __init__(
        self,
        in_channels,
        n_heads=1,
        embed_channels=None,
        out_channels=None,
        mask_center=False,
        extra_input_channels=0,
    ):
        """Initializes a new CausalAttention instance.

        Args:
            in_channels: Number of input channels.
            n_heads: Number of causal self-attention heads.
            embed_channels: Number of embedding channels. Defaults to in_channels.
            out_channels: Number of output channels. Defaults to in_channels.
            extra_input_channels: Extra input channels which are only used to compute
                the embeddings and not the attention weights since doing so may break
                the autoregressive property. For example, in [1] these channels include
                the original input image.
            mask_center: Whether to mask the center pixel of the attention matrices.
        """
        super().__init__()
        self._n_heads = n_heads
        self._embed_channels = embed_channels or in_channels
        self._out_channels = out_channels or in_channels
        self._mask_center = mask_center

        self._q = nn.Conv2d(
            in_channels=in_channels, out_channels=self._embed_channels, kernel_size=1
        )
        self._kv = nn.Conv2d(
            in_channels=in_channels + extra_input_channels,
            out_channels=self._embed_channels + self._out_channels,
            kernel_size=1,
        )
        # TODO(eugenhotaj): Should we only project if n_heads > 1?
        self._proj = nn.Conv2d(
            in_channels=self._out_channels,
            out_channels=self._out_channels,
            kernel_size=1,
        )

    def forward(self, x, extra_x=None):
        """Computes the forward pass.

        Args:
            x: The input used to compute both embeddings and attention weights.
            extra_x: Extra channels concatenated with 'x' only used to compute the
                embeddings. See the 'extra_input_channels' argument for more info.
        Returns:
            The result of the forward pass.
        """

        def _to_multihead(t):
            """Reshapes an (N, C, H, W) tensor into (N, n_heads, H * W, head_size)."""
            c = t.shape[1]
            t = t.view(n, self._n_heads, c // self._n_heads, -1)
            return t.transpose(2, 3)

        n, _, h, w = x.shape

        # Compute the query, key, and value.
        q = _to_multihead(self._q(x))
        if extra_x is not None:
            x = torch.cat((x, extra_x), dim=1)
        k, v = self._kv(x).split([self._embed_channels, self._out_channels], dim=1)
        k, v = _to_multihead(k), _to_multihead(v)

        # Compute the causal attention weights.
        mask = (
            _get_causal_mask(h * w, self._mask_center)
            .view(1, 1, h * w, h * w)
            .to(next(self.parameters()).device)
        )
        attn = (q @ k.transpose(2, 3)) / np.sqrt(k.shape[-1])
        attn = attn.masked_fill(mask == 0, -np.inf)
        # NOTE: When self._mask_center is True, the first row of the attention matrix
        # will be NaNs. We replace the NaNs with 0s here to prevent downstream issues.

        attn = F.softmax(attn, dim=-1).masked_fill(mask == 0, 0)

        # Attend to output for each head, stack, and project.
        out = (attn @ v).transpose(2, 3).contiguous().view(n, -1, h, w)
        return self._proj(out)
def _get_causal_mask(size, mask_center):
    """Generates causal masks for attention weights."""
    return torch.tril(torch.ones((size, size)), diagonal=-int(mask_center))

class CausalAttention(nn.Module):
    """Autoregresively masked, multihead self-attention layer.

    Autoregressive masking means that the current pixel can only attend to itself,
    pixels to the left, and pixels above. When mask_center=True, the current pixel does
    not attend to itself.

    This Module generalizes attention to use 2D convolutions instead of fully connected
    layers. As such, the input is expected to be 4D image tensors.
    """

    def __init__(
        self,
        in_channels,
        n_heads=1,
        embed_channels=None,
        out_channels=None,
        mask_center=False,
        extra_input_channels=0,
    ):
        """Initializes a new CausalAttention instance.

        Args:
            in_channels: Number of input channels.
            n_heads: Number of causal self-attention heads.
            embed_channels: Number of embedding channels. Defaults to in_channels.
            out_channels: Number of output channels. Defaults to in_channels.
            extra_input_channels: Extra input channels which are only used to compute
                the embeddings and not the attention weights since doing so may break
                the autoregressive property. For example, in [1] these channels include
                the original input image.
            mask_center: Whether to mask the center pixel of the attention matrices.
        """
        super().__init__()
        self._n_heads = n_heads
        self._embed_channels = embed_channels or in_channels
        self._out_channels = out_channels or in_channels
        self._mask_center = mask_center

        self._q = nn.Conv2d(
            in_channels=in_channels, out_channels=self._embed_channels, kernel_size=1
        )
        self._kv = nn.Conv2d(
            in_channels=in_channels + extra_input_channels,
            out_channels=self._embed_channels + self._out_channels,
            kernel_size=1,
        )
        # TODO(eugenhotaj): Should we only project if n_heads > 1?
        self._proj = nn.Conv2d(
            in_channels=self._out_channels,
            out_channels=self._out_channels,
            kernel_size=1,
        )

    def forward(self, x, extra_x=None):
        """Computes the forward pass.

        Args:
            x: The input used to compute both embeddings and attention weights.
            extra_x: Extra channels concatenated with 'x' only used to compute the
                embeddings. See the 'extra_input_channels' argument for more info.
        Returns:
            The result of the forward pass.
        """

        def _to_multihead(t):
            """Reshapes an (N, C, H, W) tensor into (N, n_heads, H * W, head_size)."""
            c = t.shape[1]
            t = t.view(n, self._n_heads, c // self._n_heads, -1)
            return t.transpose(2, 3)

        n, _, h, w = x.shape

        # Compute the query, key, and value.
        q = _to_multihead(self._q(x))
        if extra_x is not None:
            x = torch.cat((x, extra_x), dim=1)
        k, v = self._kv(x).split([self._embed_channels, self._out_channels], dim=1)
        k, v = _to_multihead(k), _to_multihead(v)

        # Compute the causal attention weights.
        mask = (
            _get_causal_mask(h * w, self._mask_center)
            .view(1, 1, h * w, h * w)
            .to(next(self.parameters()).device)
        )
        attn = (q @ k.transpose(2, 3)) / np.sqrt(k.shape[-1])
        attn = attn.masked_fill(mask == 0, -np.inf)
        # NOTE: When self._mask_center is True, the first row of the attention matrix
        # will be NaNs. We replace the NaNs with 0s here to prevent downstream issues.

        attn = F.softmax(attn, dim=-1).masked_fill(mask == 0, 0)

        # Attend to output for each head, stack, and project.
        out = (attn @ v).transpose(2, 3).contiguous().view(n, -1, h, w)
        return self._proj(out)


def _idx(i):
    return (slice(None), slice(None), slice(i, i + 1, 1), slice(None))
class _UnnormalizedLinearCausalAttention(autograd.Function):
    """Computes unnormalized causal attention using only O(N*C) memory."""

    @staticmethod
    def forward(ctx, Q, K, V):
        ctx.save_for_backward(Q, K, V)

        Vnew, S = torch.zeros_like(V), 0
        for i in range(V.shape[2]):
            S = S + K[_idx(i)].transpose(2, 3) @ V[_idx(i)]
            Vnew[_idx(i)] = Q[_idx(i)] @ S
        return Vnew

    @staticmethod
    def backward(ctx, G):
        Q, K, V = ctx.saved_tensors

        dQ, S = torch.zeros_like(Q), 0
        for i in range(V.shape[2]):
            S = S + K[_idx(i)].transpose(2, 3) @ V[_idx(i)]
            dQ[_idx(i)] = G[_idx(i)] @ S.transpose(2, 3)

        dK, dV, S = torch.zeros_like(K), torch.zeros_like(V), 0
        for i in range(V.shape[2] - 1, -1, -1):
            S = S + Q[_idx(i)].transpose(2, 3) @ G[_idx(i)]
            dV[_idx(i)] = K[_idx(i)] @ S
            dK[_idx(i)] = V[_idx(i)] @ S.transpose(2, 3)
        return dQ, dK, dV

class LinearCausalAttention(nn.Module):
    """Memory efficient implementation of CausalAttention as introduced in [2].

    NOTE: LinearCausalAttention is *much* slower than CausalAttention and should
    only be used if your model cannot fit in memory.

    This implementation only requires O(N) memory (instead of O(N^2)) for a
    sequence of N elements (e.g. an image with N pixels). To achieve this memory
    reduction, the implementation avoids storing the full attention matrix in
    memory and instead computes the output directly as Q @ (K @ V). However, this
    output cannot be vectorized and requires iterating over the sequence, which
    drastically slows down the computation.
    """

    def __init__(
        self,
        in_channels,
        feature_fn=lambda x: F.elu(x) + 1,
        n_heads=1,
        embed_channels=None,
        out_channels=None,
    ):
        """Initializes a new LinearCausalAttention instance.

        Args:
            in_channels: Number of input channels.
            feature_fn: A kernel feature map applied to the Query and Key activations.
                Defaults to lambda x: elu(x) + 1.
            n_heads: Number of causal self-attention heads.
            embed_channels: Number of embedding channels. Defaults to in_channels.
            out_channels: Number of output channels. Defaults to in_channels.
        """
        super().__init__()
        self._feature_fn = feature_fn
        self._n_heads = n_heads
        self._embed_channels = embed_channels or in_channels
        self._out_channels = out_channels or in_channels

        self._query = nn.Conv2d(
            in_channels=in_channels, out_channels=self._embed_channels, kernel_size=1
        )
        self._kv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self._embed_channels + self._out_channels,
            kernel_size=1,
        )
        self._numerator = _UnnormalizedLinearCausalAttention.apply

    def forward(self, x):
        def _to_multihead(t):
            """Reshapes an (N, C, H, W) tensor into (N, n_heads, H * W, head_size)."""
            c = t.shape[1]
            t = t.view(n, self._n_heads, c // self._n_heads, -1)
            return t.transpose(2, 3)

        n, _, h, w = x.shape

        # Compute the Query, Key, and Value.
        Q = _to_multihead(self._query(x))
        K, V = self._kv(x).split([self._embed_channels, self._out_channels], dim=1)
        K, V = _to_multihead(K), _to_multihead(V)

        # Compute the causal attention weights.
        Q, K = self._feature_fn(Q), self._feature_fn(K)
        den = 1 / (torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + 1e-10)
        num = self._numerator(Q, K, V)
        out = num * torch.unsqueeze(den, -1)
        return out.transpose(2, 3).contiguous().view(n, -1, h, w)

class CausalConv2d(nn.Conv2d):
    """A Conv2d layer masked to respect the autoregressive property.

    Autoregressive masking means that the computation of the current pixel only
    depends on itself, pixels to the left, and pixels above. When mask_center=True, the
    computation of the current pixel does not depend on itself.

    E.g. for a 3x3 kernel, the following masks are generated for each channel:
                          [[1 1 1],                     [[1 1 1],
        mask_center=False  [1 1 0],    mask_center=True  [1 0 0],
                           [0 0 0]]                      [0 0 0]
    In [1], they refer to the left masks as 'type A' and right as 'type B'.

    NOTE: This layer does *not* implement autoregressive channel masking.
    """

    def __init__(self, mask_center, *args, **kwargs):
        """Initializes a new CausalConv2d instance.

        Args:
            mask_center: Whether to mask the center pixel of the convolution filters.
        """
        super().__init__(*args, **kwargs)
        i, o, h, w = self.weight.shape
        mask = torch.zeros((i, o, h, w))
        mask.data[:, :, : h // 2, :] = 1
        mask.data[:, :, h // 2, : w // 2 + int(not mask_center)] = 1
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
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
    def __init__(self, dim, mult=4, dropout = 0.):
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
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 2, dim_head = 64, dropout = 0., input_size: int=15):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.input_size = input_size

        self.feature_size = input_size * input_size

        self.spatial_causal = CausalAttention(dim, self.heads, 256, 256)
        self.spatial_causal_t = CausalAttention(dim)
        self.spectral_causal = CausalSelfAttention(self.feature_size, self.heads, 256)
        self.gamma = nn.Parameter(torch.ones(2))

        self.to_out = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Dropout(dropout)
        )

    def _get_sizes(self):
        x = torch.zeros((1, self.dim, self.input_size, self.input_size))
        x = self.feature(x)
        w, h = x.size()
        size0 = w * h
        # print(size0)
        return size0

    def forward(self, x):
        b, n, h, w = x.shape
        # h_x = x.reshape(b, n, h*w).permute(0, 2, 1)
        attn_spatial = self.spatial_causal_t(x)
        s_x = x.reshape(b, n, h*w)
        attn_spectral = self.spectral_causal(s_x).reshape(b, n, h, w)
        attn = self.gamma[0] * attn_spectral + self.gamma[1] * attn_spatial

        return self.to_out(attn)

class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head = 64, mlp_mult = 4, dropout = 0., input_size: int=15):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout = dropout, input_size=input_size)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim):
        super().__init__()
        self.input_channels = in_chans
        self.proj = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3), padding=0, stride=(1, 1, 1)),
            nn.ReLU(),
        )
        self.feature_size = self._get_sizes()
        self.proj2d = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

    def _get_sizes(self):
        x = torch.zeros((1, 1, self.input_channels, 15, 15))
        x = self.proj(x)
        _, c, s, w, h = x.size()
        size0 = c * s
        # print(size0)
        return size0

    def forward(self, x):
        x = x.squeeze()
        x = self.proj2d(x)
        # b, c, s, h, w = x.size()
        # x = x.view(b, c * s, h, w)
        # x_out = self.proj2d(x)
        return x
class CAT(nn.Module):
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
        s2_heads=1,
        s2_depth=1,
        s2_mlp_mult=4,
        s3_emb_dim=256,  # stage 3 - (same as above)
        s3_emb_kernel=3,
        s3_emb_stride=1,
        s3_proj_kernel=3,
        s3_kv_proj_stride=1,
        s3_heads=1,
        s3_depth=1,
        s3_mlp_mult=4,
        s4_emb_dim=256,  # stage 3 - (same as above)
        s4_emb_kernel=3,
        s4_emb_stride=1,
        s4_proj_kernel=3,
        s4_kv_proj_stride=1,
        s4_heads=1,
        s4_depth=1,
        s4_mlp_mult=4,
        dropout=0.
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = channels
        layers = []

        self.gammc = nn.Parameter(torch.ones(2))

        self.patch_embed = PatchEmbed(in_chans=channels, embed_dim=s4_emb_dim)

        self.transformer_1 = nn.Sequential(
            nn.Conv2d(s4_emb_dim, s2_emb_dim, kernel_size=s1_emb_kernel, padding=s1_emb_kernel // 2, stride=s1_emb_stride),
            LayerNorm(s2_emb_dim),
            Transformer(dim=s2_emb_dim, proj_kernel=s1_proj_kernel, kv_proj_stride=s1_kv_proj_stride, depth=s1_depth, heads=s1_heads, mlp_mult=s1_mlp_mult, dropout=dropout, input_size=9)
        )

        self.transformer_2 = nn.Sequential(
            nn.Conv2d(s2_emb_dim, s2_emb_dim, kernel_size=s2_emb_kernel, padding=s2_emb_kernel // 2, stride=s2_emb_stride),
            LayerNorm(s2_emb_dim),
            Transformer(dim=s2_emb_dim, proj_kernel=s2_proj_kernel, kv_proj_stride=s2_kv_proj_stride, depth=s2_depth,
                        heads=s2_heads, mlp_mult=s2_mlp_mult, dropout=dropout, input_size=9),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.transformer_3 = nn.Sequential(
            nn.Conv2d(s2_emb_dim, s3_emb_dim, kernel_size=s3_emb_kernel, padding=s3_emb_kernel // 2, stride=s3_emb_stride),
            LayerNorm(s3_emb_dim),
            Transformer(dim=s3_emb_dim, proj_kernel=s3_proj_kernel, kv_proj_stride=s3_kv_proj_stride, depth=s3_depth,
                        heads=s3_heads, mlp_mult=s3_mlp_mult, dropout=dropout, input_size=4)
        )

        self.transformer_4 = nn.Sequential(
            nn.Conv2d(s3_emb_dim, s4_emb_dim, kernel_size=s4_emb_kernel, padding=s4_emb_kernel // 2,
                      stride=s4_emb_stride),
            LayerNorm(s4_emb_dim),
            Transformer(dim=s4_emb_dim, proj_kernel=s4_proj_kernel, kv_proj_stride=s4_kv_proj_stride, depth=s4_depth,
                        heads=s4_heads, mlp_mult=s4_mlp_mult, dropout=dropout, input_size=4),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.final_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            # nn.Linear(s4_emb_dim, num_classes)
        )

        self.stem_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            # nn.Linear(s4_emb_dim, num_classes)
        )

        self.cls = nn.Linear(s4_emb_dim, num_classes)


        # for prefix in ('s1','s2', 's3'):
        #     config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
        #
        #     layers.append(nn.Sequential(
        #         nn.Conv2d(dim, config['emb_dim'], kernel_size = config['emb_kernel'], padding = (config['emb_kernel'] // 2), stride = config['emb_stride']),
        #         LayerNorm(config['emb_dim']),
        #         Transformer(dim=config['emb_dim'], proj_kernel = config['proj_kernel'], kv_proj_stride = config['kv_proj_stride'], depth = config['depth'], heads = config['heads'], mlp_mult = config['mlp_mult'], dropout = dropout)
        #     ))
        #
        #     dim = config['emb_dim']
        #
        # self.layers = nn.Sequential(
        #     *layers,
        #     nn.AdaptiveAvgPool2d(1),
        #     Rearrange('... () () -> ...'),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, x):
        x = x.squeeze()
        x = self.patch_embed(x)
        residual = x
        x = self.transformer_1(x)
        x = self.transformer_2(x)
        x = self.transformer_3(x)
        x = self.transformer_4(x)
        x = self.final_cls(x)
        residual = self.stem_cls(residual)
        x = self.gammc[0] * residual + self.gammc[1] * x
        # print(x.shape)
        x = self.cls(x)

        return x

if __name__ == '__main__':
    img_size = 9
    x = torch.rand(2, 1, 200, img_size, img_size)
    model = CAT(num_classes=16, channels=200,)

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