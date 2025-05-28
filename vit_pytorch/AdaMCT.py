# -*- coding: utf-8 -*-
import torch
import copy
import math
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
import torch.nn.functional as fn
from kan_convolutional .KANConv2D import KKAN_Convolutional_Network

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

class LocalKConv(nn.Module):
    def __init__(self, hidden_size, hidden_act):
        super(LocalKConv, self).__init__()
        self.conv_1 = KKAN_Convolutional_Network(hidden_size, 2, hidden_size)
        self.conv_2 = nn.Conv2d(hidden_size, hidden_size, 1)
        self.conv_act_fn = self.get_hidden_act(hidden_act)
        self.LayerNorm = LayerNorm(hidden_size)
        # self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.LayerNorm(input_tensor)
        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.conv_2(hidden_states)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.conv_act_fn(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

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
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class AdaptiveMixtureUnits(nn.Module):
    def __init__(self, hidden_size):
        super(AdaptiveMixtureUnits, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.adaptive_act_fn = torch.sigmoid
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor, global_output, local_output):
        input_tensor = rearrange(input_tensor, 'b c h w -> b c (h w)')
        input_tensor_avg = torch.mean(input_tensor, dim=1)  # [B, D]
        ada_score_alpha = self.adaptive_act_fn(self.linear(input_tensor_avg)).unsqueeze(-1)  # [B, 1, 1]
        ada_score_beta = 1 - ada_score_alpha

        mixture_output = torch.mul(global_output, ada_score_beta) + torch.mul(local_output,
                                                                              ada_score_alpha)  # [B, N, D]
        input_tensor = input_tensor.permute(0, 2, 1)
        mixture_output = self.linear_out(mixture_output)
        print(mixture_output.shape)

        output = self.LayerNorm(self.dropout(self.linear_out(mixture_output)) + input_tensor)  # [B, N, D]
        return output

class SqueezeExcitationAttention(nn.Module):
    def __init__(self, seq_len, reduction_ratio):
        super(SqueezeExcitationAttention, self).__init__()
        self.dense_1 = nn.Linear(seq_len, seq_len // reduction_ratio)
        self.squeeze_act_fn = fn.relu

        self.dense_2 = nn.Linear(seq_len // reduction_ratio, seq_len)
        self.excitation_act_fn = torch.sigmoid

    def forward(self, input_tensor):
        input_tensor_avg = torch.mean(input_tensor, dim=-1, keepdim=True)  # [B, N, 1]
        # print(input_tensor_avg.shape)

        hidden_states = self.dense_1(input_tensor_avg.permute(0, 2, 1))  # [B, 1, N] -> [B, 1, N/r]
        hidden_states = self.squeeze_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)  # [B, 1, N/r] -> [B, 1, N]
        att_score = self.excitation_act_fn(hidden_states)  # sigmoid

        # reweight
        input_tensor = torch.mul(input_tensor, att_score.permute(0, 2, 1))  # [B, N, D]
        return input_tensor
class AdaMCTLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        pool='mean',
    ):
        super(AdaMCTLayer, self).__init__()
        self.linear_en = nn.Linear(hidden_size, hidden_size)
        # self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        # self.dropout = nn.Dropout(0)
        self.window_size = 3
        self.shift_size = 0
        self.LayerNorm = LayerNorm(hidden_size)
        num_patches = (15 // 3) * (15 // 3)
        patch_dim = hidden_size * 3 * 3
        dim = 15 * 15 * 3
        dim2 = 15*15
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=3, p2=3),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))


        self.multi_head_attention = Transformer(dim, 4, 4, 8, 1024, 0)
        self.local_conv = LocalKConv(
            hidden_size,
            "relu",
        )
        self.pool = pool
        self.global_seatt = SqueezeExcitationAttention(dim2, 2)
        self.local_seatt = SqueezeExcitationAttention(dim2, 2)

        self.adaptive_mixture_units = AdaptiveMixtureUnits(dim2)

    def forward(self, hidden_states):
        hidden_states_en = self.LayerNorm(hidden_states)

        x = self.to_patch_embedding(hidden_states_en)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.multi_head_attention(x)
        # print(x.shape)

        global_output = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # print(global_output.shape)
        global_output = global_output.reshape(b, 225, 3)

        # global_output = self.multi_head_attention(hidden_states)
        global_output = self.global_seatt(global_output)
        # print(global_output.shape)

        local_output = self.local_conv(hidden_states_en)
        local_output = rearrange(local_output, 'b c h w -> b (h w) c')
        local_output = self.local_seatt(local_output)

        layer_output = self.adaptive_mixture_units(hidden_states, global_output, local_output)
        return layer_output

if __name__ == '__main__':
    img_size = 15
    x = torch.rand(1, 3, img_size, img_size)
    model = AdaMCTLayer(3,)
    y = model(x)
    print(y.shape)
    model.eval()
    from thop import profile
    flops, params = profile(model, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')