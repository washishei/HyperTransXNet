import torch
import math
import numpy as np
import copy
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import ast
from numpy import load
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import chi
from .core.quaternion_layers import *
from .QCNN.generateq import *

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        values, indices = torch.topk(y, 3, dim=1, largest=True, sorted=True)
        b, c, h, w = x.shape
        # output = x * y.expand_as(x)
        # output = torch.sum(output, dim=1).unsqueeze(1)
        out =[]
        for i in range(b):
            m = x[i, :, :, :]
            j = indices[i]
            j = torch.squeeze(j)
            t = m.index_select(0, j)
            t = torch.unsqueeze(t, 0)
            out.append(t)
        out = torch.cat(out, dim=0)
        # z = torch.cat([output, out], dim=1)
        return out


class eca_layer_one(nn.Module):
    def __init__(self, channel, k_size):
        super(eca_layer_one, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        x = torch.sum(x, dim=1)
        return x.unsqueeze(1)

class eca_layer_two(nn.Module):
    def __init__(self, channel, k_size):
        super(eca_layer_two, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv2d(channel, 1, kernel_size=k_size, bias=False, groups=1)
        self.relu = torch.nn.GELU()


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.conv(x)
        y = self.relu(y)
        return y

class generateq(nn.Module):
    def __init__(self, channel, k_size):
        super(generateq, self).__init__()
        self.conv1 = eca_layer(channel, k_size=1)
        self.conv2 = eca_layer_one(channel, k_size=1)
       # self.conv3 = eca_layer_two(channel, k_size=1)

    def forward(self, x):
        # x = x.squeeze()
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        #x3 = self.conv3(x)
        y1 = torch.cat([x2, x1], dim=1)
        #y2 = torch.cat([x3, x1], dim=1)
        return y1

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=120):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda(0)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = QuaternionLinearAutograd(d_model, d_model)
        self.v_linear = QuaternionLinearAutograd(d_model, d_model)
        self.k_linear = QuaternionLinearAutograd(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = QuaternionLinearAutograd(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next

        # print(k.shape,q.shape,v.shape)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def quarternion_multiplication(a, b, transpose=True):
    """ Performs hamilton product between two quarternion sequences.
    a = (r,x,y,z)
    b = (r',x',y',z')
    following:
    (rr' - xx' - yy' - zz')  +
    (rx' + xr' + yz' - zy')i +
    (ry' - xz' + yr' + zx')j +
    (rz' + xy' - yx' + zr')k
    """

    ar, ax, ay, az = torch.chunk(a, chunks=4, dim=-1)
    br, bx, by, bz = torch.chunk(b, chunks=4, dim=-1)
    # print(ar.shape)
    # print(br.shape)

    if transpose == True:
        if len(br.shape) > 2:
            # r = torch.matmul(br.transpose(-2,-1),ar) - torch.matmul(bx.transpose(-2,-1),ax) - torch.matmul(by.transpose(-2,-1),ay) - torch.matmul(bz.transpose(-2,-1),az)
            # i = torch.matmul(bx.transpose(-2,-1),ar) + torch.matmul(br.transpose(-2,-1),ax) + torch.matmul(bz.transpose(-2,-1),ay) - torch.matmul(by.transpose(-2,-1),az)
            # j = torch.matmul(by.transpose(-2,-1),ar) - torch.matmul(bz.transpose(-2,-1),ax) + torch.matmul(br.transpose(-2,-1),ay) + torch.matmul(bx.transpose(-2,-1),az)
            # k = torch.matmul(bz.transpose(-2,-1),ar) + torch.matmul(by.transpose(-2,-1),ax) - torch.matmul(bx.transpose(-2,-1),ay) + torch.matmul(br.transpose(-2,-1),az)

            r = torch.matmul(ar, br.transpose(-2, -1)) - torch.matmul(ax, bx.transpose(-2, -1)) - torch.matmul(ay,
                                                                                                               by.transpose(
                                                                                                                   -2,
                                                                                                                   -1)) - torch.matmul(
                az, bz.transpose(-2, -1))
            i = torch.matmul(ar, bx.transpose(-2, -1)) + torch.matmul(ax, br.transpose(-2, -1)) + torch.matmul(ay,
                                                                                                               bz.transpose(
                                                                                                                   -2,
                                                                                                                   -1)) - torch.matmul(
                az, by.transpose(-2, -1))
            j = torch.matmul(ar, by.transpose(-2, -1)) - torch.matmul(ax, bz.transpose(-2, -1)) + torch.matmul(ay,
                                                                                                               br.transpose(
                                                                                                                   -2,
                                                                                                                   -1)) + torch.matmul(
                az, bx.transpose(-2, -1))
            k = torch.matmul(ar, bz.transpose(-2, -1)) + torch.matmul(ax, by.transpose(-2, -1)) - torch.matmul(ay,
                                                                                                               bx.transpose(
                                                                                                                   -2,
                                                                                                                   -1)) + torch.matmul(
                az, br.transpose(-2, -1))


        else:
            r = torch.matmul(ar, br.t()) - torch.matmul(ax, bx.t()) - torch.matmul(ay, by.t()) - torch.matmul(az,
                                                                                                              bz.t())
            i = torch.matmul(ar, bx.t()) + torch.matmul(ax, br.t()) + torch.matmul(ay, bz.t()) - torch.matmul(az,
                                                                                                              by.t())
            j = torch.matmul(ar, by.t()) - torch.matmul(ax, bz.t()) + torch.matmul(ay, br.t()) + torch.matmul(az,
                                                                                                              bx.t())
            k = torch.matmul(ar, bz.t()) + torch.matmul(ax, by.t()) - torch.matmul(ay, bx.t()) + torch.matmul(az,
                                                                                                              br.t())
    else:
        r = torch.matmul(ar, br) - torch.matmul(ax, bx) - torch.matmul(ay, by) - torch.matmul(az, bz)
        i = torch.matmul(ar, bx) + torch.matmul(ax, br) + torch.matmul(ay, bz) - torch.matmul(az, by)
        j = torch.matmul(ar, by) - torch.matmul(ax, bz) + torch.matmul(ay, br) + torch.matmul(az, bx)
        k = torch.matmul(ar, bz) + torch.matmul(ax, by) - torch.matmul(ay, bx) + torch.matmul(az, br)

    return [r, i, j, k]


def attention(q, k, v, d_k, mask=None, dropout=None):
    [scores_r, scores_i, scores_j, scores_k] = [x / math.sqrt(d_k) for x in quarternion_multiplication(q, k)]

    if mask is not None:
        # print("mask",mask)
        mask = mask.unsqueeze(1)

        # print(scores_r.shape)
        scores_r = scores_r.masked_fill(mask == 0, -1e9)
        scores_r = F.softmax(scores_r, dim=-1)
        scores_i = scores_i.masked_fill(mask == 0, -1e9)
        scores_i = F.softmax(scores_i, dim=-1)
        scores_j = scores_j.masked_fill(mask == 0, -1e9)
        scores_j = F.softmax(scores_j, dim=-1)
        scores_k = scores_k.masked_fill(mask == 0, -1e9)
        scores_k = F.softmax(scores_k, dim=-1)

    if dropout is not None:
        scores_r = dropout(scores_r)
        scores_i = dropout(scores_i)
        scores_j = dropout(scores_j)
        scores_k = dropout(scores_k)

    scores = torch.cat([scores_r, scores_i, scores_j, scores_k], dim=-1)
    # print(scores.shape)
    # print(v.shape)
    output = quarternion_multiplication(scores, v, transpose=False)
    output = torch.cat(output, dim=-1)
    return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = QuaternionLinearAutograd(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = QuaternionLinearAutograd(d_ff, d_model)
    def forward(self, x):
        #x = self.dropout(F.relu(self.linear_1(x)))
        x = (F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def QNorm(x, eps):
    r, i, j, k = torch.chunk(x, chunks=4, dim=-1)
    qnorm = torch.sqrt(r * r + i * i + j * j + k * k + eps)
    r = r / qnorm
    i = i / qnorm
    j = j / qnorm
    k = k / qnorm

    return [r, i, j, k]


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model // 4
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        [r, i, j, k] = QNorm(x, self.eps)

        norm_r = self.alpha * r + self.bias
        norm_i = self.alpha * i + self.bias
        norm_j = self.alpha * j + self.bias
        norm_k = self.alpha * k + self.bias
        norm = torch.cat([norm_r, norm_i, norm_j, norm_k], dim=-1)

        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model=80, heads=4, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm_1(x)

        #x = x + self.dropout_1(self.attn(x2, x2, x2))
        x = x + (self.attn(x2, x2, x2))
        x2 = self.norm_2(x)
        #x = x + self.dropout_2(self.ff(x2))
        x = x + (self.ff(x2))
        return x
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, src):
        # x = self.embed(src)
        # x = self.pe(x)
        x = src
        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads)
        # self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src):
        e_outputs = self.encoder(src)
        # output = self.out(e_outputs)
        return e_outputs


class QtN(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, pool = 'cls', channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        in_channels = channels

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pre_press = generateq(channel=channels, k_size=1)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = x.squeeze()
        #x = self.pre_press(x)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



if __name__ == '__main__':
    img_size = 15
    x = torch.rand(10, 1, 200, img_size, img_size)
    dim = 128
    heads = 4
    N = 3
    model = QtN(dim=1024, image_size=15, patch_size=3, depth=6, heads=16, dropout=0.1, emb_dropout=0.1, num_classes=16, channels=200,)

    model.eval()

    # flops = model.flops()
    # print(f"number of GFLOPs: {flops / 1e9}")
    #
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"number of params: {n_parameters}")
    # out = model(x)
    # print(out)

    from thop import profile
    flops, params = profile(model, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
