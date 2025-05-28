import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from AKConv import AKConv
# from AKConv import AKConv
import math
# helper methods
# classes

class ECANet(nn.Module):
    def __init__(self, in_channels, out_channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()
        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, x):
        output = self.fgp(x)
        # print(output.shape)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return self.to_out(output)

class ECANetH(nn.Module):
    def __init__(self, in_channels, out_channels, gamma=2, b=1):
        super(ECANetH, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()
        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        output = self.fgp(x)
        # print(output.shape)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        # print(output.shape)
        output = output.permute(0, 2, 1, 3)
        return self.to_out(output)

class ECANetW(nn.Module):
    def __init__(self, in_channels, out_channels, gamma=2, b=1):
        super(ECANetW, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()
        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        output = output.permute(0, 3, 2, 1)
        return self.to_out(output)

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
    def __init__(self, dim, mult=4, dropout=0.):
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

class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_q = ECANet(dim, inner_dim)
        self.to_k = ECANetH(dim, inner_dim)
        self.to_v = ECANetW(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim*3, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        out = torch.cat([q, k, v], dim=1)

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64, mlp_mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, dim, emb_dim, depth, heads, mlp_mult, dropout=0.):
        super().__init__()
        self.maxpool_conv = nn.MaxPool2d(2)
        self.avgpoo_conv = nn.AvgPool2d(2)
        self.down_conv = nn.Sequential(
                nn.Conv2d(dim, emb_dim, kernel_size=1),
                AKConv(emb_dim, emb_dim, 2),
                LayerNorm(emb_dim),
                Transformer(dim=emb_dim, proj_kernel=3, kv_proj_stride=1, depth=depth, heads=heads, mlp_mult=mlp_mult, dropout=dropout)
            )

    def forward(self, x):
        x1 = self.maxpool_conv(x)
        x2 = self.avgpoo_conv(x)
        x = x1 + x2
        return self.down_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, dim, emb_dim, bilinear, depth, heads, mlp_mult, dropout=0. ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                nn.Conv2d(dim, emb_dim, kernel_size=1),
                AKConv(emb_dim, emb_dim, 2),
                LayerNorm(emb_dim),
                Transformer(dim=emb_dim, proj_kernel=3, kv_proj_stride=1, depth=depth, heads=heads, mlp_mult=mlp_mult, dropout=dropout)
            )
        else:
            self.up = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                AKConv(dim, emb_dim, 2),
                LayerNorm(emb_dim),
                Transformer(dim=emb_dim, proj_kernel=3, kv_proj_stride=1, depth=depth, heads=heads, mlp_mult=mlp_mult, dropout=dropout)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PreConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreConv, self).__init__()
        self.preconv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                AKConv(out_channels, out_channels, 2),
                LayerNorm(out_channels),
                Transformer(dim=out_channels, proj_kernel=3, kv_proj_stride=1, depth=2, heads=2, mlp_mult=4, dropout=0.),
            )

    def forward(self, x):
        x = self.preconv(x)
        return x

class ETUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        dim = n_channels
        layers = []
        self.inc = PreConv(dim, 64)

        #### Encoder-part
        self.down1 = Down(64, 128, depth=2, heads=2, mlp_mult=4, dropout=0.)
        self.down2 = Down(128, 256, depth=2, heads=3, mlp_mult=4, dropout=0.)
        self.down3 = Down(256, 512, depth=10, heads=4, mlp_mult=4, dropout=0.)

        #### Bridge-part
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, depth=2, heads=5, mlp_mult=4, dropout=0.)

        ### Decoder-part
        self.up1 = Up(1024, 512 // factor, bilinear, depth=10, heads=4, mlp_mult=4, dropout=0.)
        self.up2 = Up(512, 256 // factor, bilinear, depth=2, heads=3, mlp_mult=4, dropout=0.)
        self.up3 = Up(256, 128 // factor, bilinear, depth=2, heads=2, mlp_mult=4, dropout=0.)
        self.up4 = Up(128, 64, bilinear, depth=2, heads=1, mlp_mult=4, dropout=0.)

        ### To_Output-part
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    img_size = 128
    x = torch.rand(2, 256, img_size, img_size)
    model = ETUNet(n_channels=256, n_classes=16,)

    from thop import profile
    flops, params = profile(model, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')






