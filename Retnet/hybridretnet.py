import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from timm.models import register_model
# from timm.models.vision_transformer import _cfg
from typing import Tuple, Union
from functools import partial
from .retention import MultiScaleRetention
from .retnet2d import *
from .retnet2d import PatchEmbed, BasicLayer, PatchMerging


class VisRetNetNew(nn.Module):

    def __init__(self, in_chans=256, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 init_values=[1, 1, 1, 1], heads_ranges=[3, 3, 3, 3], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoints=[False, False, False, False],
                 chunkwise_recurrents=[True, True, False, False],
                 layerscales=[False, False, False, False], layer_init_values=1e-6,
                 **kwargs):

        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.hidden_dim = hidden_dim = embed_dims[0]
        self.layer_x = layers = 12
        ffn_size = 1024
        heads = 4
        self.gamma = nn.Parameter(torch.ones(3))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0],
                                      norm_layer=norm_layer if self.patch_norm else None)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.pool1 = nn.AdaptiveAvgPool2d(1)

        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim=False)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoints[i_layer],
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values
            )
            self.layers.append(layer)

        self.norm = nn.BatchNorm2d(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_s = x
        x_s = x_s.squeeze()
        x = self.patch_embed(x)
        residual = x
        # print(residual.shape)

        x_s = self.encoder(x_s)
        x_s = x_s.view(-1, self.hidden_dim)
        # x_s = self.pool1(x_s)
        x_s = x_s.unsqueeze(1)

        for i in range(self.layer_x):
            Y = self.retentions[i](self.layer_norms_1[i](x_s)) + x_s
            x_s = self.ffns[i](self.layer_norms_2[i](Y)) + Y

        for layer in self.layers:
            x = layer(x)

        b, h, w, c = x.shape
        if h != 1:
            x = self.norm(x.permute(0, 3, 1, 2)).flatten(2, 3)  # (b c h*w)
            x = self.avgpool(x)  # B C 1
        else:
            x = torch.flatten(x, 1)
        x_s = torch.flatten(x_s, 1)
        residual = torch.flatten(residual, 1)
        # print(x.shape, x_s.shape, residual.shape)
        out = self.gamma[0] * x_s + self.gamma[1] * x + self.gamma[2] * residual
        # print(out.shape)
        return out

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    img_size = 15
    x = torch.rand(2, 1, 256, img_size, img_size)
    model = VisRetNetNew(
        embed_dims=[256, 256, 256, 256],
        depths=[2, 2, 4, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, True, False, False],
        layerscales=[False, False, False, False],
    )

    from thop import profile
    flops, params = profile(model, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

