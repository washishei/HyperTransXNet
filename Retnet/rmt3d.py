import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from timm.models.layers import DropPath


class ManhattanRelPos3d(nn.Module):
    """基于曼哈顿距离的3D相对位置编码"""

    def __init__(self, embed_dim, num_heads, base_decay=1.0, gamma=0.9):
        super().__init__()
        self.num_heads = num_heads
        self.base_decay = base_decay
        self.gamma = gamma

        # 可学习的衰减参数
        self.decay = nn.Parameter(torch.ones(1, num_heads, 1, 1, 1) * base_decay)

        # 光谱维度位置编码
        self.spec_pe = nn.Parameter(torch.randn(1, num_heads, 1, 1, embed_dim // num_heads))

    def get_manhattan_mask(self, H, W, C):
        """生成曼哈顿距离衰减矩阵"""
        idx_h = torch.arange(H).cuda()
        idx_w = torch.arange(W).cuda()
        idx_c = torch.arange(C).cuda()

        # 生成3D坐标网格
        grid_h, grid_w, grid_c = torch.meshgrid(idx_h, idx_w, idx_c)
        grid = torch.stack([grid_h, grid_w, grid_c], dim=-1)  # (H,W,C,3)

        # 计算曼哈顿距离
        dist = (grid[:, :, :, None, None, None] - grid[None, None, None, :, :, :]).abs().sum(dim=-1)  # (H,W,C,H,W,C)

        # 应用指数衰减
        # print('disk shape is: ', dist.shape)
        mask = self.gamma ** dist # (H,W,C,H,W,C)
        # print(mask.shape)
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W,C,H,W,C)

    def forward(self, x):
        """
        输入: x (B, H, W, C, D)
        输出: 衰减矩阵 (B, num_heads, H, W, C, H, W, C)
        """
        B, H, W, C, D = x.shape
        mask = self.get_manhattan_mask(H, W, C)
        # print(mask.shape)
        #
        # 加入光谱位置编码
        # print(mask.mean(dim=(5, 6, 7), keepdim=True).shape, self.spec_pe.shape)
        spec_pe = self.spec_pe * mask.mean(dim=(5, 6, 7), keepdim=True)
        # print(spec_pe.shape, mask.shape)
        mask = mask * (1.0 + spec_pe)

        # 应用可学习衰减
        mask = mask * self.decay
        return mask


class ManhattanRetention3D(nn.Module):
    """3D曼哈顿注意力保留机制"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.pos_encoder = ManhattanRelPos3d(embed_dim, num_heads)

    def forward(self, x):
        # print(x.shape)
        B, H, W, C, D = x.shape
        q = self.q_proj(x).view(B, H, W, C, self.num_heads, self.head_dim).permute(0, 4, 1, 2, 3, 5)
        k = self.k_proj(x).view(B, H, W, C, self.num_heads, self.head_dim).permute(0, 4, 1, 2, 3, 5)
        v = self.v_proj(x).view(B, H, W, C, self.num_heads, self.head_dim).permute(0, 4, 1, 2, 3, 5)

        # 计算注意力分数
        attn = torch.einsum('bnhwcd,bmhwcd->bnhwm', q, k) / (self.head_dim ** 0.5)

        # 获取位置衰减矩阵
        pos_mask = self.pos_encoder(x)  # (B, num_heads, H, W, C, H, W, C)

        # 应用曼哈顿衰减
        print(attn.unsqueeze(-1).shape, pos_mask.shape)
        attn = attn.unsqueeze(-1) * pos_mask  # (B, num_heads, H, W, C, H, W, C)
        attn = F.softmax(attn, dim=-1)

        # 值聚合
        out = torch.einsum('bnhwmhwc,bmhwcd->bnhwcd', attn, v)
        out = out.permute(0, 2, 3, 4, 1, 5).reshape(B, H, W, C, D)
        return self.out_proj(out)


class SpectralSpatialBlock(nn.Module):
    """空间-光谱联合处理块"""

    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ManhattanRetention3D(dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        # 形状转换 (B, D, H, W, C) -> (B, H, W, C, D)
        x = x.permute(0, 2, 3, 4, 1)
        # print(x.shape)

        # 曼哈顿注意力
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 形状恢复
        return x.permute(0, 4, 1, 2, 3)


class RMT3D(nn.Module):
    """高光谱3D保留网络"""

    def __init__(self, in_channels=200, num_classes=16,
                 embed_dims=[64, 128, 256], num_heads=[4, 8, 16],
                 depths=[2, 2, 2], mlp_ratios=[4, 4, 4],
                 drop_path_rate=0.1):
        super().__init__()

        # 输入嵌入
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, embed_dims[0], kernel_size=(3, 3, 7),
                      padding=(1, 1, 3), stride=(1, 1, 2)),
            nn.InstanceNorm3d(embed_dims[0]),
            nn.GELU()
        )

        # 构建层级结构
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        for i in range(len(embed_dims)):
            stage = nn.Sequential(
                *[SpectralSpatialBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    drop_path=dpr[sum(depths[:i]) + j]
                ) for j in range(depths[i])],
                nn.Conv3d(embed_dims[i], embed_dims[i + 1] if i < len(embed_dims) - 1 else embed_dims[i],
                          kernel_size=3, stride=2, padding=1) if i < len(embed_dims) - 1 else nn.Identity()
            )
            self.stages.append(stage)

        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(embed_dims[-1], num_classes)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.stem(x)
        print(x.shape)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)

if __name__ == '__main__':
    img_size = 21
    # x = torch.rand(2, 1, 200, img_size, img_size)
#     model = RMT3D(
#     in_channels=1,    # 输入光谱通道数
#     num_classes=16,     # 分类类别数
#     embed_dims=[8],
#     num_heads=[4],
#     depths=[2,]
# )
    model = RMT3D(
        in_channels=1,  # 输入光谱通道数
        num_classes=16,  # 分类类别数
        embed_dims=[8],
        num_heads=[4],
        depths=[2]
    ).cuda()

    x = torch.randn(4, 1, 15, 15, 200).cuda()  # (B, C, H, W, D)
    output = model(x)  # (B, num_classes)

    from thop import profile
    flops, params = profile(model, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')