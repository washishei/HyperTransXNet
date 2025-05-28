import torch
import torch.nn as nn
from torch.nn import functional as F


class ManhattanAttention3D(nn.Module):
    """基于曼哈顿距离的3D空间-光谱注意力"""

    def __init__(self, dim, num_heads=4, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        # 空间下采样
        self.sr_conv = nn.Conv3d(dim, dim, kernel_size=(1, sr_ratio, sr_ratio),
                                 stride=(1, sr_ratio, sr_ratio), groups=dim)

        # 投影层
        self.q = nn.Conv3d(dim, dim, 1)
        self.kv = nn.Conv3d(dim, dim * 2, 1)

        # 位置衰减参数
        self.decay_gamma = nn.Parameter(torch.ones(num_heads)).cuda()
        self.decay_beta = nn.Parameter(torch.zeros(num_heads)).cuda()

    # def generate_2d_decay(self, H: int, W: int):
    #     '''
    #     generate 2d decay mask, the result is (HW)*(HW)
    #     '''
    #     index_h = torch.arange(H).to(self.decay)
    #     index_w = torch.arange(W).to(self.decay)
    #     grid = torch.meshgrid([index_h, index_w])
    #     grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # (H*W 2)
    #     mask = grid[:, None, :] - grid[None, :, :]  # (H*W H*W 2)
    #     mask = (mask.abs()).sum(dim=-1)
    #     mask = mask * self.decay[:, None, None]  # (n H*W H*W)
    #     return mask
    #
    # def generate_1d_decay(self, l: int):
    #     '''
    #     generate 1d decay mask, the result is l*l
    #     '''
    #     index = torch.arange(l).to(self.decay)
    #     mask = index[:, None] - index[None, :]  # (l l)
    #     mask = mask.abs()  # (l l)
    #     mask = mask * self.decay[:, None, None]  # (n l l)
    #     return mask
    def _manhattan_decay(self, H, W, D):
        """生成曼哈顿距离衰减矩阵"""
        # 空间距离
        # print(H, W)
        h_idx = torch.arange(H).view(H, 1)
        w_idx = torch.arange(W).view(1, W)
        spatial_dist = (h_idx - h_idx.T).abs() + (w_idx - w_idx.T).abs()
        # spatial_dist = self.generate_2d_decay(H, W)
        #
        # 光谱距离
        d_idx = torch.arange(D).view(D)
        spectral_dist = (d_idx - d_idx.T).abs()
        # spectral_dist = self.generate_1d_decay(D)

        # print(spectral_dist.shape, spatial_dist.shape)

        # 组合衰减
        decay = (spatial_dist.view(1, H, W, 1) + spectral_dist.view(1, 1, 1, D)).cuda()
        decay = decay * self.decay_gamma.view(-1, 1, 1, 1) + self.decay_beta.view(-1, 1, 1, 1)
        return torch.exp(-F.softplus(decay))

    def forward(self, x):
        # print(x.shape)
        B, C, H, W, D = x.shape
        q = self.q(x).view(B, self.num_heads, self.head_dim, H, W, D)

        # 下采样KV
        kv = self.sr_conv(x)
        k, v = self.kv(kv).chunk(2, dim=1)
        k = k.view(B, self.num_heads, self.head_dim, H // self.sr_ratio, W // self.sr_ratio, D)
        v = v.view(B, self.num_heads, self.head_dim, H // self.sr_ratio, W // self.sr_ratio, D)

        # 曼哈顿衰减矩阵
        attn_decay = self._manhattan_decay(H, W, D).to(x.device)

        # 注意力计算
        q = q.permute(0, 1, 5, 2, 3, 4)  # [B, H, D, C/h, H, W]
        k = k.permute(0, 1, 5, 2, 3, 4)
        # print(q.shape, k.shape)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.permute(0, 1, 3, 4, 5, 2)
        # print(attn.shape, attn_decay.unsqueeze(0).unsqueeze(2).shape)
        attn = attn + attn_decay.unsqueeze(0).unsqueeze(2)

        attn = attn.softmax(dim=-1)
        # print(attn.shape,v.shape)
        x = (attn.permute(0, 1, 5, 2, 3, 4) @ v.permute(0, 1, 5, 2, 3, 4)).permute(0, 1, 3, 4, 5, 2)
        # print(x.reshape(B, C, H, W, D).shape)
        return x.reshape(B, C, H, W, D)


class HybridEncoderBlock(nn.Module):
    """1D光谱+2D空间混合编码块"""

    def __init__(self, dim, num_heads, expansion=4):
        super().__init__()
        # 光谱1D路径
        self.spectral_conv = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.spectral_norm = nn.BatchNorm3d(dim)

        # 空间2D路径
        self.spatial_attn = ManhattanAttention3D(dim, num_heads)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv3d(dim * 2, dim, 1),
            nn.GELU(),
            nn.BatchNorm3d(dim)
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, dim * expansion, 1),
            nn.GELU(),
            nn.Conv3d(dim * expansion, dim, 1))

    def forward(self, x):
        # 光谱1D处理
        spectral = self.spectral_conv(x)
        spectral = self.spectral_norm(spectral)

        # 空间2D处理
        spatial = self.spatial_attn(x)

        # 特征融合
        fused = self.fusion(torch.cat([spectral, spatial], dim=1))
        return fused + x  # 残差连接


class RMT3D(nn.Module):
    """轻量化RMT-3D模型"""

    def __init__(self, in_channels=128, num_classes=16, stages=[2, 2, 4, 2], dims=[32, 64, 128, 256]):
        super().__init__()
        # 三维特征提取
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=(3, 3, 5), stride=(1, 1, 1), padding=(1, 1, 2)),
            nn.BatchNorm3d(dims[0]),
            nn.GELU())

        # 多阶段编码
        self.blocks = nn.ModuleList()
        for i in range(len(stages)):
            stage = nn.Sequential(
                *[HybridEncoderBlock(dims[i], num_heads=4) for _ in range(stages[i])],
                nn.Conv3d(dims[i], dims[i + 1] if i < 3 else dims[i],
                          kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
            )
            self.blocks.append(stage)

        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(dims[-1], num_classes))

    def forward(self, x):
        """输入形状: [B, C, H, W, D]"""
        x = self.stem(x)
        # print(x.shape)
        for block in self.blocks:
            x = block(x)
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
        stages=[2, 2, 4, 2],  # 各阶段块数
        dims=[32, 64, 128, 256]  # 特征维度变化
    ).cuda()

    # 输入形状: [Batch, Channels, Height, Width, Depth]
    input_data = torch.randn(4, 1, 15, 15, 100).cuda()
    output = model(input_data)  # 输出分类结果


    from thop import profile
    flops, params = profile(model, (input_data,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')