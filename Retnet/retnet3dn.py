import torch
import torch.nn as nn
import math


class MultiScaleRetention3D(nn.Module):
    def __init__(self, dim, heads, double_v_dim=False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.v_dim = dim * 2 if double_v_dim else dim
        assert dim % heads == 0, "dim must be divisible by heads"
        self.head_dim = dim // heads
        self.gamma = nn.Parameter(torch.randn(1))

        # 动态卷积核保证维度一致性
        self.spatial_conv = nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1))
        self.spectral_conv = nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0))
        self.qkv = nn.Conv3d(dim, dim * 3, 1)
        self.proj = nn.Conv3d(dim, dim, 1)

    def _get_dist_matrix(self, D, H, W, device):
        # 完全动态化的距离矩阵生成
        d_idx = torch.arange(D, device=device).view(D, 1, 1, 1, 1, 1)
        h_idx = torch.arange(H, device=device).view(1, 1, H, 1, 1, 1)
        w_idx = torch.arange(W, device=device).view(1, 1, 1, 1, W, 1)

        # 计算三维曼哈顿距离 [D,D,H,H,W,W]
        dist = (torch.abs(d_idx - d_idx.transpose(0, 1)) +
                torch.abs(h_idx - h_idx.transpose(2, 3)) +
                torch.abs(w_idx - w_idx.transpose(4, 5)))
        return dist.float()

    def forward(self, x):
        B, C, D, H, W = x.shape
        device = x.device

        # 特征变换
        x = self.spatial_conv(x)
        x = self.spectral_conv(x)

        # 动态生成衰减矩阵
        dist_mat = self._get_dist_matrix(D, H, W, device)  # [D,D,H,H,W,W]
        decay_mask = torch.pow(self.gamma.sigmoid(), dist_mat)  # [D,D,H,H,W,W]
        # print(decay_mask.shape)

        # 多头特征分解
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = [x.view(B, self.heads, self.head_dim, D, H, W) for x in qkv]
        # print(q.shape, v.shape)

        # 注意力计算（优化维度映射）
        attn = torch.einsum('bhdijk,bhdmno->bhijkmno', q, k) / math.sqrt(self.head_dim)

        # 精准广播机制
        decay_mask = decay_mask.unsqueeze(0).unsqueeze(0)  # [1,1,D,D,H,H,W,W]
        # print(attn.shape, decay_mask.shape)
        attn = attn.permute(0, 1, 2, 5, 3, 4, 6, 7) * decay_mask  # 自动对齐所有维度

        attn = torch.softmax(attn.permute(0, 1, 2, 4, 5, 3, 6, 7), dim=-1)
        # print(attn.shape, v.shape)

        # 输出投影
        out = torch.einsum('bhijkmno,bhdmno->bhdijk', attn, v)
        return self.proj(out.contiguous().view(B, C, D, H, W))


class RetNet3D(nn.Module):
    def __init__(self, in_chans, num_classes, layers=4, dim=16, ffn_dim=128, heads=4):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"

        # 输入编码器（兼容非对称下采样）
        self.encoder = nn.Sequential(
            nn.Conv3d(in_chans, dim, kernel_size=(7, 3, 3),
                      stride=(2, 1, 1), padding=(3, 1, 1)),
            nn.InstanceNorm3d(dim),
            nn.GELU()
        )

        # 残差块结构（强化梯度流）
        self.retentions = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                MultiScaleRetention3D(dim, heads),
                nn.LayerNorm(dim),
                nn.Conv3d(dim, ffn_dim, 1),
                nn.GELU(),
                nn.Conv3d(ffn_dim, dim, 1)
            ) for _ in range(layers)
        ])

        # 分类头（保持维度鲁棒性）
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.encoder(x)  # [B,C,D,H,W]
        # print(x.shape)
        for block in self.retentions:
            residual = x
            # 标准化维度处理流程
            x = block[0](x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)  # LayerNorm
            # print(x.shape)
            x = block[1](x) + residual  # Retention
            x = block[2](x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)  # LayerNorm
            x = block[3](x)  # FFN conv1
            x = block[4](x)  # GELU
            x = block[5](x) + residual  # FFN conv2
        return self.head(x)


# 测试案例
if __name__ == '__main__':
    model = RetNet3D(in_chans=1, num_classes=16, layers=4, dim=16, heads=4).cuda()
    x = torch.randn(2, 1, 128, 15, 15).cuda()  # (B,C,D,H,W)
    print(model(x).shape)  # torch.Size([2, 16])
    from thop import profile
    flops, params = profile(model, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')