import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba


class SpectralSpatialGate(nn.Module):
    """光谱-空间自适应门控机制"""

    def __init__(self, embed_dim):
        super().__init__()
        self.spectral_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(embed_dim, embed_dim // 8, 1),
            nn.ReLU(),
            nn.Conv3d(embed_dim // 8, embed_dim, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 8, 1),
            nn.ReLU(),
            nn.Conv3d(embed_dim // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spectral_weight = self.spectral_att(x)
        spatial_weight = self.spatial_att(x)
        return x * spectral_weight * spatial_weight


class HyperMambaBlock(nn.Module):
    """HyperMamba核心模块"""

    def __init__(self, embed_dim=64, d_state=16, d_conv=3):
        super().__init__()
        self.mamba = Mamba(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
            #bimamba_type="v2"
        )
        self.gate = SpectralSpatialGate(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.conv3d = nn.Conv3d(embed_dim, embed_dim, kernel_size=(3, 3, 3), padding=1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        residual = x

        # 光谱维度处理
        spectral_seq = rearrange(x, 'b c t h w -> (b h w) t c')
        spectral_out = self.mamba(spectral_seq)
        spectral_out = rearrange(spectral_out, '(b h w) t c -> b c t h w', h=H, w=W)

        # 空间维度处理
        spatial_out = self.conv3d(x)

        # 特征融合
        fused = spectral_out + spatial_out
        fused = self.gate(fused)
        return self.norm((fused + residual).permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)


class HyperMamba(nn.Module):
    """HyperMamba主网络"""

    def __init__(self, in_channels=1, num_classes=16, embed_dim=64, depths=[2, 2, 2]):
        super().__init__()
        # 3D特征提取
        self.stem = nn.Sequential(
            nn.Conv3d(1, embed_dim, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2))
        )

        # Mamba块堆叠
        self.blocks = nn.ModuleList([
            nn.Sequential(*[
                HyperMambaBlock(embed_dim)
                for _ in range(depths[i])
            ]) for i in range(len(depths))
        ])

        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)  # [B,C,T,H,W]
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# 测试代码
if __name__ == '__main__':
    model = HyperMamba(in_channels=1, num_classes=16).cuda()
    dummy_input = torch.randn(2, 1, 200, 9, 9).cuda()  # [B,C,T,H,W]

    # 前向测试
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # 应输出[2, 16]

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # FLOPs测试（需安装thop）
    try:
        from thop import profile

        flops, _ = profile(model, inputs=(dummy_input,))
        print(f"FLOPs: {flops / 1e9:.2f}G")
    except ImportError:
        print("Install thop for FLOPs calculation: pip install thop")