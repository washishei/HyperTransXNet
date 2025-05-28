import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba


class SpectralSpatialScanner(nn.Module):
    """3D光谱-空间扫描模块"""

    def __init__(self, scan_type='SpectralPriority'):
        super().__init__()
        self.scan_type = scan_type

    def forward(self, x):
        B, C, T, H, W = x.shape
        if self.scan_type == 'SpectralPriority':
            x = rearrange(x, 'b c t h w -> b c (h w) t')  # 光谱优先
        elif self.scan_type == 'SpatialPriority':
            x = rearrange(x, 'b c t h w -> b c t (h w)')  # 空间优先
        x = rearrange(x, 'b c ... -> b c (...)')  # 展平为序列
        return x


class MambaBlock(nn.Module):
    """3DSS-Mamba核心模块"""

    def __init__(self, embed_dim, d_state=16, d_inner=256, dt_rank=8, scan_type='SpectralPriority'):
        super().__init__()
        self.scanner = SpectralSpatialScanner(scan_type)
        self.mamba = Mamba(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=4,
            expand=2,
            dt_rank=dt_rank,
            bimamba_type="v2"
        )
        self.norm = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        self.proj =  nn.Sequential(
                nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1),
                #nn.GroupNorm(4, embed_dim),
                #nn.SiLU()
            )

    def forward(self, x):
        residual = x
        #print(x.shape)
        B, C, T, H,W = residual.shape
        x = self.scanner(x)  # 形状变换 [B,C,T,H,W] -> [B,C,Seq]
        x = x.permute(0, 2, 1)  # [B, Seq, C]
        #print(x.shape)
        x = self.mamba(x)
        x_recon = x.view(B, C, T, H, W)
        x = self.proj(x_recon)
        return self.norm(x + residual)


class ThreeDSSMamba(nn.Module):
    """3DSS-Mamba主网络"""

    def __init__(self, in_channels=1, num_classes=16, embed_dim=64,
                 depth=4, scan_types=['SpectralPriority', 'SpatialPriority']):
        super().__init__()
        # 3D特征提取
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(2, 1, 1))
        )

        # 嵌入层
        self.embed = nn.Conv3d(32, embed_dim, kernel_size=1)

        # Mamba块堆叠
        self.blocks = nn.ModuleList([
            MambaBlock(embed_dim=embed_dim, scan_type=scan_types[i % 2])
            for i in range(depth)
        ])

        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.conv3d(x)  # [B, C, T, H, W]
        x = self.embed(x)  # 通道升维

        for block in self.blocks:
            x = block(x)

        return self.head(x)


# 测试代码
if __name__ == '__main__':
    model = ThreeDSSMamba(in_channels=1, num_classes=16).cuda()
    dummy_input = torch.randn(2, 1, 200, 9, 9).float().cuda()  # [B, C, T, H, W]

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