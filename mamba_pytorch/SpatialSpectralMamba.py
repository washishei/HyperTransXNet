import torch
import torch.nn as nn
from mamba_ssm import Mamba
from timm.models.layers import DropPath

class SpectralMambaBlock(nn.Module):
    """三维光谱-空间Mamba块"""

    def __init__(self, dim, d_state=16, expand=2, spectral_dim=16):
        super().__init__()
        # 空间维度处理
        self.spatial_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=4,
            expand=expand
        )
        # 光谱维度处理
        self.spectral_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=4,
            expand=expand
        )
        # 跨维度融合
        self.fusion = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm3d(dim)
        )

    def forward(self, x):
        """输入形状: [B, C, H, W, D]"""
        B, C, H, W, D = x.shape

        # 空间路径 (HxW为序列)
        spatial = x.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)
        spatial = spatial.view(B * D, C, -1).permute(0, 2, 1)
        spatial = self.spatial_mamba(spatial)
        spatial = spatial.permute(0, 2, 1).view(B * D, C, H, W)
        spatial = spatial.view(B, D, C, H, W).permute(0, 2, 3, 4, 1)

        # 光谱路径 (D为序列)
        spectral = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, C, D)
        spectral = self.spectral_mamba(spectral)
        spectral = spectral.view(B, H, W, C, D).permute(0, 3, 0, 1, 2)

        # 特征融合
        fused = self.fusion(torch.cat([spatial, spectral], dim=1))
        return fused + x  # 残差连接


class TransMamba(nn.Module):
    def __init__(self, num_classes, in_channels=128, spectral_dim=16, stages=[3, 3, 9, 3], dims=[48, 96, 224, 448]):
        super().__init__()
        # 三维嵌入层
        self.patch_embed = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.GELU(),
            nn.BatchNorm3d(dims[0]))

        # 多阶段处理
        self.stages = nn.ModuleList()
        for i in range(len(stages)):
            stage = nn.Sequential(
                *[SpectralMambaBlock(dims[i], spectral_dim=spectral_dim) for _ in range(stages[i])],
                nn.Conv3d(dims[i], dims[i + 1] if i < 3 else dims[i],
                          kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            )
            self.stages.append(stage)

        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(dims[-1], num_classes))

    def forward(self, x):
        """输入形状: [B, C, H, W, D] (C:光谱通道数)"""
        # x = x.squeeze(1)
        # print(x.shape)
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


class MultiScaleSpectralMamba(SpectralMambaBlock):
    """增强版多尺度光谱Mamba"""

    def __init__(self, dim, scales=[1, 2, 4], **kwargs):
        super().__init__(dim, **kwargs)
        self.scale_convs = nn.ModuleList()
        for s in scales:
            self.scale_convs.append(
                nn.Conv3d(dim, dim, kernel_size=(s * 2 + 1, 3, 3),
                          padding=(s, 1, 1), groups=dim))
        self.attn = nn.Parameter(torch.ones(len(scales)) / len(scales))

    def forward(self, x):
        # 多尺度特征提取
        ms_feats = [conv(x) for conv in self.scale_convs]
        # 自适应融合
        fused = sum([a * f for a, f in zip(self.attn.softmax(dim=0), ms_feats)])
        return super().forward(fused)
if __name__ == '__main__':
    img_size = 15
    x = torch.rand(2, 1, 200, img_size, img_size)
    net = TransMamba(
    num_classes=16,
    in_channels=1,   # 输入光谱通道数
    spectral_dim=16,   # 光谱维度压缩后长度
    stages=[3,3,9,3],  # 各阶段块数
    dims=[48,96,224,448] # 各阶段通道数
)
    # print(net)
    net.eval()


    from thop import profile
    flops, params = profile(net, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

