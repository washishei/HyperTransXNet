import torch
import torch.nn as nn
import torch.fft

class QDASA(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio

        # Spectral domain processing
        self.spectral_gate = nn.Sequential(
            nn.Linear(dim, dim // sr_ratio),
            nn.ReLU(),
            nn.Linear(dim // sr_ratio, dim),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 3, padding=1, groups=8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1)
        )
        self.spatial_norm = nn.LayerNorm(dim)

        # Frequency domain processing
        self.freq_proj = nn.Linear(dim * 2, dim * 2)  # 修改输出维度为2*dim
        self.freq_mask = nn.Parameter(torch.randn(dim * 2))  # 调整掩码维度

        # Statistical attention
        # self.stat_proj = nn.Sequential(
        #     nn.Conv2d(dim, dim, 1,),
        #     nn.Conv2d(dim, dim // 8, 1, groups=8),
        #     nn.GELU(),
        #     nn.Conv2d(dim // 8, dim, 1)
        # )
        self.stat_proj = nn.Sequential(
            nn.Conv2d(2, dim, 1, ),
            nn.Conv2d(dim, dim // 8, 3, 1, 1, groups=8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 3, 1, 1)
        )

        self.conv_r = nn.Conv2d(dim, dim // 4, 3, 1, 1)
        self.conv_i = nn.Conv2d(dim, dim // 4, 3, 1, 1)
        self.conv_j = nn.Conv2d(dim, dim // 4, 3, 1, 1)
        self.conv_k = nn.Conv2d(dim, dim // 4, 3, 1, 1)

        # Fusion layer
        self.fusion = nn.Conv2d(dim, dim, 1, groups=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # 1. Spectral Domain Attention
        spectral_weight = self.spectral_gate(x.mean([2, 3]).view(B, C))
        x_spectral = x * spectral_weight.view(B, C, 1, 1) ### k

        # 2. Spatial Domain Attention
        x_spatial = self.spatial_attn(x)
        x_spatial = x_spatial.permute(0, 2, 3, 1)
        x_spatial = self.spatial_norm(x_spatial).permute(0, 3, 1, 2) ### j

        # 3. Frequency Domain Attention
        x_freq = torch.fft.rfft2(x, norm='ortho')
        x_freq = torch.view_as_real(x_freq)
        x_freq = x_freq.permute(0, 4, 1, 2, 3)
        x_freq = x_freq.reshape(B, 2 * C, H, -1)
        x_freq = self.freq_proj(x_freq.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_freq = x_freq * torch.sigmoid(self.freq_mask.view(1, -1, 1, 1))

        # 逆变换处理
        x_freq = x_freq.view(B, 2, C, H, -1).permute(0, 2, 3, 4, 1)
        x_freq = torch.fft.irfft2(torch.view_as_complex(x_freq.contiguous()), s=(H, W), norm='ortho') ### i

        # 4. Statistical Domain Attention
        mean_feat = x.mean(dim=1, keepdim=True)
        std_feat = x.std(dim=1, keepdim=True)
        # x_stat = torch.cat([mean_feat, std_feat], dim=1)
        # print(x_stat.shape)
        x_stat = self.stat_proj(torch.cat([mean_feat, std_feat], dim=1))
        # x_stat = self.stat_proj(x) ### r
        # print(x_stat.shape)

        # print(x_spectral.shape, x_spatial.shape, x_freq.shape, x_stat.shape)
        # Multi-domain fusion
        # fused = torch.cat([
        #     x_spectral.reshape(B, C, -1),
        #     x_spatial.reshape(B, C // 8, -1),
        #     x_freq.reshape(B, C, -1),
        #     x_stat.reshape(B, C // 8, -1)
        # ], dim=1)
        # print(x_spectral.shape)
        # fused = torch.cat([
        #     x_spectral.reshape(B, C, -1),
        #     x_spatial.reshape(B, C, -1),
        #     x_freq.reshape(B, C, -1),
        #     x_stat.reshape(B, C, -1)
        # ], dim=1)
        # out_r = x_stat - x_freq - x_spatial - x_spectral
        # out_i = x_stat + x_freq + x_spatial - x_spectral
        # out_j = x_stat - x_freq + x_spatial + x_spectral
        # out_k = x_stat + x_freq + x_spatial - x_spectral
        r = x_stat
        i = x_freq
        j = x_spatial
        k = x_spectral
        out_r = self.conv_r(r) - self.conv_i(i) - self.conv_j(j) - self.conv_k(k)
        out_i = self.conv_r(i) + self.conv_i(r) + self.conv_j(k) - self.conv_k(j)
        out_j = self.conv_r(j) - self.conv_i(k) + self.conv_j(r) + self.conv_k(i)
        out_k = self.conv_r(k) + self.conv_i(j) - self.conv_j(i) + self.conv_k(r)

        output = torch.cat([out_r, out_i, out_j, out_k], dim=1)
        ### new output
        # output = torch.cat([x_stat, x_freq, x_spatial, x_spectral], dim=1)
        # print(output.shape)
        output = self.fusion(output) + identity
        # ###residual
        # output = output + identity
        # fused = torch.cat([
        #     x_spectral.reshape(B, C, -1),
        #     x_spatial.reshape(B, C, -1),
        #     x_freq.reshape(B, C, -1),
        #     x_stat.reshape(B, C, -1)
        # ], dim=1)
        #
        # x = self.fusion(fused.permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)
        # ###residual
        # output = x + identity
        return output

### 测试代码 ###
if __name__ == "__main__":
    dim = 256
    h = w = 64
    batch_size = 4
    model = QDASA(dim=dim)
    x = torch.randn(batch_size, dim, h, w)
    output = model(x)
    assert output.shape == x.shape, f"形状错误: 输入{x.shape} 输出{output.shape}"
    print(f"测试通过! 输入输出形状一致: {tuple(output.shape)}")
    params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {params / 1e6:.2f}M")