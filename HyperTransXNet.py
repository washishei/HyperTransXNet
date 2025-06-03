import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# 修正1：调整光谱预处理维度处理
class HSIProcessor(nn.Module):
    """高光谱数据预处理（维度修正版）"""

    def __init__(self, in_channels, reduce_dim=30):
        super().__init__()
        # 使用2D卷积保持光谱维度
        self.pca = nn.Sequential(
            nn.Conv2d(in_channels, reduce_dim, 1),  # 通道降维
            nn.BatchNorm2d(reduce_dim),
            nn.GELU()
        )

    def forward(self, x):
        # 输入形状: (B, D, H, W)
        return self.pca(x)  # 输出形状: (B, C, H, W)


# 修正2：重构光谱-空间特征提取
class SpectralLocalUnit(nn.Module):
    """修正通道一致性的局部单元"""

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.pad_size = kernel_size // 2

        # 动态权重生成（通道维度对齐）
        self.dynamic_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),  # 保持通道一致
            nn.GELU(),
            nn.Conv2d(dim, dim * kernel_size ** 2, 1)  # 输出通道: dim*K²
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.dim, f"输入通道{C}与初始化维度{self.dim}不匹配"

        # 生成动态卷积核
        weights = self.dynamic_net(x)  # (B, C*K², 1, 1)
        weights = weights.view(B, self.dim, self.kernel_size, self.kernel_size)

        # 输入展开
        x_pad = F.pad(x, [self.pad_size] * 4, mode='reflect')
        x_unfold = F.unfold(x_pad, self.kernel_size)  # (B, C*K², H*W)
        x_unfold = x_unfold.view(B, self.dim, self.kernel_size ** 2, H, W)

        # 应用动态权重
        weights = weights.view(B, self.dim, self.kernel_size ** 2, 1, 1)
        output = (x_unfold * weights).sum(dim=2)
        return output


# 修正3：优化全局注意力维度
class SpectralGlobalUnit(nn.Module):
    """光谱-空间注意力（维度修正版）"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 光谱变换
        self.spec_fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        # 空间注意力
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads)

    def forward(self, x):
        B, C, H, W = x.shape

        # 光谱维度处理
        x_spec = rearrange(x, 'b c h w -> (b h w) c')
        x_spec = torch.fft.rfft(self.spec_fc(x_spec), dim=-1)
        x_spec = torch.fft.irfft(x_spec, n=C, dim=-1)
        x_spec = self.norm(x_spec)
        x_spec = rearrange(x_spec, '(b h w) c -> b c h w', b=B, h=H, w=W)

        # 空间注意力
        x_spatial = rearrange(x_spec, 'b c h w -> (h w) b c')
        attn_out, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        attn_out = rearrange(attn_out, '(h w) b c -> b c h w', h=H, w=W)
        return attn_out


# 修正4：优化MoE维度处理
class SpectralMoE(nn.Module):
    """混合专家模块（维度修正版）"""

    def __init__(self, dim, num_experts=4, top_k=1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1)
            ) for _ in range(num_experts)
        ])

        # 路由网络
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 路由权重
        router_weights = self.router(x)  # (B, E)
        top_weights, top_indices = torch.topk(router_weights, self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 专家计算
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, E, C, H, W)

        # 构建选择矩阵
        gates = torch.zeros(B, self.num_experts, device=x.device)
        gates.scatter_(1, top_indices, top_weights)

        # 融合输出
        combined = torch.einsum('bechw,be->bchw', expert_outputs, gates)
        return combined


# 主网络架构
class HyperSpectralClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, reduce_dim=30, num_blocks=4):
        super().__init__()
        # 预处理层（修正通道对齐）
        self.preprocess = nn.Sequential(
            nn.Conv2d(input_dim, reduce_dim, 1),  # 光谱降维
            nn.BatchNorm2d(reduce_dim),
            nn.GELU(),
            nn.Conv2d(reduce_dim, 64, 3, padding=1)  # 通道对齐
        )

        # 特征提取块（修正通道扩展逻辑）
        self.blocks = nn.ModuleList()
        current_dim = 64
        for _ in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    SpectralHybridBlock(current_dim),
                    nn.Conv2d(current_dim, current_dim * 2, 3, stride=2, padding=1),  # 下采样并扩展通道
                    nn.BatchNorm2d(current_dim * 2),
                    nn.GELU()
                )
            )
            current_dim *= 2  # 更新通道维度

        # 分类头（动态适应最终维度）
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_dim, num_classes)
        )

    def forward(self, x):
        x = x.squeeze()
        x = self.preprocess(x)  # (B, 64, H, W)

        for block in self.blocks:
            x = block(x)

        return self.head(x)


class SpectralHybridBlock(nn.Module):
    """修正通道一致性的混合块"""

    def __init__(self, dim):
        super().__init__()
        # 确保所有子模块使用统一维度
        self.local = SpectralLocalUnit(dim)
        self.global_unit = SpectralGlobalUnit(dim)
        self.moe = SpectralMoE(dim)

        # 通道对齐投影
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x):
        identity = x

        # 局部特征
        local_feat = self.local(x)

        # 全局特征
        global_feat = self.global_unit(x)

        # 动态融合（保持维度一致）
        fused = self.proj(local_feat + global_feat)
        moe_out = self.moe(fused)

        return moe_out + identity


# 测试代码
if __name__ == "__main__":
    # 输入参数
    B, D, H, W = 4, 200, 15, 15
    num_classes = 16

    # 模型测试
    model = HyperSpectralClassifier(D, num_classes)
    x = torch.randn(B, D, H, W)
    out = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")  # 应输出 (4, 16)

    # 梯度测试
    loss = out.sum()
    loss.backward()
    print("梯度回传成功")
