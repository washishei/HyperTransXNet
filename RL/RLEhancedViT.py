import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Normal
from einops import rearrange, repeat
import numpy as np
import matplotlib.pyplot as plt
from conv2d import ConvEtAl


# ------------------ 核心模块 ------------------
class MultiScaleSpatialRLAttn(nn.Module):
    """多尺度空间注意力（带膨胀卷积）"""

    def __init__(self, in_channels):
        super().__init__()
        self.pyramid = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 3,
                      dilation=2 ** i, padding=2 ** i)
            for i in range(3)
        ])
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels * 3, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
        self.critic = nn.AdaptiveAvgPool2d(1)
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        # 多尺度特征
        pyramid_feats = [conv(x) for conv in self.pyramid]
        fused = torch.cat(pyramid_feats, dim=1)

        # 生成注意力
        action_mean = self.actor(fused)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        attn_map = torch.sigmoid(dist.rsample())

        # 价值估计
        value = self.critic(x).flatten(1).mean(1, keepdim=True)

        return x * attn_map, value, dist, attn_map


class EnhancedSpectralRLAttn(nn.Module):
    """增强型光谱注意力（带位置编码）"""

    def __init__(self, embed_dim):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(1, 4096, embed_dim))
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.critic = nn.Linear(embed_dim, 1)
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        # 位置编码
        seq_len = x.size(1)
        x = x + self.pos_enc[:, :seq_len]

        # 生成注意力
        action_mean = self.actor(x).squeeze(-1)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        attn_weights = torch.softmax(dist.rsample(), dim=-1)

        # 特征聚合
        attended = torch.einsum('bs,bsd->bd', attn_weights, x)
        value = self.critic(attended)

        return attended, value, dist, attn_weights


class AdaptiveGateFusion(nn.Module):
    """自适应门控融合"""

    def __init__(self, spatial_dim, spectral_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(spatial_dim + spectral_dim, spatial_dim),
            nn.Sigmoid()
        )
        self.spectral_adapter = nn.Linear(spectral_dim, spatial_dim)

    def forward(self, spatial_feat, spectral_feat):
        # 维度对齐 [B, C, H, W] 和 [B, D]
        b, c, h, w = spatial_feat.shape
        spectral = self.spectral_adapter(spectral_feat).view(b, c, 1, 1)

        # 门控生成
        spatial_vec = F.adaptive_avg_pool2d(spatial_feat, 1).view(b, -1)
        gate_input = torch.cat([spatial_vec, spectral_feat], dim=1)
        gate = self.gate_net(gate_input).view(b, c, 1, 1)

        return gate * spatial_feat + (1 - gate) * spectral


# ------------------ 完整模型 ------------------
class FinalRLSST(nn.Module):
    """最终强化学习空谱Transformer"""

    def __init__(self, in_channels=200, num_classes=16,
                 spatial_dim=15, spectral_dim=128):
        super().__init__()

        self.stem = ConvEtAl(input_channels=in_channels, flatten=False)
        # 空间处理路径
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(512, 512, 1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.spatial_attn = MultiScaleSpatialRLAttn(512)

        self.stem_pool = nn.AvgPool2d(2)

        # 光谱处理路径
        self.spectral_encoder = nn.Linear(512, 512)
        self.spectral_attn = EnhancedSpectralRLAttn(512)

        # 融合与分类
        self.fusion = AdaptiveGateFusion(512, 512)
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 输入维度转换 [B,H,W,C] -> [B,C,H,W]
        x = x.squeeze()
        # print(x.shape)
        # x = x.permute(0, 3, 1, 2)
        x = self.stem(x)
        # print(x.shape)
        residual = self.stem_pool(x)
        # print(residual.shape)

        # first
        # 空间分支
        # print(x.shape)
        spatial = self.spatial_encoder(x)
        # print(x.shape)
        spatial_out, s_val, s_dist, s_map = self.spatial_attn(spatial)

        # 光谱分支
        spectral_in = rearrange(x, 'b c h w -> b (h w) c')
        # print(spectral_in.shape)
        spectral_feat = self.spectral_encoder(spectral_in)
        spectral_out, sp_val, sp_dist, sp_weights = self.spectral_attn(spectral_feat)

        # 融合与分类
        fused = self.fusion(spatial_out, spectral_out)
        # print(fused.shape)
        # fused = fused + residual
        fused = torch.cat([fused, residual], dim=1)
        logits = self.classifier(fused)

        return logits, (s_val, sp_val), (s_dist, sp_dist), (s_map, sp_weights)


# ------------------ 训练框架 ------------------
class AdvancedRLTrainer:
    def __init__(self, model, lr=3e-4, max_epoch=100):
        self.model = model
        self.scaler = GradScaler()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epoch, eta_min=1e-6)

        self.cls_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()

    def adaptive_reward(self, pred, target, s_map, sp_weights):
        # 分类准确率
        acc = (pred.argmax(1) == target).float()

        # 空间注意力熵（鼓励多样性）
        s_entropy = -torch.mean(s_map * torch.log(s_map + 1e-6))

        # 光谱稀疏度（鼓励专注）
        sp_sparsity = torch.mean(torch.sum(sp_weights ** 2, dim=1))

        return 0.5 * acc + 0.3 * (1 - s_entropy) + 0.2 * sp_sparsity

    @autocast()
    def train_step(self, data, target):
        # 前向传播
        # logits, values, dists = self.model(data)
        logits, values, dists, attn_data = self.model(data)
        s_map, sp_weights = attn_data  # 解包注意力权重

        # 计算奖励（传入注意力权重）
        reward = self.adaptive_reward(logits.detach(), target, s_map, sp_weights)

        # 计算奖励
        # reward = self.compute_reward(logits, target)

        # 分类损失
        cls_loss = self.cls_criterion(logits, target)

        # 价值损失
        value_loss = sum([self.value_criterion(v.squeeze(), reward) for v in values])

        # 策略损失
        policy_loss = 0

        for dist in dists:  # dists是(s_dist, sp_dist)两个Normal对象
            # 调整reward维度以匹配分布均值
            reward_reshaped = reward.view(-1, *([1] * (dist.mean.dim() - 1)))
            advantage = reward_reshaped - dist.mean.detach()

            # 计算策略损失
            policy_loss += -dist.log_prob(advantage).mean()

                    # 总损失
        total_loss = cls_loss + 0.4 * value_loss + 0.6 * policy_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

# ----------------- 训练框架 ------------------
# class AdvancedRLTrainer:
#     def __init__(self, model, lr=3e-4, max_epoch=100):
#         self.model = model
#         self.scaler = GradScaler()
#         self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#         self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epoch, eta_min=1e-6)
#         self.cls_criterion = nn.CrossEntropyLoss()
#         self.value_criterion = nn.MSELoss()
#
#     def adaptive_reward(self, pred, target, s_map, sp_weights):
#         acc = (pred.argmax(1) == target).float()
#         s_entropy = -torch.mean(s_map * torch.log(s_map + 1e-6))
#         sp_sparsity = torch.mean(torch.sum(sp_weights ** 2, dim=1))
#         return 0.5 * acc + 0.3 * (1 - s_entropy) + 0.2 * sp_sparsity
#
#     @autocast()
#     def train_step(self, data, target):
#         self.optimizer.zero_grad()
#
#         # 前向传播并解包
#         logits, values, dists, attn_data = self.model(data)
#         s_map, sp_weights = attn_data
#
#         # 计算奖励
#         reward = self.adaptive_reward(logits.detach(), target, s_map, sp_weights)
#
#         # 分类损失
#         cls_loss = self.cls_criterion(logits, target)
#
#         # 价值损失（空间和光谱）
#         value_loss = 0
#         for v in values:  # values是(s_val, sp_val)
#             value_loss += self.value_criterion(v.squeeze(), reward)
#
#         # 策略梯度损失（空间和光谱分布）
#         policy_loss = 0
#         for dist_group in dists:  # dists是(s_dist, sp_dist)
#             # print(dist_group)
#             for dist in dist_group:
#                 advantage = reward - dist.mean.detach()
#                 policy_loss += -dist.log_prob(advantage).mean()
#
#         # 总损失
#         total_loss = cls_loss + 0.4 * value_loss + 0.6 * policy_loss
#
#         # 反向传播
#         self.scaler.scale(total_loss).backward()
#         self.scaler.step(self.optimizer)
#         self.scaler.update()
#         self.scheduler.step()
#
#         return total_loss.item()


# ------------------ 可视化与测试 ------------------
class AttentionVisualizer:
    @staticmethod
    def visualize(attn_data, rgb_slice=(30, 20, 10)):
        s_map, sp_weights = attn_data

        plt.figure(figsize=(18, 6))

        # 空间注意力
        plt.subplot(131)
        plt.imshow(s_map[0].cpu().detach().mean(0).squeeze(), cmap='hot')
        plt.title('Spatial Attention')
        plt.colorbar()

        # 光谱注意力
        plt.subplot(132)
        plt.plot(sp_weights[0].cpu().detach().numpy())
        plt.title('Spectral Attention Weights')

        # RGB合成
        plt.subplot(133)
        rgb = np.stack([rgb_slice], axis=-1)
        plt.imshow(rgb / rgb.max())
        plt.title('RGB Composite')

        plt.tight_layout()
        plt.show()


# ------------------ 量化支持 ------------------
def quantize_model(model):
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )


# ------------------ 测试用例 ------------------
if __name__ == "__main__":
    # 配置参数
    B, H, W, C = 8, 15, 15, 200
    num_classes = 16

    # 初始化
    model = FinalRLSST(in_channels=C, num_classes=num_classes)
    trainer = AdvancedRLTrainer(model)
    visualizer = AttentionVisualizer()

    # 生成测试数据
    dummy_data = torch.randn(B, C, H, W)
    dummy_labels = torch.randint(0, num_classes, (B,))

    # 训练循环
    for epoch in range(5):
        loss = trainer.train_step(dummy_data, dummy_labels)
        print(f"Epoch {epoch + 1} | Loss: {loss:.4f} | LR: {trainer.scheduler.get_last_lr()[0]:.2e}")

        # 可视化
        if epoch % 2 == 0:
            with torch.no_grad():
                _, _, _, attn_data = model(dummy_data[:1])
                # visualizer.visualize(attn_data)

    # 量化测试
    quant_model = quantize_model(model)
    test_logits, _, _, _ = quant_model(dummy_data[:1])
    print("\n量化后测试logits:", test_logits.argmax(dim=1))