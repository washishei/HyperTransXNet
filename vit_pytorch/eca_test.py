import torch
import torch.nn as nn


class EfficientGlobalLocalizationAttention(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(EfficientGlobalLocalizationAttention, self).__init__()
        self.stem_conv = nn.Conv2d(channel, 256, kernel_size=3, padding=1, bias=False)
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(256, 256, kernel_size=kernel_size, padding=self.pad, groups=256, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_z = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.gn = nn.GroupNorm(16, 256)
        self.sigmoid = nn.Sigmoid()
        self.final_conv = nn.Conv2d(256, channel, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.stem_conv(x)
        b, c, h, w = x.size()

        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)

        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        x_z = self.avg_pool(x)
        x_z = self.sigmoid(self.conv_z(x_z.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))

        # print(x_h.shape, x_w.shape, x_z.shape)
        # 在两个维度上应用注意力
        x = x * x_z * x_h * x_w + x
        x = self.final_conv(x)
        return x


# 示例用法 ELABase(ELA-B)
if __name__ == "__main__":
    # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
    dummy_input = torch.randn(2, 200, 32, 32)

    # 初始化模块
    ela = EfficientGlobalLocalizationAttention(channel=dummy_input.size(1), kernel_size=7)

    # 前向传播
    output = ela(dummy_input)
    # 打印出输出张量的形状，它将与输入形状相匹配。
    print(f"输出形状: {output.shape}")