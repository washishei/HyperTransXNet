import torch
import torch.nn as nn
import torch.fft
import collections
import math
from einops import rearrange
from .QDASA import QDASA



class ConvEtAl(torch.nn.Module):
    def __init__(self, input_channels, flatten=True):
        super(ConvEtAl, self).__init__()
        self.feature_size = 64
        self.name = "conv4"

        self.layer1 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(input_channels, self.feature_size, kernel_size=3, stride=1, padding=1, bias=False)),
          # ('qconv', QuaternionConv(self.feature_size, self.feature_size, kernel_size=3, stride=1, padding=1, )),
          ('bn',      nn.BatchNorm2d(self.feature_size)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer2 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(self.feature_size, self.feature_size * 2, kernel_size=3, stride=1, padding=1, bias=False)),
          # ('qconv', QuaternionConv(self.feature_size * 2, self.feature_size, kernel_size=3, stride=1, padding=1, )),
          ('bn',      nn.BatchNorm2d(self.feature_size * 2)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer3 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(self.feature_size * 2, self.feature_size * 4, kernel_size=3, stride=1, padding=1, bias=False)),
          # ('qconv', QuaternionConv(self.feature_size * 4, self.feature_size, kernel_size=3, stride=1, padding=1, )),
          ('bn',      nn.BatchNorm2d(self.feature_size * 4)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2)),
          ('glbpool', nn.AdaptiveAvgPool2d(1))
        ]))

        self.layer4 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(self.feature_size * 4, self.feature_size * 4, kernel_size=1, stride=1, padding=0, bias=False)),
          ('bn',      nn.BatchNorm2d(self.feature_size * 4)),
          ('relu',    nn.ReLU()),
          # ('avgpool', nn.AvgPool2d(kernel_size=4))
          ('glbpool', nn.AdaptiveAvgPool2d(1))
        ]))

        self.is_flatten = flatten
        self.flatten = nn.Flatten()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        # h = self.layer4(h)
        # print(h.shape)
        if(self.is_flatten): h = self.flatten(h)
        return h

class QuaternionConv(nn.Module):
    """四元数卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        assert in_channels % 4 == 0, "输入通道数必须是4的倍数"
        self.conv_r = nn.Conv2d(in_channels // 4, out_channels, kernel_size, stride, padding)
        self.conv_i = nn.Conv2d(in_channels // 4, out_channels, kernel_size, stride, padding)
        self.conv_j = nn.Conv2d(in_channels // 4, out_channels, kernel_size, stride, padding)
        self.conv_k = nn.Conv2d(in_channels // 4, out_channels, kernel_size, stride, padding)
        self.proj_conv = nn.Conv2d(out_channels * 4, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # print(x.shape)
        x = x.view(B, 4, C // 4, H, W)
        # print(x.shape)
        r, i, j, k = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        out_r = self.conv_r(r) - self.conv_i(i) - self.conv_j(j) - self.conv_k(k)
        out_i = self.conv_r(i) + self.conv_i(r) + self.conv_j(k) - self.conv_k(j)
        out_j = self.conv_r(j) - self.conv_i(k) + self.conv_j(r) + self.conv_k(i)
        out_k = self.conv_r(k) + self.conv_i(j) - self.conv_j(i) + self.conv_k(r)

        output = torch.cat([out_r, out_i, out_j, out_k], dim=1)
        # print(output.shape)
        output = self.proj_conv(output)

        return output


class QMambaBlock(nn.Module):
    """四元数曼巴核心模块"""

    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        assert dim % 4 == 0, "输入维度必须是4的倍数"

        # 四元数特征提取
        self.q_conv = nn.Conv2d(dim, dim, 3, padding=1)

        # 状态空间模型参数
        self.dt_proj = nn.Linear(dim, dim)
        self.A = nn.Parameter(torch.randn(dim, d_state))
        self.D = nn.Parameter(torch.randn(dim))

        # 动态谱权门控
        self.spectral_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Linear(dim * 4, dim),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(),
            # nn.Linear(dim, dim),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def ssm_scan(self, x):
        """状态空间扫描"""
        B, C, H, W = x.shape
        x_seq = rearrange(x, "b c h w -> b (h w) c")  # [B, L, C]

        dt = torch.sigmoid(self.dt_proj(x_seq))  # [B, L, C]
        A = -torch.exp(self.A)  # [C, d_state]
        D = torch.exp(self.D)  # [C]

        # 状态空间计算
        h = torch.zeros(B, C, self.d_state).to(x.device)
        outputs = []
        for t in range(x_seq.size(1)):
            current_dt = dt[:, t, :].unsqueeze(-1)  # [B, C, 1]
            current_x = x_seq[:, t, :].unsqueeze(-1)  # [B, C, 1]

            h = h * torch.exp(A * current_dt) + current_x
            # print((h @ A.T).shape, D.shape, x_seq.shape,)

            # h_1 = (h @ A.T).squeeze(-1)
            # print(h_1.shape)
            # h_2 = D * x_seq[:, t]
            # print(h_2.shape)
            y = (h * A).sum(dim=-1) + D * x_seq[:, t]
            # print(y.shape)
            outputs.append(y)

        return torch.stack(outputs, dim=1).view(B, C, H, W)

    def forward(self, x):
        identity = x
        # print(x.shape)
        # 四元数特征提取
        x = self.q_conv(x)
        # print(x.shape)

        # 动态谱权
        gate = self.spectral_gate(x)
        # print(gate.shape)
        x = x * gate
        # print(x.shape)
        # 状态空间扫描
        x = self.ssm_scan(x)
        # print(x.shape)
        return x + identity

class QMambaNet(nn.Module):
    """完整分类网络"""
    def __init__(self, in_channels, num_classes, dim=256):
        super().__init__()
        self.dim = dim
        self.stem_channel = nn.Sequential(
            nn.Conv2d(in_channels, dim, 1),
            # nn.BatchNorm2d(dim),
            # nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1, stride=1, groups=dim),
            nn.Conv2d(dim, dim, 3, 1, 1),
            # nn.ReLU()
        )
        # self.stem_channel = nn.Sequential(
        #     nn.Conv2d(in_channels, dim, 3, padding=1, stride=1,),
        #     # QuaternionConv(dim, dim // 4, 3, padding=1, stride=1),
        #     nn.ReLU(),
        #     # QuaternionConv(dim, dim // 4, 1)
        # )
        self.multi_fusion = QDASA(dim)
        self.stem_convnet = ConvEtAl(dim)
        self.start_convnet = ConvEtAl(in_channels)
        self.pos_enc = nn.Parameter(torch.randn(2))
        self.stem = nn.Sequential(
            # nn.Conv2d(dim, dim, 3, 1, 1),
            QuaternionConv(dim, 64, 3, padding=1),
            # nn.ReLU(),
            nn.AvgPool2d(2),
            # nn.Conv2d(dim, dim, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.blocks = nn.Sequential(
            # nn.Conv2d(dim, dim, 3, 1, 1),
            QMambaBlock(dim),
            nn.AvgPool2d(2),
            # nn.Conv2d(dim, dim, 1, 1, ),

            # QMambaBlock(dim),
            # nn.AvgPool2d(2),
            # QMambaBlock(dim),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Sequential(
            # nn.Linear(dim, dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = x.squeeze()
        # x_start = x
        # x_start = self.start_convnet(x_start)
        x = self.stem_channel(x)
        # print(f"测试通过！输入通道作用之后，输出形状：{tuple(x.shape)}")
        x = self.multi_fusion(x)
        x_conv = x
        # x_qmam = x
        x_conv = self.stem_convnet(x_conv)
        # print(x_conv.shape)
        # print(f"测试通过！光谱-频域-空间-正常特征融合之后，输出形状：{tuple(x.shape)}")
        # x_qmam = self.stem(x_qmam)          # [B,64,H/2,W/2]
        # x = x + identity
        # print(f"测试通过！四元素前期融合之后，输出形状：{tuple(x.shape)}")
        # x_qmam = self.blocks(x_qmam)        # [B,64,1,1]
        # print(f"测试通过！四元素mamba作用之后，输出形状：{tuple(x.shape)}")
        # logits = self.pos_enc[0] * x_qmam.flatten(1) + self.pos_enc[1] * x_conv    # [B,10]
        # logits = self.head(logits)
        ### second
        # logits = self.pos_enc[0] * x_start + self.pos_enc[1] * x_conv    # [B,10]
        # logits = self.head(logits)
        logits = self.head(x_conv)
        # print(f"测试通过！最后层作用之后，输出形状：{tuple(logits.shape)}")
        return logits

### 测试代码 ###
if __name__ == "__main__":
    # 输入配置 [Batch, Channels, Height, Width]
    dim = 200  # 必须为4的倍数
    h = w = 64
    batch_size = 4

    # 初始化模型
    model = QMambaNet(in_channels=dim, num_classes=16)

    # 生成测试数据
    x = torch.randn(batch_size, dim, h, w)

    # 前向传播
    try:
        output = model(x)
        assert output.shape == x.shape
        print(f"测试通过！输入输出形状一致：{tuple(output.shape)}")
        print(f"总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    except Exception as e:
        print(f"运行错误：{str(e)}")