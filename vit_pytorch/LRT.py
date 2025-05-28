from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
from einops.layers.torch import Reduce

sequencer_settings = {
    'S': [[4, 3,  8, 3], [1, 4, 8, 16], [48, 96, 96, 96], 3],
    'M': [[4, 3, 14, 3], [1, 4, 8, 16], [48, 96, 96, 96], 3],
    'L': [[3, 3, 5, 3], [1, 4, 8, 16], [48, 96, 96, 96], 3]
}

class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = kernel_size
        self.conv = nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, bias=False, groups=in_ch)
        self.sigmoid = nn.Sigmoid()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            # groups=1,
            bias=True
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        y = self.avg_pool(input)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        out = input * y.expand_as(input)
        out = self.point_conv(self.BN(out))
        out = self.Act1(out)
        # out = self.depth_conv(out)
        # out = self.Act2(out)
        return out

class HSISubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in LRT for HSI
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=True):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(HSISubNet, self).__init__()
        self.rnn_v = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True, bias=True)
        self.rnn_h = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                             batch_first=True, bias=True)
        self.rnn_z = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                             batch_first=True, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size * 6, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        B, S, H, W, C = x.shape
        v, _ = self.rnn_v(x.permute(0, 1, 3, 2, 4).reshape(-1, H, C))
        # v = self.dropout(v[0].squeeze())
        v = v.reshape(B, S, W, H, -1).permute(0, 1, 3, 2, 4)
        h, _ = self.rnn_h(x.reshape(-1, W, C))
        # h = self.dropout(h[0].squeeze())
        h = h.reshape(B, S, H, W, -1)
        z, _ = self.rnn_z(x.permute(0, 2, 3, 1, 4).reshape(-1, S, C))
        z = z.reshape(B, S, H, W, -1)
        x = torch.cat([v, h, z], dim=-1)
        y_1 = self.linear_1(x)
        return y_1

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # return self.fn(self.norm(x)) + x

        return self.fn(x) + x

class Sequencer3DBlock(nn.Module):
    def __init__(self, d_model, depth, hidden_d_model, expansion_factor=3, dropout=0.):
        super().__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model, nn.Sequential(
                    HSISubNet(d_model, hidden_d_model, d_model)
                )),
                PreNormResidual(d_model, nn.Sequential(
                    nn.Linear(d_model, d_model * expansion_factor),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * expansion_factor, d_model),
                    nn.Dropout(dropout)
                ))
            ) for _ in range(depth)]
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.model(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x

class Sequencer3D(nn.Module):
    def __init__(self, model_name: str = 'M', pretrained: str = None, num_classes: int = 16, in_channels=200, out_channel=45, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in sequencer_settings.keys(), f"Sequencer model name should be in {list(sequencer_settings.keys())}"
        depth, embed_dims, hidden_dims, expansion_factor = sequencer_settings[model_name]

        self.patch_size = [2, 1, 2, 1]

        self.kerner_size = [3, 3, 3, 1]

        self.patch_embed = SSConv(in_channels, out_channel, 7)
        self.hlstm = HSISubNet(out_channel, out_channel*2, out_channel)
        self.stage = len(depth)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                nn.Conv3d(1 if i == 0 else embed_dims[i - 1], embed_dims[i], kernel_size=self.kerner_size[i], padding=self.kerner_size[i]//2, stride=self.patch_size[i]),
                Sequencer3DBlock(embed_dims[i], depth[i], hidden_dims[i], expansion_factor, dropout=0.)
            ) for i in range(self.stage)]
        )

        self.mlp_head = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        x = x.squeeze()
        x = self.patch_embed(x)
        x = x.unsqueeze(1)
        embedding = self.stages(x)
        b, c, s, h, w = embedding.shape
        embedding = embedding.reshape(b, -1, h, w)
        out = self.mlp_head(embedding)
        return out

if __name__ == '__main__':
    model = Sequencer3D('L')
    # model = SequencerConv3D('L')
    x = torch.randn(2, 1, 200, 15, 15)
    # y = model(x)
    # print(y.shape)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')

    from thop import profile

    flops, params = profile(model, (x,))
    with torch.autograd.profiler.profile(enabled=True) as prof:
        out = model(x)

    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    repetitions = 100
    total_time = 0
    optimal_batch_size = 2
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * optimal_batch_size) / total_time
    print("FinalThroughput:", Throughput)