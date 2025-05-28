from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, inputs):
        '''
        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t], # (B, C, H, W)
                                  hx], dim=1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for idx, params in enumerate(config.encoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
        elif type == 'convlstm':
            layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        outputs = [x]
        for layer in self.layers:
            if 'conv_' in layer:
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
            x = getattr(self, layer)(x)
            if 'conv_' in layer: x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
            if 'convlstm' in layer: outputs.append(x)
        return outputs

# class ConvLSTM(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.layers = []
#         for idx, params in enumerate(config.encoder):
#             setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
#             self.layers.append(params[0]+'_'+str(idx))
#
#     def _make_layer(self, type, activation, in_ch, out_ch,):
#         layers = []
#         # if type == 'conv':
#         #     layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
#         #     layers.append(nn.BatchNorm2d(out_ch))
#         #     if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
#         #     elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
#         elif type == 'convlstm':
#             layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         '''
#         :param x: (B, S, C, H, W)
#         :return:
#         '''
#         outputs = [x]
#         for layer in self.layers:
#             if 'conv_' in layer:
#                 B, S, C, H, W = x.shape
#                 x = x.view(B*S, C, H, W)
#             x = getattr(self, layer)(x)
#             if 'conv_' in layer: x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
#             if 'convlstm' in layer: outputs.append(x)
#         return outputs


class Config:
    # (type, activation, in_ch, out_ch, kernel_size, padding, stride)
    encoder = [('conv', 'leaky', 1, 16, 3, 1, 2),
             ('convlstm', '', 16, 16, 3, 1, 1),]
             # ('conv', 'leaky', 16, 32, 3, 1, 2),
             # ('convlstm', '', 32, 32, 3, 1, 1),
             # ('conv', 'leaky', 32, 64, 3, 1, 2),
             # ('convlstm', '', 64, 64, 3, 1, 1)]

class ConvLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)

    def forward(self, x):
        print(x.shape)
        x = self.encoder(x)
        return x

if __name__ == '__main__':
    from thop import profile
    model = ConvLSTM(Config)
    flops, params = profile(model, inputs=(torch.Tensor(4, 10, 1, 64, 64),))
    print(flops / 1e9, params / 1e6)