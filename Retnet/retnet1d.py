import torch
import torch.nn as nn
from torch.nn import init
import math
from .retention import MultiScaleRetention


class BoulchEtAl(nn.Module):
    """
    Autoencodeurs pour la visualisation d'images hyperspectrales
    A.Boulch, N. Audebert, D. Dubucq
    GRETSI 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, planes=15):
        super(BoulchEtAl, self).__init__()
        self.input_channels = input_channels
        self.aux_loss_weight = 0.1

        encoder_modules = []
        n = input_channels
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            # print(x.size())
            while n > 1:
                # print("---------- {} ---------".format(n))
                if n == input_channels:
                    p1, p2 = 1, 2 * planes
                elif n == input_channels // 2:
                    p1, p2 = 2 * planes, planes
                else:
                    p1, p2 = planes, planes
                encoder_modules.append(nn.Conv1d(p1, p2, 3, padding=1))
                x = encoder_modules[-1](x)
                # print(x.size())
                encoder_modules.append(nn.MaxPool1d(2))
                x = encoder_modules[-1](x)
                # print(x.size())
                encoder_modules.append(nn.ReLU(inplace=True))
                x = encoder_modules[-1](x)
                # print(x.size())
                encoder_modules.append(nn.BatchNorm1d(p2))
                x = encoder_modules[-1](x)
                # print(x.size())
                n = n // 2

            encoder_modules.append(nn.Conv1d(planes, n_classes, 3, padding=1))
        encoder_modules.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_modules)
        self.features_sizes = self._get_sizes()
        #
        # self.classifier = nn.Linear(self.features_sizes, n_classes)
        # self.regressor = nn.Linear(self.features_sizes, input_channels)
        self.apply(self.weight_init)

    def _get_sizes(self):
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            x = self.encoder(x)
            _, c, w = x.size()
        return c * w

    def forward(self, x):
        # x = x.unsqueeze(1)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        # x = x.view(-1, self.features_sizes)
        # print(x.shape)
        x = x.permute(0, 2, 1)
        return x
class RetNet(nn.Module):
    def __init__(self, num_class, channels, layers, hidden_dim, ffn_size, heads, double_v_dim=False):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

        kernel_size = math.ceil(channels / 9)

        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        # self.encoder = nn.Sequential(
        #     nn.Linear(channels, hidden_dim),
        #     # nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Linear(hidden_dim, num_class) if num_class > 0 else nn.Identity()
    
    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        X = X.squeeze()
        # print(X.shape)
        X = self.encoder(X)
        X = X.view(-1, self.hidden_dim)
        X = X.unsqueeze(1)
        # print(X.shape)
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y
        X = torch.flatten(X, 1)
        # print(X.shape)
        X = self.head(X)
        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        s_ns = []
        for i in range(self.layers):
            # list index out of range
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n
        
        return x_n, s_ns
    
    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        X: (batch_size, sequence_length, hidden_size)
        r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        r_is = []
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
            y_i = o_i + x_i
            r_is.append(r_i)
            x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i
        
        return x_i, r_is



def RetNet_tiny():
    layers = 12
    hidden_dim = 768
    ffn_size = 4096
    heads = 12
    model = RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim=True)
    return model


def RetNet_small_V1():
    layers = 16
    hidden_dim = 2048
    ffn_size = 4096
    heads = 16
    model = RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim=True)
    return model


def RetNet_small_V2():
    layers = 24
    hidden_dim = 2048
    ffn_size = 4096
    heads = 16
    model = RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim=True)
    return model


def RetNet_large():
    layers = 24
    hidden_dim = 2560
    ffn_size = 4096
    heads = 16
    model = RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim=True)
    return model


def RetNet_huge():
    layers = 24
    hidden_dim = 4096
    ffn_size = 4096
    heads = 16
    model = RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim=True)
    return model


if __name__ == '__main__':
    img_size = 1
    x = torch.rand(64, 256)
    layers = 12
    hidden_dim = 768
    ffn_size = 4096
    heads = 12
    model = RetNet(10, 256, layers, hidden_dim, ffn_size, heads, double_v_dim=True)

    from thop import profile
    flops, params = profile(model, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')