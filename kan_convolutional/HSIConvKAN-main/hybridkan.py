import torch.nn as nn
import torch.nn.functional as F
from ConvKAN import ConvKAN
from ConvKAN3D import effConvKAN3D
from fast_kan import FastKAN as KAN
import torch


class Hybridkan(nn.Module):
    # Define the architecture of the network
    def __init__(self, input_channels, n_classes, patch_size):
        super(Hybridkan, self).__init__()

        self.in_chs = 15
        self.patch_size = patch_size
        self.ConvKAN1 = effConvKAN3D(in_channels=15, out_channels=8, kernel_size=1)
        self.ConvKAN2 = effConvKAN3D(in_channels=8, out_channels=16, kernel_size=1)
        self.ConvKAN3 = effConvKAN3D(in_channels=16, out_channels=32, kernel_size=1)

        self.ConvKAN4 = ConvKAN(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, version="Fast")
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

        # 5*5 from image dimension
        # self.KAN1 =  KAN([4 * 4 * 4 , 64, 32])
        self.KAN1 = KAN([64, 32, n_classes])

    # Set the flow of data through the network for the forward pass
    # x represents the data
    def forward(self, x):
        # F.relu is the rectified-linear activation function

        x = self.ConvKAN1(x)
        x = self.ConvKAN2(x)
        x = self.ConvKAN3(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4])
        x = self.pool(self.ConvKAN4(x))

        # flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        x = self.KAN1(x)
        # x = self.KAN2(x)
        # x = self.KAN3(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    img_size = 15
    x = torch.rand(64, 1, 15, img_size, img_size)
    model = Hybridkan(15, 10, patch_size=15)
    y = model(x)
    print(y.shape)
    model.eval()
    from thop import profile
    flops, params = profile(model, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')