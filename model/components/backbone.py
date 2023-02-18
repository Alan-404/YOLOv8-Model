import torch
import torch.nn as nn
from model.utils.conv import Conv
from model.utils.c2f import C2F
from model.utils.sppf import SPPF

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class Backbone(nn.Module):
    def __init__(self, d: float, w: float, r:float, kernel_size: int = 3, stride: int = 2, padding: int = 1):
        super().__init__()
        self.conv_1 = Conv(in_channels=3, out_channels=int(64*w), kernel_size=kernel_size, stride=stride, padding=padding)

        self.conv_2 = Conv(in_channels=int(64*w), out_channels=int(128*w), kernel_size=kernel_size, stride=stride, padding=padding)

        self.c2f_1 = C2F(in_channels=int(128*w), out_channels=int(128*w), n_bottles=3*d,shortcut=True)

        self.conv_3 = Conv(in_channels=int(128*w), out_channels=int(256*w), kernel_size=kernel_size, stride=stride, padding=padding)

        self.c2f_2 = C2F(in_channels=int(256*w), out_channels=int(256*w), n_bottles=6*d, shortcut=True)

        self.conv_4 = Conv(in_channels=int(256*w), out_channels=int(512*w), kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.c2f_3 = C2F(in_channels=int(512*w), out_channels=int(512*w), n_bottles=6*d, shortcut=True)

        self.conv_5 = Conv(in_channels=int(512*w), out_channels=int(512*w*r), kernel_size=kernel_size, stride=stride, padding=padding)

        self.c2f_4 = C2F(in_channels=int(512*w*r), out_channels=int(512*w*r), n_bottles=3*d, shortcut=True)

        self.sppf = SPPF(channels=int(512*w*r))

        self.to(device)

    def forward(self, x: torch.Tensor):
        output = list()

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.c2f_1(x)
        x = self.conv_3(x)
        x = self.c2f_2(x)
        output.append(x)

        x = self.conv_4(x)
        x = self.c2f_3(x)
        output.append(x)

        x = self.conv_5(x)
        x = self.c2f_4(x)
        print(x.size())
        x = self.sppf(x)
        output.append(x)

        return output