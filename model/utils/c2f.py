import torch
import torch.nn as nn
import math

from .split import Split
from .conv import Conv
from .bottle_neck import BottleNeck

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class C2F(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_bottles: float, shortcut: bool):
        super().__init__()
        n_bottles = math.ceil(n_bottles)
        self.conv = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        
        self.split = Split()
        
        self.bottles = [BottleNeck(channels=(out_channels//2), shortcut=shortcut) for _ in range(n_bottles)]

        self.concat = torch.concat

        self.final_conv = Conv(in_channels=(out_channels//2)*(n_bottles+2), out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.to(device)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        out1, out2 = self.split(x)

        bottles_list = list()
        
        bottles_list.append(out1)
        bottles_list.append(out2)

        for bottle in self.bottles:
            out2 = bottle(out2)
            bottles_list.append(out2)


        x = self.concat(bottles_list, dim=1)
        x = self.final_conv(x)

        return x



