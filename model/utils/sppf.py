import torch
import torch.nn as nn
from .conv import Conv
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
class SPPF(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = Conv(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.max_poo2d_1 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.max_poo2d_2 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.max_poo2d_3 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

        self.concat = torch.concat

        self.final_conv = Conv(in_channels=channels*4, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.to(device)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        maxpool_1 = self.max_poo2d_1(x)
        maxpool_2 = self.max_poo2d_2(maxpool_1)
        maxpool_3 = self.max_poo2d_3(maxpool_2)

        concat = self.concat([x, maxpool_1, maxpool_2, maxpool_3], dim=1)
        x = self.final_conv(concat)

        return x

