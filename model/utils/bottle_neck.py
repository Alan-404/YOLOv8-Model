import torch
import torch.nn as nn
from .conv import Conv

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class BottleNeck(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, shortcut: bool = True):
        super().__init__()
        self.conv_1 = Conv(in_channels=channels, out_channels=channels//2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = Conv(in_channels=channels//2, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.shortcut = shortcut

        self.to(device)
    
    def forward(self, x: torch.Tensor):
        conv_output = self.conv_1(x)
        conv_output = self.conv_2(conv_output)

        if self.shortcut:
            x = x + conv_output

        return x