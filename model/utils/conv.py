import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batch_norm_2d = nn.BatchNorm2d(num_features=out_channels)
        self.siLu = F.silu
        self.to(device)

    def forward(self, x: torch.Tensor):
        x = self.conv2d(x)
        x = self.batch_norm_2d(x)
        x = self.siLu(x)

        return x