import torch
import torch.nn as nn

class Split(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        channels = x.size(1)
        center = channels//2
        return x[:, :center, :, :], x[:, center:, :, :]