import torch
import torch.nn as nn

from model.utils.c2f import C2F
from model.utils.conv import Conv
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class Head(nn.Module):
    def __init__(self, w: float, d:float, r: float, kernel_size: int = 3, stride: int = 2, padding: int = 1):
        super().__init__()
        self.upsample_1 = nn.Upsample(scale_factor=(0.5, 0.5))
        
        self.concat_1 = torch.concat

        self.c2f_1 = C2F(in_channels=int(512*w*(1+r)), out_channels=int(512*w), n_bottles=3*d, shortcut=False)

        self.upsample_2 = nn.Upsample(scale_factor=(2, 2, 0.5))

        self.concat = torch.concat

        self.c2f_2 = C2F(in_channels=int(512*w), out_channels=int(256*w), n_bottles=3*d, shortcut=False)

        self.conv_1 = Conv(in_channels=int(256*w), out_channels=int(256*w), kernel_size=kernel_size, stride=stride, padding=padding)

        self.c2f_3 = C2F(in_channels=int((512+256)*w), out_channels=int(512*w), n_bottles=3*d, shortcut=False)

        self.conv_2 = Conv(in_channels=int(512*w), out_channels=int(512*w), kernel_size=kernel_size, stride=stride, padding=padding)

        self.c2f_4 = C2F(in_channels=int(512*w*(1+r)), out_channels=int(512*w), n_bottles=3*d, shortcut=False)

        self.to(device)

    def forward(self, backbone: list):
        x = self.upsample_1(backbone[2])
        x = self.concat([x, backbone[1]])

        x = c2f_1_output = self.c2f_1(x)

        x = x.unsqueeze(1)
        x = self.upsample_2(x)
        x = self.squeeze(1)

        x = self.concat([x, backbone[0]], dim=1)

        x = detect_1 = self.c2f_2(x)

        x = self.conv_1(x)

        x = self.concat([x, c2f_1_output], dim=1)

        x = detect_2 = self.c2f_3(x)

        x = self.conv_2(x)

        x = self.concat([x, backbone[2]], dim=1)

        x = detect_3 = self.c2f_4(x)

        return x

