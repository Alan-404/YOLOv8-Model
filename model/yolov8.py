import torch
import torch.nn as nn
from model.components.backbone import Backbone
from model.components.head import Head

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class YOLOv8Model(nn.Module):
    def __init__(self, d: float, w: float, r: float):
        super().__init__()
        self.backbone = Backbone(d=d, w=w, r=r)
        self.head = Head(d=d, w=w, r=r)

        self.to(device)

    def forward(self, x: torch.Tensor):
        backbone_outputs = self.backbone(x)
        outputs = self.head(backbone_outputs)

        return outputs

