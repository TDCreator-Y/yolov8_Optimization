import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class DCNConv(nn.Module):
    """Deformable Conv v2 (no modulation, aligned with most YOLO-DCN papers)"""

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1):
        super().__init__()
        self.offset = nn.Conv2d(
            c1,
            2 * k * k,
            kernel_size=k,
            stride=s,
            padding=p
        )
        self.dcn = DeformConv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        offset = self.offset(x)
        x = self.dcn(x, offset)
        return self.act(self.bn(x))
