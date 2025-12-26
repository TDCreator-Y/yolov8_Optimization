import torch.nn as nn
from torchvision.ops import DeformConv2d

from ultralytics.nn.modules.block import C2f  # 原版C2f


class DCNBottleneck(nn.Module):
    """DCN Bottleneck that keeps channels unchanged. 输入输出通道固定为 c（=C2f内部hidden通道 self.c）.
    """

    def __init__(self, c: int, shortcut: bool = True, k: int = 3):
        super().__init__()
        p = k // 2

        # offset: 2*k*k channels
        self.offset = nn.Conv2d(c, 2 * k * k, kernel_size=k, stride=1, padding=p)

        # deformable conv: c -> c
        self.dcn = DeformConv2d(c, c, kernel_size=k, stride=1, padding=p, bias=False)

        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

        self.add = shortcut  # C2f里通常 shortcut 控制残差

    def forward(self, x):
        y = self.offset(x)
        y = self.dcn(x, y)
        y = self.act(self.bn(y))
        return x + y if self.add else y


class DCN_C2f(C2f):
    """继承官方 C2f，不改 split/concat/cv1/cv2 的所有逻辑， 只替换内部 self.m（原本是 Bottleneck）为 DCNBottleneck。.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)

        # 关键：C2f内部真正走的通道是 self.c，不是 c2
        # 用DCNBottleneck替换原来的Bottleneck
        self.m = nn.ModuleList(DCNBottleneck(self.c, shortcut=shortcut) for _ in range(n))
