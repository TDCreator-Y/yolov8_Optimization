# ultralytics/nn/modules/coordatt.py
import torch
import torch.nn as nn


class h_swish(nn.Module):
    """Hard-Swish activation used in many CoordAtt implementations."""

    def forward(self, x):
        return x * torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


class h_sigmoid(nn.Module):
    """Hard-Sigmoid activation."""

    def forward(self, x):
        return torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


class CoordAtt(nn.Module):
    """Coordinate Attention (CoordAtt) Paper: "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021)
    Matches the description in your target paper Section 2.3.3: - Pool along H and W separately (1D encodings) -
    Concat -> 1x1 conv + BN + activation - Split -> two 1x1 conv -> sigmoid - Multiply back to input feature map.
    """

    def __init__(self, inp: int, reduction: int = 32, use_hs: bool = True):
        """
        Args:
            inp: input channels C
            reduction: channel reduction ratio (commonly 32)
            use_hs: whether to use hard-swish/hard-sigmoid (recommended).
        """
        super().__init__()
        self.inp = inp
        self.reduction = reduction

        # In CoordAtt, the intermediate channels mip is usually max(8, C//reduction)
        mip = max(8, inp // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B,C,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B,C,1,W)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)

        self.act = h_swish() if use_hs else nn.SiLU()

        # Two separate convs produce attention weights for H and W
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0, bias=True)

        self.sigmoid = h_sigmoid() if use_hs else nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) returns: (B, C, H, W).
        """
        _b, _c, h, w = x.shape

        # 1) 1D global pooling along H and W directions
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x)  # (B, C, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, W, 1)  so we can concat on "spatial" dim

        # 2) concat along spatial dim (H + W)
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)

        # 3) shared 1x1 conv + BN + activation
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 4) split and generate two attention maps
        y_h, y_w = torch.split(y, [h, w], dim=2)  # y_h:(B,mip,H,1), y_w:(B,mip,W,1)
        y_w = y_w.permute(0, 1, 3, 2)  # (B,mip,1,W)

        a_h = self.sigmoid(self.conv_h(y_h))  # (B,C,H,1)
        a_w = self.sigmoid(self.conv_w(y_w))  # (B,C,1,W)

        # 5) apply attention
        out = x * a_h * a_w
        return out
