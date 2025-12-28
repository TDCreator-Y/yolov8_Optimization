# SCC增强的C2f模块实现
# ===================

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C2f

# 处理直接运行和包导入两种情况
try:
    # 当作为包的一部分导入时
    from .scc_attention import SCC_Bottleneck, SCC_Attention
except ImportError:
    # 当直接运行或从外部导入时
    from scc_attention import SCC_Bottleneck, SCC_Attention


class SCC_C2f(C2f):
    """
    带SCC注意力增强的C2f (CSP bottleneck with 2 convolutions)

    将标准Bottleneck块替换为SCC增强版本，
    用于YOLOv8主干网络层中更好的特征表示。

    架构特点：
        - 继承标准C2f的CSP设计理念
        - 集成SCC注意力机制进行特征增强
        - 保持与YOLOv8的完全兼容性
        - 增强的特征提取和表示能力

    技术优势：
        - 注意力增强：通过SCC提升特征质量
        - 结构保持：维持原有的网络拓扑
        - 性能提升：在计算效率基础上增强准确性
        - 即插即用：无缝集成到现有架构
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        初始化SCC增强的C2f模块

        参数：
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): SCC瓶颈块的数量，默认1
            shortcut (bool): 是否使用残差连接，默认False
            g (int): 卷积分组数，默认1
            e (float): 隐藏层通道扩展比例，默认0.5

        网络结构：
            输入 -> 1x1卷积分支 -> 分割为两路 -> SCC瓶颈块处理 -> 拼接 -> 1x1输出卷积
            |                        |                      |
            +------------------------+----------------------+

        设计理念：
            - CSP架构：减少计算量，提高特征重用
            - SCC增强：每个瓶颈块都包含注意力机制
            - 模块化设计：便于集成和扩展
        """
        # 调用父类初始化基础结构
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)

        # 核心创新：将标准瓶颈块替换为SCC增强版本
        self.m = nn.ModuleList(
            SCC_Bottleneck(self.c, self.c, shortcut=shortcut, g=g, e=1.0, use_scc=True)
            for _ in range(n)
        )

    def forward(self, x):
        """
        SCC增强的C2f前向传播

        参数：
            x (torch.Tensor): 输入特征图，形状为[B, C1, H, W]

        返回值：
            torch.Tensor: 输出特征图，形状为[B, C2, H, W]

        处理流程：
            1. 输入预处理和分支分割（CSP结构）
            2. SCC瓶颈块的并行几何建模和注意力增强
            3. 多分支特征拼接融合
            4. 最终输出卷积

        技术特点：
            - 分支处理：并行计算提高效率
            - 注意力增强：SCC提升每个分支的特征质量
            - 特征融合：有效的多路信息整合
            - 梯度优化：残差连接保持梯度流动
        """
        # CSP分支分割：将输入特征分为两个分支进行并行处理
        y = list(self.cv1(x).chunk(2, 1))

        # 应用SCC增强的瓶颈块
        # 每个瓶颈块都包含注意力机制进行特征增强
        for m in self.m:
            y.append(m(y[-1]))

        # 拼接所有分支并进行最终卷积输出
        return self.cv2(torch.cat(y, 1))


class SCC_C2f_YAML(nn.Module):
    """
    SCC-enhanced C2f for YAML-based model definition

    This version is designed to work with YAML configuration files
    and can replace standard C2f blocks in YOLOv8 architecture.
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Args:
            c1: input channels
            c2: output channels
            n: number of SCC bottleneck blocks
            shortcut: residual connection flag
            g: convolution groups
            e: expansion ratio
        """
        super().__init__()

        self.c = int(c2 * e)  # hidden channels

        # Input convolution and split
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU()
        )

        # SCC bottleneck blocks
        self.m = nn.ModuleList(
            SCC_Bottleneck(self.c, self.c, shortcut=shortcut, g=g, use_scc=True)
            for _ in range(n)
        )

        # Output convolution
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: input tensor [B, C1, H, W]

        Returns:
            output tensor [B, C2, H, W]
        """
        y = list(self.cv1(x).chunk(2, 1))

        for m in self.m:
            y.append(m(y[-1]))

        return self.cv2(torch.cat(y, 1))


class Enhanced_SCC_C2f(nn.Module):
    """
    Enhanced SCC C2f with additional attention mechanisms

    Features:
    - SCC attention in bottlenecks
    - Additional channel-spatial attention after C2f
    - Improved feature recalibration
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Args:
            c1: input channels
            c2: output channels
            n: number of SCC bottleneck blocks
            shortcut: residual connection flag
            g: convolution groups
            e: expansion ratio
        """
        super().__init__()

        self.c = int(c2 * e)  # hidden channels

        # Input convolution and split
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU()
        )

        # SCC bottleneck blocks
        self.m = nn.ModuleList(
            SCC_Bottleneck(self.c, self.c, shortcut=shortcut, g=g, use_scc=True)
            for _ in range(n)
        )

        # Additional SCC attention after concatenation
        self.attention = SCC_Attention(c2)

        # Output convolution
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        """
        Forward pass with enhanced SCC attention

        Args:
            x: input tensor [B, C1, H, W]

        Returns:
            output tensor [B, C2, H, W]
        """
        y = list(self.cv1(x).chunk(2, 1))

        for m in self.m:
            y.append(m(y[-1]))

        # Concatenate and apply additional attention
        out = self.cv2(torch.cat(y, 1))
        out = self.attention(out)

        return out


# Runtime replacement function for existing YOLO models
def replace_c2f_with_scc(model, target_channels=[256, 512], enhanced=False):
    """
    Runtime replacement of C2f blocks with SCC-enhanced versions

    Args:
        model: YOLO model instance
        target_channels: list of channel sizes to replace with SCC
        enhanced: whether to use enhanced SCC version

    Returns:
        modified model with SCC blocks
    """
    net = model.model if hasattr(model, 'model') else model

    replaced_count = 0
    for name, module in net.named_modules():
        if isinstance(module, C2f):
            try:
                out_channels = module.cv2.conv.out_channels
                if out_channels in target_channels:
                    # Choose SCC version
                    if enhanced:
                        scc_c2f = Enhanced_SCC_C2f(
                            c1=module.cv1.conv.in_channels,
                            c2=out_channels,
                            n=len(module.m),
                            shortcut=getattr(module, 'shortcut', False),
                            e=getattr(module, 'e', 0.5)
                        )
                    else:
                        scc_c2f = SCC_C2f(
                            c1=module.cv1.conv.in_channels,
                            c2=out_channels,
                            n=len(module.m),
                            shortcut=getattr(module, 'shortcut', False),
                            e=getattr(module, 'e', 0.5)
                        )

                    # Replace the module
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]

                    if parent_name:
                        parent = net
                        for part in parent_name.split('.'):
                            parent = getattr(parent, part)
                        setattr(parent, child_name, scc_c2f)
                    else:
                        setattr(net, child_name, scc_c2f)

                    # Copy Ultralytics-specific attributes that may be needed
                    for attr in ["i", "f", "type", "np"]:
                        if hasattr(module, attr):
                            setattr(scc_c2f, attr, getattr(module, attr))

                    replaced_count += 1
                    print(f"✅ Replaced C2f at {name} with SCC_C2f (out_channels={out_channels})")

            except Exception as e:
                print(f"⚠️  Failed to replace C2f at {name}: {e}")
                continue

    print(f"🎯 Total SCC replacements: {replaced_count}")
    return model


# Test the SCC C2f implementation
if __name__ == "__main__":
    from utils import setup_device, validate_model, benchmark_inference

    device = setup_device()

    # Test SCC_C2f
    print("\n🔧 Testing SCC_C2f:")
    scc_c2f = SCC_C2f(c1=256, c2=256, n=2, shortcut=True)
    validate_model(scc_c2f, device, input_size=(1, 256, 32, 32))
    benchmark_inference(scc_c2f, device, input_size=(1, 256, 32, 32))

    # Test YAML version
    print("\n🔧 Testing SCC_C2f_YAML:")
    scc_c2f_yaml = SCC_C2f_YAML(c1=512, c2=512, n=3, shortcut=True)
    validate_model(scc_c2f_yaml, device, input_size=(1, 512, 16, 16))
    benchmark_inference(scc_c2f_yaml, device, input_size=(1, 512, 16, 16))

    # Test Enhanced version
    print("\n🔧 Testing Enhanced_SCC_C2f:")
    enhanced_scc = Enhanced_SCC_C2f(c1=256, c2=256, n=2, shortcut=True)
    validate_model(enhanced_scc, device, input_size=(1, 256, 32, 32))
    benchmark_inference(enhanced_scc, device, input_size=(1, 256, 32, 32))

    print("\n✅ SCC C2f modules tested successfully!")
    print("💡 Use replace_c2f_with_scc() to apply SCC attention to existing YOLO models")
