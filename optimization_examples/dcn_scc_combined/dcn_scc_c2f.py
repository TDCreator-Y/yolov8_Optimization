# DCN + SCC 组合C2f模块实现
# =========================

"""
YOLOv8的DCN + SCC组合C2f模块

在C2f块中结合可变形卷积网络和空间-通道交叉注意力，
实现最优的特征表示和几何建模能力。
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C2f

# 处理直接运行和包导入两种情况
try:
    # 当作为包的一部分导入时
    from .dcn_scc_attention import DCN_SCC_Bottleneck, Enhanced_DCN_SCC_Conv
except ImportError:
    # 当直接运行或从外部导入时
    from dcn_scc_attention import DCN_SCC_Bottleneck, Enhanced_DCN_SCC_Conv


class DCN_SCC_C2f(C2f):
    """
    DCN + SCC组合增强的C2f模块

    将标准Bottleneck块替换为DCN+SCC增强版本，
    用于卓越的特征提取和几何建模。

    架构特点：
        - 继承标准C2f的CSP设计理念
        - 集成DCN的几何变换建模能力
        - 加入SCC的空间-通道注意力机制
        - 保持与YOLOv8的完全兼容性

    技术优势：
        - 几何建模：处理变形和不规则形状物体
        - 注意力增强：优化特征选择和表达
        - 多尺度处理：适应不同大小的目标检测
        - 计算效率：优化的网络结构设计

    应用场景：
        - YOLOv8主干网络的关键特征层
        - 需要几何建模和注意力增强的任务
        - 对检测精度要求较高的应用
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        初始化DCN + SCC增强的C2f模块

        参数：
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): DCN_SCC瓶颈块的数量，默认1
            shortcut (bool): 是否使用残差连接，默认False
            g (int): 卷积分组数，默认1
            e (float): 隐藏层通道扩展比例，默认0.5

        网络结构：
            输入 -> 1x1卷积分支 -> 分割为两路 -> DCN_SCC瓶颈块处理 -> 拼接 -> 1x1输出卷积
            |                        |                      |
            +------------------------+----------------------+

        设计理念：
            - CSP架构：减少计算量，提高特征重用
            - DCN增强：每个瓶颈块都有几何建模能力
            - SCC优化：通过注意力机制提升特征质量
            - 模块化设计：便于集成和扩展
        """
        # 调用父类初始化基础结构
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)

        # 核心创新：将标准瓶颈块替换为DCN+SCC增强版本
        self.m = nn.ModuleList(
            DCN_SCC_Bottleneck(self.c, self.c, shortcut=shortcut, g=g, e=1.0, use_scc=True)
            for _ in range(n)
        )

    def forward(self, x):
        """
        DCN+SCC增强的C2f前向传播

        参数：
            x (torch.Tensor): 输入特征图，形状为[B, C1, H, W]

        返回值：
            torch.Tensor: 输出特征图，形状为[B, C2, H, W]

        处理流程：
            1. 输入预处理和分支分割（CSP结构）
            2. DCN_SCC瓶颈块的并行几何建模和注意力增强
            3. 多分支特征拼接融合
            4. 最终输出卷积

        技术特点：
            - 分支处理：并行计算提高效率
            - 几何建模：DCN学习空间变换参数
            - 注意力优化：SCC进行特征重标定
            - 特征融合：有效的多路信息整合

        性能优势：
            - 检测精度：几何建模提升变形物体检测
            - 特征质量：注意力机制优化表达能力
            - 计算效率：CSP设计减少冗余计算
        """
        # CSP分支分割：将输入特征分为两个分支进行并行处理
        y = list(self.cv1(x).chunk(2, 1))

        # 应用DCN+SCC增强的瓶颈块
        # 每个瓶颈块都包含几何建模和注意力增强
        for m in self.m:
            y.append(m(y[-1]))

        # 拼接所有分支并进行最终卷积输出
        return self.cv2(torch.cat(y, 1))


class Enhanced_DCN_SCC_C2f(nn.Module):
    """
    高级增强版的DCN + SCC C2f模块

    特性：
    - DCN + SCC瓶颈块集成
    - 增强的特征融合机制
    - 多尺度注意力机制
    - 自适应特征处理
    - 优化的计算效率

    架构创新：
        - 独立实现，不依赖标准C2f
        - 多层次特征处理
        - 增强的融合策略
        - 自适应注意力调节

    技术优势：
        - 更强的几何建模能力
        - 更精细的注意力控制
        - 更好的特征融合效果
        - 更高的计算效率

    应用场景：
        - 高精度目标检测任务
        - 复杂场景的特征提取
        - 对性能要求极高的应用
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Args:
            c1: input channels
            c2: output channels
            n: number of enhanced bottleneck blocks
            shortcut: residual connection flag
            g: convolution groups
            e: expansion ratio
        """
        super().__init__()

        self.c = int(c2 * e)  # hidden channels

        # Enhanced input processing with DCN+SCC
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU()
        )

        # DCN + SCC bottleneck blocks
        self.m = nn.ModuleList(
            DCN_SCC_Bottleneck(self.c, self.c, shortcut=shortcut, g=g, use_scc=True)
            for _ in range(n)
        )

        # Enhanced output processing
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # Additional feature refinement
        self.refinement = Enhanced_DCN_SCC_Conv(c2, c2, k=3, s=1, p=1)

    def forward(self, x):
        """
        Enhanced forward pass with multi-stage processing

        Args:
            x: input tensor [B, C1, H, W]

        Returns:
            output tensor [B, C2, H, W]
        """
        y = list(self.cv1(x).chunk(2, 1))

        # Apply DCN+SCC bottlenecks
        for m in self.m:
            y.append(m(y[-1]))

        # Primary output
        primary_out = self.cv2(torch.cat(y, 1))

        # Feature refinement with enhanced DCN+SCC
        refined_out = self.refinement(primary_out)

        return refined_out


class Adaptive_DCN_SCC_C2f(nn.Module):
    """
    Adaptive DCN + SCC C2f with Dynamic Feature Selection

    Features:
    - Adaptive bottleneck selection based on input complexity
    - Dynamic SCC attention strength
    - Computational efficiency optimization
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Args:
            c1: input channels
            c2: output channels
            n: maximum number of bottleneck blocks
            shortcut: residual connection flag
            g: convolution groups
            e: expansion ratio
        """
        super().__init__()

        self.c = int(c2 * e)
        self.n = n

        # Input processing
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU()
        )

        # Multiple bottleneck options
        self.m = nn.ModuleList([
            DCN_SCC_Bottleneck(self.c, self.c, shortcut=shortcut, g=g, use_scc=True)
            for _ in range(n)
        ])

        # Adaptive weighting network
        self.adaptive_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, n, 1, 1, bias=False),
            nn.Sigmoid()
        )

        # Output processing
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        """
        Adaptive forward pass with dynamic bottleneck selection

        Args:
            x: input tensor [B, C1, H, W]

        Returns:
            output tensor [B, C2, H, W]
        """
        # Compute adaptive weights
        weights = self.adaptive_weight(x)  # [B, n, 1, 1]

        # Base features
        y = list(self.cv1(x).chunk(2, 1))

        # Apply weighted bottlenecks
        for i, m in enumerate(self.m):
            bottleneck_out = m(y[-1])
            weight = weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
            y.append(bottleneck_out * weight + y[-1] * (1 - weight))

        return self.cv2(torch.cat(y, 1))


# Runtime replacement function for existing YOLO models
def replace_c2f_with_dcn_scc(model, target_channels=[256, 512], mode='standard'):
    """
    Runtime replacement of C2f blocks with DCN+SCC enhanced versions

    Args:
        model: YOLO model instance
        target_channels: list of channel sizes to replace
        mode: replacement mode ('standard', 'enhanced', 'adaptive')

    Returns:
        modified model with DCN+SCC blocks
    """
    net = model.model if hasattr(model, 'model') else model

    replaced_count = 0
    for name, module in net.named_modules():
        if isinstance(module, C2f):
            try:
                out_channels = module.cv2.conv.out_channels
                if out_channels in target_channels:
                    # Choose replacement type
                    if mode == 'enhanced':
                        dcn_scc_c2f = Enhanced_DCN_SCC_C2f(
                            c1=module.cv1.conv.in_channels,
                            c2=out_channels,
                            n=len(module.m),
                            shortcut=getattr(module, 'shortcut', False),
                            e=getattr(module, 'e', 0.5)
                        )
                    elif mode == 'adaptive':
                        dcn_scc_c2f = Adaptive_DCN_SCC_C2f(
                            c1=module.cv1.conv.in_channels,
                            c2=out_channels,
                            n=len(module.m),
                            shortcut=getattr(module, 'shortcut', False),
                            e=getattr(module, 'e', 0.5)
                        )
                    else:  # standard
                        dcn_scc_c2f = DCN_SCC_C2f(
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
                        setattr(parent, child_name, dcn_scc_c2f)
                    else:
                        setattr(net, child_name, dcn_scc_c2f)

                    # Copy Ultralytics-specific attributes that may be needed
                    for attr in ["i", "f", "type", "np"]:
                        if hasattr(module, attr):
                            setattr(dcn_scc_c2f, attr, getattr(module, attr))

                    replaced_count += 1
                    print(f"✅ Replaced C2f at {name} with DCN+SCC C2f ({mode}, out_channels={out_channels})")

            except Exception as e:
                print(f"⚠️  Failed to replace C2f at {name}: {e}")
                continue

    print(f"🎯 Total DCN+SCC replacements: {replaced_count}")
    return model


# Test the DCN + SCC C2f implementation
if __name__ == "__main__":
    # When run as main script, import from parent directory
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    from utils import setup_device, validate_model, benchmark_inference

    device = setup_device()

    # Test Standard DCN+SCC C2f
    print("\n🔧 Testing DCN_SCC_C2f:")
    dcn_scc_c2f = DCN_SCC_C2f(c1=256, c2=256, n=2, shortcut=True)
    validate_model(dcn_scc_c2f, device, input_size=(1, 256, 32, 32))
    benchmark_inference(dcn_scc_c2f, device, input_size=(1, 256, 32, 32))

    # Test Enhanced version
    print("\n🔧 Testing Enhanced_DCN_SCC_C2f:")
    enhanced_dcn_scc = Enhanced_DCN_SCC_C2f(c1=256, c2=256, n=2, shortcut=True)
    validate_model(enhanced_dcn_scc, device, input_size=(1, 256, 32, 32))
    benchmark_inference(enhanced_dcn_scc, device, input_size=(1, 256, 32, 32))

    # Test Adaptive version
    print("\n🔧 Testing Adaptive_DCN_SCC_C2f:")
    adaptive_dcn_scc = Adaptive_DCN_SCC_C2f(c1=256, c2=256, n=3, shortcut=True)
    validate_model(adaptive_dcn_scc, device, input_size=(1, 256, 32, 32))
    benchmark_inference(adaptive_dcn_scc, device, input_size=(1, 256, 32, 32))

    print("\n✅ DCN + SCC C2f modules tested successfully!")
    print("🎯 Combines DCN geometric modeling with SCC attention mechanisms")
    print("💡 Use replace_c2f_with_dcn_scc() to apply combined optimization to existing YOLO models")


class DCN_SCC_C2f_YAML(nn.Module):
    """
    基于YAML配置的DCN + SCC组合增强C2f模块

    此版本专门设计用于YAML配置文件中，
    可以直接在YOLOv8架构中替换标准C2f块。

    技术特点：
        - 独立的网络结构实现，不依赖标准C2f
        - 集成DCN几何建模和SCC注意力机制
        - 优化的特征融合策略
        - 完整的多尺度处理能力

    YAML配置示例：
        ```yaml
        backbone:
          - [-1, 6, DCN_SCC_C2f_YAML, [256, True]]  # P3层DCN+SCC组合增强
        ```

    参数：
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        n (int): 瓶颈块数量，默认2
        shortcut (bool): 是否使用残差连接，默认True
        g (int): 卷积分组数，默认1
        e (float): 扩展比例，默认0.5
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        初始化YAML配置版本的DCN+SCC组合C2f模块

        参数：
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): DCN+SCC瓶颈块的数量，默认1
            shortcut (bool): 残差连接标志，默认True
            g (int): 卷积分组数，默认1
            e (float): 扩展比例，默认0.5
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        # Input processing
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU()
        )

        # DCN+SCC bottleneck blocks
        self.m = nn.ModuleList(
            DCN_SCC_Bottleneck(self.c, self.c, shortcut=shortcut, g=g)
            for _ in range(n)
        )

        # Output processing
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        """
        前向传播

        参数：
            x (torch.Tensor): 输入特征图 [B, C1, H, W]

        返回：
            torch.Tensor: 输出特征图 [B, C2, H, W]
        """
        # Split and process
        y = list(self.cv1(x).chunk(2, 1))

        # Apply DCN+SCC bottlenecks
        for m in self.m:
            y.append(m(y[-1]))

        # Concatenate and output
        return self.cv2(torch.cat(y, 1))
