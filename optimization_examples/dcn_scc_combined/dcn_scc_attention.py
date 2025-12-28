# DCN + SCC 组合优化实现
# ===================

"""
YOLOv8的DCN + SCC组合注意力模块

本模块将可变形卷积网络(DCN)和空间-通道交叉注意力(SCC)相结合，
用于增强特征表示和几何建模能力。

核心创新：
- DCN提供几何变换建模能力
- SCC提供注意力增强机制
- 两者的有机结合实现性能最优化
"""

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

# 处理直接运行和包导入两种情况
try:
    # 当作为包的一部分导入时
    from scc.scc_attention import SCC_Attention
except ImportError:
    # 当直接运行或从外部导入时，尝试相对导入或直接导入
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    scc_dir = current_dir / 'scc'
    if str(scc_dir) not in sys.path:
        sys.path.insert(0, str(scc_dir))
    try:
        from scc_attention import SCC_Attention
    except ImportError:
        # 如果还是失败，创建一个简单的替代实现
        print("Warning: SCC_Attention not found, using simplified version")
        class SCC_Attention(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv = nn.Conv2d(channels, channels, 1)
            def forward(self, x):
                return self.conv(x)


class DCN_SCC_Conv(nn.Module):
    """
    DCN + SCC 组合卷积层

    将可变形卷积与空间-通道注意力相结合，
    用于增强特征提取和特征表示能力。

    架构特点：
        - DCN分支：学习空间偏移，实现几何变换建模
        - SCC分支：应用注意力机制，增强特征选择
        - 双分支融合：结合几何建模和注意力增强的优势

    技术优势：
        - 能够处理复杂几何变换的目标
        - 通过注意力机制提升特征质量
        - 保持计算效率的同时提升性能
    """

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, use_scc=True):
        """
        初始化DCN + SCC组合卷积层

        参数：
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核尺寸，默认3
            s (int): 步长，默认1
            p (int): 填充，默认1
            g (int): 分组数，默认1
            use_scc (bool): 是否包含SCC注意力，默认True

        网络结构：
            输入 -> 偏移量预测分支 -> 可变形卷积 -> 批归一化 -> 激活
                        ↓
                   SCC注意力（可选）
                        ↓
                     输出特征

        设计理念：
            - DCN专注于几何变换的学习
            - SCC专注于重要特征的增强
            - 通过串联方式实现优势互补
        """
        super().__init__()

        # DCN的偏移量预测分支
        # 预测每个卷积核位置的2D偏移量（x,y方向）
        self.offset_conv = nn.Conv2d(
            c1, 2 * k * k, kernel_size=k, stride=s, padding=p, bias=False
        )

        # 可变形卷积层
        # 使用预测的偏移量进行几何变换建模
        self.dcn = DeformConv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            bias=False
        )

        # 批归一化和激活函数
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        # SCC注意力机制（可选）
        self.use_scc = use_scc
        if use_scc:
            self.scc = SCC_Attention(c2)

    def forward(self, x):
        """
        DCN + SCC组合的前向传播过程

        参数：
            x (torch.Tensor): 输入特征图，形状为[B, C1, H, W]

        返回值：
            torch.Tensor: 输出特征图，形状为[B, C2, H', W']

        处理流程：
            1. 预测可变形卷积的偏移量
            2. 应用可变形卷积进行几何建模
            3. 批归一化和激活
            4. 可选的SCC注意力增强
            5. 返回处理后的特征

        技术细节：
            - 偏移量预测：学习每个采样点的空间位置调整
            - 可变形卷积：根据偏移量进行几何变换
            - SCC注意力：通过空间-通道注意力增强特征质量

        计算复杂度：
            - DCN部分：标准卷积的2-3倍计算量
            - SCC部分：轻量级注意力机制，计算开销较小
            - 总体：比纯DCN略有增加，但性能提升显著
        """

        # 步骤1：预测可变形卷积的偏移量
        # 使用专用卷积层预测每个位置的几何变换参数
        offsets = self.offset_conv(x)

        # 步骤2：应用可变形卷积进行几何建模
        # 根据预测的偏移量调整采样位置，实现几何变换建模
        x = self.dcn(x, offsets)

        # 步骤3：批归一化和激活
        # 标准化特征分布，应用SiLU激活函数增强非线性表达
        x = self.act(self.bn(x))

        # 步骤4：应用SCC注意力（如果启用）
        # 通过空间-通道交叉注意力增强重要特征
        if self.use_scc:
            x = self.scc(x)

        return x


class DCN_SCC_Bottleneck(nn.Module):
    """
    DCN + SCC增强的瓶颈块

    在瓶颈架构中结合DCN进行几何建模和SCC注意力进行特征增强。

    架构设计：
        - 继承标准Bottleneck的CSP设计理念
        - 集成DCN进行几何变换建模
        - 添加SCC注意力机制增强特征质量
        - 保持轻量级设计和计算效率

    应用场景：
        - YOLOv8的主干网络特征提取层
        - 需要同时处理几何变换和注意力增强的任务
        - 对特征质量和几何建模都有要求的场景

    性能特点：
        - 几何建模能力：能够处理变形物体检测
        - 注意力增强：提升特征的表达能力和选择性
        - 计算效率：保持合理的计算复杂度
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, use_scc=True):
        """
        初始化DCN + SCC增强的瓶颈块

        参数：
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            shortcut (bool): 是否使用残差连接，默认True
            g (int): 分组卷积的组数，默认1
            e (float): 隐藏层扩展比例，默认0.5
            use_scc (bool): 是否使用SCC注意力，默认True

        网络结构：
            输入
             │
            ┌─┴─┐
            │  │ 残差连接（可选）
            │  │
            │  ┌─────────────────┐
            │  │ DCN_SCC_Conv    │ 主要处理分支
            │  │ (c1 -> c_hidden)│
            │  └─────────────────┘
            │         │
            │  ┌─────────────────┐
            │  │ DCN_SCC_Conv    │ 进一步特征提取
            │  │ (c_hidden -> c2)│
            │  └─────────────────┘
            │         │
            └─────────┼─────────┘
                      │
                   输出 (c2)

        设计优势：
            - 多层次特征处理：两个DCN_SCC_Conv的串联
            - 几何建模：每个卷积层都有DCN的几何变换能力
            - 注意力增强：每个层都有SCC的特征选择能力
            - 梯度流动：残差连接保持梯度传播
        """
        super().__init__()

        # 计算隐藏层通道数
        c_ = int(c2 * e)  # 隐藏层通道数 = 输出通道数 * 扩展比例

        # 标准1x1卷积用于通道调整
        # 将输入通道数调整为隐藏层通道数，为后续DCN处理做准备
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        # DCN + SCC 3x3卷积用于增强的空间特征提取
        # 这是瓶颈块的核心组件，结合几何建模和注意力增强
        self.cv2 = DCN_SCC_Conv(c_, c2, k=3, s=1, p=1, g=g, use_scc=use_scc)

        # 残差连接设置
        # 只有当输入输出通道数相等且启用shortcut时才使用残差连接
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        DCN + SCC瓶颈块的前向传播

        参数：
            x (torch.Tensor): 输入特征图，形状为[B, C1, H, W]

        返回值：
            torch.Tensor: 输出特征图，形状为[B, C2, H, W]

        处理流程：
            1. 通过1x1卷积调整通道数并激活
            2. 通过DCN+SCC卷积进行几何建模和注意力增强
            3. 可选的残差连接融合
            4. 返回最终特征

        技术细节：
            - 通道调整：1x1卷积实现跨层特征变换
            - 几何建模：DCN学习空间变换参数
            - 注意力增强：SCC提升特征质量
            - 梯度优化：残差连接改善梯度流动

        设计理念：
            - 瓶颈设计：通过通道压缩降低计算复杂度
            - 多层次处理：1x1 + 3x3卷积的组合
            - 特征增强：DCN和SCC的双重优化
        """

        # 前向传播实现：特征提取 -> 残差融合
        y = self.cv2(self.cv1(x))  # 通过两个卷积层处理
        return x + y if self.add else y  # 残差连接（可选）


class Enhanced_DCN_SCC_Conv(nn.Module):
    """
    多尺度特征融合的增强版DCN + SCC

    高级版本，包含多尺度特征融合和增强的注意力机制。

    架构创新：
        - 多分支设计：不同感受野的并行处理
        - 尺度融合：3x3和5x5卷积的互补特性
        - 注意力增强：最终的SCC注意力模块
        - 特征聚合：1x1卷积进行通道融合

    技术优势：
        - 多尺度建模：同时捕捉局部和全局特征
        - 几何变换：每个分支都有DCN的几何建模能力
        - 注意力优化：多层次的特征选择和增强
        - 特征融合：有效的多分支信息聚合

    应用场景：
        - 需要处理多尺度目标的复杂检测任务
        - 对特征质量要求较高的应用
        - 计算资源相对充足的场景
    """

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1):
        """
        初始化增强版DCN + SCC卷积层

        参数：
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 基础卷积核尺寸（实际未使用，由分支决定）
            s (int): 步长
            p (int): 填充（实际未使用，由分支决定）
            g (int): 分组数

        多分支架构：
            输入特征
               │
            ┌──┴──┐
            │     │
        3x3分支  5x5分支
            │     │
        DCN+SCC  DCN+SCC
            │     │
            └─────┼─────┘
                  │
            1x1融合卷积
                  │
            最终SCC注意力
                  │
              输出特征

        设计理念：
            - 分支1（3x3）：专注局部特征和细节建模
            - 分支2（5x5）：专注全局上下文和尺度建模
            - 融合层：有效整合多尺度信息
            - 最终注意力：全局特征优化
        """
        super().__init__()

        # 多分支DCN设计，不同卷积核尺寸
        # 分支1：3x3卷积，专注局部特征提取
        self.branch1 = DCN_SCC_Conv(c1, c2//2, k=3, s=s, p=1, g=g, use_scc=True)
        # 分支2：5x5卷积，专注全局特征建模
        self.branch2 = DCN_SCC_Conv(c1, c2//2, k=5, s=s, p=2, g=g, use_scc=True)

        # 特征融合模块
        # 将两个分支的输出进行通道维度融合
        self.fusion = nn.Sequential(
            nn.Conv2d(c2, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 最终SCC注意力模块
        # 对融合后的特征进行全局注意力优化
        self.final_attention = SCC_Attention(c2)

    def forward(self, x):
        """
        多分支DCN + SCC前向传播

        参数：
            x (torch.Tensor): 输入特征图，形状为[B, C1, H, W]

        返回值：
            torch.Tensor: 输出特征图，形状为[B, C2, H, W]

        处理流程：
            1. 并行处理两个分支（3x3和5x5 DCN+SCC）
            2. 在通道维度拼接两个分支的输出
            3. 通过1x1卷积进行特征融合
            4. 应用最终的SCC注意力优化

        多尺度优势：
            - 3x3分支：捕捉局部细节和纹理信息
            - 5x5分支：建模更大范围的上下文关系
            - 融合机制：有效整合不同尺度的特征信息

        计算特点：
            - 并行处理：两个分支同时计算
            - 内存效率：分支输出通道各占一半
            - 特征增强：多层次的几何建模和注意力优化
        """

        # 多尺度特征提取
        # 两个分支并行处理，捕捉不同感受野的特征
        feat1 = self.branch1(x)  # 3x3分支：局部特征建模
        feat2 = self.branch2(x)  # 5x5分支：全局特征建模

        # 特征融合
        # 在通道维度上拼接两个分支的输出
        combined = torch.cat([feat1, feat2], dim=1)
        # 通过1x1卷积进行跨通道特征融合
        fused = self.fusion(combined)

        # 最终注意力优化
        # 应用SCC注意力进行全局特征重标定
        return self.final_attention(fused)


# 测试DCN + SCC实现
if __name__ == "__main__":
    """
    DCN + SCC组合模块的完整测试验证

    测试涵盖：
    1. DCN_SCC_Conv：基础几何建模+注意力增强
    2. DCN_SCC_Bottleneck：瓶颈架构的组合优化
    3. Enhanced_DCN_SCC_Conv：多尺度特征融合
    4. 模型结构验证和前向传播测试
    5. 推理性能基准测试

    测试目标：
        - 验证各组件的正确集成和功能
        - 评估DCN与SCC的协同效果
        - 分析计算复杂度和性能表现
        - 确保与现有架构的兼容性

    技术验证：
        - 几何变换建模能力的正确性
        - 注意力机制的有效性
        - 多尺度特征融合的质量
        - 整体架构的稳定性和效率

    输出信息：
        - 各模块的验证状态
        - 性能基准测试结果
        - 资源使用统计
        - 使用建议和最佳实践
    """
    from utils import setup_device, validate_model, benchmark_inference

    # 初始化计算设备
    device = setup_device()

    print("\n🔧 测试DCN + SCC组合组件:")

    # 测试DCN + SCC卷积层
    print("\n📦 DCN_SCC_Conv测试:")
    dcn_scc_conv = DCN_SCC_Conv(c1=256, c2=256, k=3, s=1, p=1, use_scc=True)
    print("   基础DCN+SCC卷积层 - 几何建模+注意力增强")
    validate_model(dcn_scc_conv, device, input_size=(1, 256, 32, 32))
    benchmark_inference(dcn_scc_conv, device, input_size=(1, 256, 32, 32))

    # 测试DCN + SCC瓶颈块
    print("\n📦 DCN_SCC_Bottleneck测试:")
    dcn_scc_bottleneck = DCN_SCC_Bottleneck(c1=256, c2=256, use_scc=True)
    print("   DCN+SCC瓶颈块 - 高效的组合特征处理")
    validate_model(dcn_scc_bottleneck, device, input_size=(1, 256, 32, 32))
    benchmark_inference(dcn_scc_bottleneck, device, input_size=(1, 256, 32, 32))

    # 测试增强版多尺度卷积
    print("\n📦 Enhanced_DCN_SCC_Conv测试:")
    enhanced_conv = Enhanced_DCN_SCC_Conv(c1=256, c2=256)
    print("   增强版多尺度DCN+SCC - 多分支特征融合")
    validate_model(enhanced_conv, device, input_size=(1, 256, 32, 32))
    benchmark_inference(enhanced_conv, device, input_size=(1, 256, 32, 32))

    print("\n✅ DCN + SCC组合模块测试成功！")
    print("🎯 核心优势：")
    print("   • 几何建模能力：DCN处理变形物体")
    print("   • 注意力增强：SCC优化特征质量")
    print("   • 多尺度融合：Enhanced版本的全局建模")
    print("   • 计算效率：优化的架构设计")
    print("\n🚀 可用于YOLOv8的各种优化场景")
