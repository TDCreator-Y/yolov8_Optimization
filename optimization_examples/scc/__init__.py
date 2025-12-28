# SCC (空间-通道交叉)注意力模块
# ===========================

"""
YOLOv8的SCC注意力机制模块

本模块为YOLOv8提供SCC注意力机制组件：
- ChannelAttention: 通道维度特征重标定
- SpatialAttention: 空间维度特征重标定
- SCC_Attention: 组合的空间-通道注意力（CBAM风格）
- SCC_Bottleneck: SCC增强的瓶颈块
- SCC_C2f: YOLOv8主干网络的SCC增强C2f模块
- 现有模型的运行时替换功能

关键特性：
- 双重注意力机制，实现全面的特征增强
- 轻量级高效实现
- 与标准YOLOv8架构兼容
- 支持运行时模型修改
"""

import sys
from pathlib import Path

# 确保可以导入utils模块
_current_dir = Path(__file__).parent.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from .scc_attention import (
    ChannelAttention,
    SpatialAttention,
    SCC_Attention,
    SCC_Bottleneck
)
from .scc_c2f import (
    SCC_C2f,
    SCC_C2f_YAML,
    Enhanced_SCC_C2f,
    replace_c2f_with_scc
)
from .train_scc import (
    train_scc_model,
    evaluate_scc_model,
    inference_demo,
    visualize_attention
)

__all__ = [
    # 核心注意力组件
    'ChannelAttention',
    'SpatialAttention',
    'SCC_Attention',
    'SCC_Bottleneck',

    # C2f模块
    'SCC_C2f',
    'SCC_C2f_YAML',
    'Enhanced_SCC_C2f',

    # 工具函数
    'replace_c2f_with_scc',

    # 训练函数
    'train_scc_model',
    'evaluate_scc_model',
    'inference_demo',
    'visualize_attention'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "YOLOv8优化团队"
