# DCN (可变形卷积网络) 优化模块
# ============================

"""
YOLOv8的DCN优化模块

本模块为YOLOv8提供DCN增强组件：
- DCNConv: 可变形卷积层
- DCNBottleneck: DCN增强的瓶颈块
- DCN_C2f: 用于YOLOv8主干网络的DCN增强C2f模块
- 现有模型的运行时替换功能

关键特性：
- 学习空间偏移以实现更好的几何建模
- 改进可变形物体的检测能力
- 与标准YOLOv8架构兼容
- 支持运行时模型修改
"""

import sys
from pathlib import Path

# 确保可以导入utils模块
_current_dir = Path(__file__).parent.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from .dcn_conv import DCNConv, DCNBottleneck
from .dcn_c2f import DCN_C2f, DCN_C2f_YAML, replace_c2f_with_dcn
from .train_dcn import train_dcn_model, evaluate_dcn_model, inference_demo

__all__ = [
    # 核心DCN组件
    'DCNConv',
    'DCNBottleneck',
    'DCN_C2f',
    'DCN_C2f_YAML',

    # 工具函数
    'replace_c2f_with_dcn',

    # 训练函数
    'train_dcn_model',
    'evaluate_dcn_model',
    'inference_demo'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "YOLOv8优化团队"
