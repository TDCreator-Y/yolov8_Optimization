# DCN + SCC 组合优化模块
# ===================

"""
YOLOv8的DCN + SCC组合优化模块

本模块提供DCN + SCC组合优化组件：
- DCN_SCC_Conv: 集成的可变形卷积与注意力机制
- DCN_SCC_Bottleneck: DCN + SCC增强的瓶颈块
- DCN_SCC_C2f: 组合优化C2f模块
- Enhanced_DCN_SCC_C2f: 高级组合优化模块
- Adaptive_DCN_SCC_C2f: 动态组合优化模块
- 现有模型的运行时替换功能

关键特性：
- 将几何建模(DCN)和注意力机制(SCC)相结合
- 多尺度特征增强
- 自适应处理能力
- 与标准YOLOv8架构兼容
- 支持全面的消融实验
"""

import sys
from pathlib import Path

# 确保可以导入utils模块和父级模块
_current_dir = Path(__file__).parent.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from .dcn_scc_attention import (
    DCN_SCC_Conv,
    DCN_SCC_Bottleneck,
    Enhanced_DCN_SCC_Conv
)
from .dcn_scc_c2f import (
    DCN_SCC_C2f,
    Enhanced_DCN_SCC_C2f,
    Adaptive_DCN_SCC_C2f,
    replace_c2f_with_dcn_scc
)
from .train_dcn_scc import (
    train_combined_model,
    ablation_study,
    evaluate_combined_model,
    inference_demo
)

__all__ = [
    # 核心组合组件
    'DCN_SCC_Conv',
    'DCN_SCC_Bottleneck',
    'Enhanced_DCN_SCC_Conv',

    # C2f模块
    'DCN_SCC_C2f',
    'Enhanced_DCN_SCC_C2f',
    'Adaptive_DCN_SCC_C2f',

    # 工具函数
    'replace_c2f_with_dcn_scc',

    # 训练和评估
    'train_combined_model',
    'ablation_study',
    'evaluate_combined_model',
    'inference_demo'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "YOLOv8优化团队"
