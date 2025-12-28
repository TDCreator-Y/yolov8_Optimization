# YOLOv8优化示例项目 - 可运行脚本指南

本文档详细介绍optimization_examples目录下所有可以运行的Python脚本文件，及其功能、用法和输出结果。

## 📋 可运行脚本总览

本项目包含多个可运行的Python脚本，按功能分为以下几类：

### 🚀 主运行脚本

#### 1. `run_all_examples.py` - 主运行脚本
**位置**: `optimization_examples/run_all_examples.py`

**功能**: 项目的主入口脚本，提供统一的命令行接口来运行各种优化示例和实验。

**命令行参数**:
```bash
# 运行特定优化方法
python run_all_examples.py --method dcn        # DCN优化
python run_all_examples.py --method scc        # SCC注意力优化
python run_all_examples.py --method dcn_scc    # DCN+SCC组合优化
python run_all_examples.py --method all        # 运行所有方法

# 实验功能
python run_all_examples.py --ablation          # 消融实验
python run_all_examples.py --test              # 模块测试

# 帮助信息
python run_all_examples.py --help              # 显示详细帮助
```

**输出结果**:
- **训练结果**: `results_dcn/`, `results_scc/`, `results_dcn_scc_combined/`
- **实验报告**: `ablation_results/` (消融实验)
- **测试报告**: 控制台输出模块测试状态
- **日志信息**: 详细的执行过程和性能指标

**执行时间**: 5-60分钟（根据硬件和参数而定）

---

#### 2. `example_usage.py` - 使用示例脚本 ⭐⭐⭐⭐
**位置**: `optimization_examples/example_usage.py`

**功能**: 提供5个循序渐进的使用示例，从基础到高级完整展示项目功能。

**运行方式**:
```bash
python example_usage.py  # 运行所有5个示例
```

**包含示例**:
1. **DCN基础优化** - 几何建模能力演示
2. **SCC基础注意力** - 特征增强效果演示
3. **DCN+SCC组合优化** - 双重优化效果演示
4. **推理性能对比** - 效率评估和选择指南
5. **自定义训练配置** - 高级训练参数配置

**输出结果**:
- **训练结果目录**:
  - `results_dcn_demo/`
  - `results_scc_demo/`
  - `results_combined_demo/`
  - `results_custom/`
- **性能对比报告**: 控制台输出各方法的推理速度
- **可视化结果**: 训练过程中的图表和指标

**执行时间**: 15-45分钟

---

### 🏃 专项训练脚本

#### 3. `dcn/train_dcn.py` - DCN优化训练 ⭐⭐⭐⭐
**位置**: `optimization_examples/dcn/train_dcn.py`

**功能**: 专门用于DCN（可变形卷积网络）优化的完整训练流程。

**运行方式**:
```bash
python dcn/train_dcn.py
```

**主要功能**:
- 加载YOLOv8基础模型
- 应用DCN优化到P3和P4层
- 执行完整训练流程
- 生成性能评估报告

**输出结果**:
- **训练结果**: `results_dcn/`
- **模型权重**: `best.pt`, `last.pt`
- **训练日志**: TensorBoard兼容的日志文件
- **验证结果**: mAP和其他性能指标

**技术特点**: 专注于几何变换建模，适合处理变形物体检测

---

#### 4. `scc/train_scc.py` - SCC注意力训练 ⭐⭐⭐⭐
**位置**: `optimization_examples/scc/train_scc.py`

**功能**: 专门用于SCC（空间-通道交叉注意力）优化的完整训练流程。

**运行方式**:
```bash
python scc/train_scc.py
```

**主要功能**:
- 集成SCC注意力机制
- 轻量级特征增强训练
- 注意力权重可视化（可选）

**输出结果**:
- **训练结果**: `results_scc/`
- **注意力分析**: 注意力权重分布（如果启用可视化）
- **性能指标**: 检测精度和效率评估

**技术特点**: 基于CBAM架构的注意力优化，计算效率高

---

#### 5. `dcn_scc_combined/train_dcn_scc.py` - 组合优化训练 ⭐⭐⭐⭐⭐
**位置**: `optimization_examples/dcn_scc_combined/train_dcn_scc.py`

**功能**: DCN和SCC双重优化的完整训练流程，以及消融实验功能。

**运行方式**:
```bash
python dcn_scc_combined/train_dcn_scc.py
```

**主要功能**:
- 同时集成DCN和SCC优化
- 提供消融实验对比
- 生成综合性能分析

**输出结果**:
- **训练结果**: `results_dcn_scc_combined/`
- **消融实验**: `ablation_results/`（可选）
- **对比分析**: 多方法性能对比报告

**技术特点**: 几何建模+注意力增强的双重优化，性能最优

---