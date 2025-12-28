#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8优化示例项目 - 使用示例脚本
=================================

本脚本演示了如何使用各种优化模块进行训练和推理，
提供了从基础到高级的完整使用示例。

主要内容：
==========
1. DCN基础优化示例 - 展示可变形卷积网络的基本使用
2. SCC基础优化示例 - 展示空间-通道注意力机制的基本使用
3. DCN+SCC组合优化示例 - 展示两种优化方法的结合使用
4. 推理性能对比示例 - 对比不同优化方法的推理性能
5. 自定义训练配置示例 - 展示高级训练参数配置

技术特点：
==========
- 循序渐进：从简单到复杂的示例层次
- 实用导向：每个示例都可以直接运行
- 对比分析：提供性能对比和效果展示
- 错误处理：包含完善的异常处理机制

使用方法：
==========
直接运行脚本：
    python example_usage.py

或者导入特定函数：
    from example_usage import example_1_dcn_basic
    example_1_dcn_basic()

注意事项：
==========
- 需要安装Ultralytics和相关依赖
- GPU环境可以显著提升运行速度
- 示例中使用了较短的训练轮数以便快速演示
- 实际使用时可以调整参数以获得更好效果
"""

import os
import sys
from pathlib import Path

# 将当前目录添加到Python路径以便导入模块
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def example_1_dcn_basic():
    """
    示例1：DCN基础优化演示

    本示例展示如何使用DCN（可变形卷积网络）优化YOLOv8模型。
    DCN通过学习卷积核的空间偏移，实现对可变形物体的精确几何建模。

    技术要点：
    =========
    - 使用标准YOLOv8n作为基础模型
    - 在P3和P4层应用DCN优化（通道数256和512）
    - 采用简化的训练配置以便快速演示
    - 保存训练结果用于后续分析

    适用场景：
    =========
    - 处理可变形物体（如动物、人类姿势变化）
    - 需要精确几何建模的检测任务
    - 对检测边界精确度要求较高的应用

    训练配置：
    =========
    - epochs=5: 仅用于演示，实际建议50-100轮
    - imgsz=320: 较小分辨率以加快训练速度
    - batch=4: 小批量以适应不同硬件配置
    - 其他参数使用默认设置

    输出结果：
    =========
    - 训练日志和性能指标
    - 模型权重文件
    - 训练过程中的可视化结果

    预期效果：
    =========
    - 提升对变形物体的检测精度
    - 改善复杂几何形状的边界预测
    - 增强模型的几何变换鲁棒性
    """
    print("📋 示例1：DCN基础优化演示")
    print("=" * 40)
    print("🎯 目标：展示DCN可变形卷积网络的基本优化流程")
    print("🔧 重点：几何建模能力的提升")

    # 导入必要的模块
    from ultralytics import YOLO
    from dcn.dcn_c2f import replace_c2f_with_dcn

    # 步骤1：加载YOLOv8基础模型
    print("\n📥 步骤1：加载基础模型")
    print("   使用YOLOv8 Nano版本作为起点")
    model = YOLO('yolov8n.yaml')
    print(f"   ✅ 模型加载完成：{type(model).__name__}")

    # 步骤2：应用DCN优化
    print("\n🔄 步骤2：应用DCN优化")
    print("   在P3(256通道)和P4(512通道)层应用可变形卷积")
    model = replace_c2f_with_dcn(model, target_channels=[256, 512])
    print("   ✅ DCN优化应用成功")

    # 步骤3：执行训练演示
    print("\n🏃 步骤3：执行训练演示")
    training_config = {
        'data': 'coco8.yaml',  # 使用Ultralytics内置的COCO8小型数据集
        'epochs': 5,        # 演示用短周期，实际建议更长
        'imgsz': 320,       # 较小分辨率加快演示速度
        'batch': 4,         # 小批量适应不同硬件
        'project': 'results_dcn_demo',      # 结果保存目录
        'name': 'dcn_basic_example'         # 实验名称
    }

    print("   训练配置：")
    print(f"   • 数据集: {training_config['data']}")
    print(f"   • 训练轮数: {training_config['epochs']}")
    print(f"   • 图像尺寸: {training_config['imgsz']}x{training_config['imgsz']}")
    print(f"   • 批次大小: {training_config['batch']}")
    print(f"   • 保存路径: {training_config['project']}/{training_config['name']}")

    # 开始训练
    results = model.train(**training_config)

    print("\n✅ DCN训练演示完成！")
    print("📊 结果分析：")
    print("   • 训练日志保存在控制台输出")
    print("   • 模型权重保存在results_dcn_demo目录")
    print("   • 可查看tensorboard日志了解详细训练过程")
    print("   💡 提示：实际应用中建议增加训练轮数以获得更好效果")

    return model

def example_2_scc_basic():
    """
    示例2：SCC基础注意力优化演示

    本示例展示如何使用SCC（空间-通道交叉注意力）优化YOLOv8模型。
    SCC基于CBAM架构，通过注意力机制增强特征表示质量。

    技术要点：
    =========
    - 使用标准SCC注意力模块（非增强版）
    - 在P3和P4层应用注意力优化
    - 轻量级注意力机制，计算开销小
    - 保持模型推理速度的同时提升精度

    适用场景：
    =========
    - 需要提升特征质量的检测任务
    - 对计算资源有一定限制的应用
    - 希望平衡性能和效率的场景

    注意力机制：
    ===========
    - 通道注意力：学习哪些特征通道更重要
    - 空间注意力：学习图像中的哪些区域更重要
    - 顺序处理：通道优先，然后空间细化

    训练配置：
    =========
    - 使用与DCN示例相同的训练参数
    - 保持一致性便于性能对比
    - 短周期训练适合演示场景

    预期效果：
    =========
    - 提升特征的表达能力和选择性
    - 改善模型对重要区域的关注度
    - 增强检测结果的置信度和准确性
    """
    print("\n📋 示例2：SCC基础注意力优化演示")
    print("=" * 45)
    print("🧠 目标：展示SCC空间-通道交叉注意力的基本优化流程")
    print("🎯 重点：注意力机制的特征增强效果")

    # 导入必要的模块
    from ultralytics import YOLO
    from scc.scc_c2f import replace_c2f_with_scc

    # 步骤1：加载基础模型
    print("\n📥 步骤1：加载基础模型")
    model = YOLO('yolov8n.yaml')
    print("   ✅ 基础模型加载完成")

    # 步骤2：应用SCC注意力优化
    print("\n🔄 步骤2：应用SCC注意力优化")
    print("   使用标准SCC模块（非增强版）")
    print("   在P3和P4层添加空间-通道注意力机制")
    model = replace_c2f_with_scc(model, target_channels=[256, 512], enhanced=False)
    print("   ✅ SCC注意力优化应用成功")

    # 步骤3：执行训练演示
    print("\n🏃 步骤3：执行训练演示")
    training_config = {
        'data': 'coco8.yaml',
        'epochs': 5,        # 演示用短周期
        'imgsz': 320,       # 较小分辨率
        'batch': 4,         # 小批量
        'project': 'results_scc_demo',
        'name': 'scc_basic_example'
    }

    print("   训练配置：")
    print(f"   • SCC注意力: 标准版本（轻量级）")
    print(f"   • 优化层级: P3, P4 (通道数: {training_config['batch']}->{512})")
    print(f"   • 训练轮数: {training_config['epochs']} (演示用)")

    # 开始训练
    results = model.train(**training_config)

    print("\n✅ SCC训练演示完成！")
    print("📊 结果分析：")
    print("   • SCC通过注意力机制提升特征质量")
    print("   • 轻量级设计保持推理速度")
    print("   • 适合需要特征增强的检测任务")
    print("   💡 提示：与DCN相比，SCC更注重特征的选择性而非几何建模")

    return model

def example_3_dcn_scc_combined():
    """
    示例3：DCN + SCC组合优化演示

    本示例展示如何将DCN和SCC两种优化方法相结合，
    实现几何建模和注意力增强的双重优化效果。

    技术要点：
    =========
    - 同时集成DCN的几何建模能力和SCC的注意力增强
    - 使用标准组合模式（非增强版）
    - 在同一模型中实现两种优化策略的协同
    - 平衡计算复杂度与性能提升

    组合优势：
    =========
    - 几何建模：DCN处理变形和复杂形状
    - 注意力增强：SCC提升特征质量和选择性
    - 协同效应：两种方法的优势互补
    - 性能最优：通常获得最好的综合性能

    适用场景：
    =========
    - 对检测性能要求极高的应用
    - 需要同时处理几何变换和特征增强的复杂任务
    - 计算资源相对充足，希望获得最佳效果

    优化策略：
    =========
    - 标准模式：平衡性能和效率
    - 每层同时应用DCN和SCC优化
    - 保持模型架构的简洁性

    预期效果：
    =========
    - 显著提升检测精度和鲁棒性
    - 改善对复杂场景的处理能力
    - 获得最佳的性能-效率平衡
    """
    print("\n📋 示例3：DCN + SCC组合优化演示")
    print("=" * 50)
    print("⚡ 目标：展示DCN和SCC两种优化方法的完美结合")
    print("🎯 重点：几何建模+注意力增强的双重优化效果")

    # 导入必要的模块
    from ultralytics import YOLO
    from dcn_scc_combined.dcn_scc_c2f import replace_c2f_with_dcn_scc

    # 步骤1：加载基础模型
    print("\n📥 步骤1：加载基础模型")
    model = YOLO('yolov8n.yaml')
    print("   ✅ 基础模型加载完成")

    # 步骤2：应用组合优化
    print("\n🔄 步骤2：应用DCN+SCC组合优化")
    print("   同时集成几何建模和注意力增强")
    print("   使用标准组合模式以平衡性能和效率")
    model = replace_c2f_with_dcn_scc(model, target_channels=[256, 512], mode='standard')
    print("   ✅ DCN+SCC组合优化应用成功")

    # 步骤3：执行训练演示
    print("\n🏃 步骤3：执行训练演示")
    training_config = {
        'data': 'coco8.yaml',
        'epochs': 5,        # 演示用短周期
        'imgsz': 320,       # 较小分辨率
        'batch': 4,         # 小批量
        'project': 'results_combined_demo',
        'name': 'dcn_scc_combined_example'
    }

    print("   训练配置：")
    print("   • 优化策略: DCN + SCC组合")
    print(f"   • 优化层级: P3, P4 (通道数: 256, 512)")
    print(f"   • 组合模式: 标准模式")
    print(f"   • 预期效果: 几何建模 + 注意力增强的协同效应")

    # 开始训练
    results = model.train(**training_config)

    print("\n✅ DCN+SCC组合训练演示完成！")
    print("📊 结果分析：")
    print("   • 结合了DCN的几何建模能力和SCC的注意力增强")
    print("   • 通常能获得三种方法中最好的综合性能")
    print("   • 适合对检测精度要求极高的应用场景")
    print("   💡 提示：这是推荐的默认优化策略")

    return model

def example_4_inference_comparison():
    """
    示例4：不同优化方法的推理性能对比

    本示例对四种不同的模型配置进行推理性能测试和对比分析，
    帮助用户了解各种优化方法对推理速度的影响。

    对比对象：
    =========
    1. baseline: 原始YOLOv8n模型（无优化）
    2. dcn: 仅DCN优化的模型
    3. scc: 仅SCC优化的模型
    4. combined: DCN+SCC组合优化的模型

    测试内容：
    =========
    - 推理时间测量：单张图片的平均处理时间
    - 帧率计算：每秒可处理的图片数量
    - 性能对比：不同优化方法的效率差异
    - 设备适配：自动检测并使用最优计算设备

    性能指标：
    =========
    - 平均推理时间 (ms): 越小越好
    - FPS (帧率): 越大越好
    - 相对性能: 与baseline的对比

    技术要点：
    =========
    - 使用标准benchmark_inference函数
    - 固定输入尺寸确保公平对比
    - 多次运行取平均值保证准确性
    - 自动处理GPU同步和设备差异

    预期结果：
    =========
    - baseline: 最快速度，基础性能
    - dcn: 略微降低速度，提升几何建模
    - scc: 轻微速度影响，提升特征质量
    - combined: 最大速度影响，最佳综合性能

    应用价值：
    =========
    - 帮助选择适合的优化策略
    - 评估性能和效率的权衡
    - 为部署决策提供数据支持
    """
    print("\n📋 示例4：不同优化方法的推理性能对比")
    print("=" * 55)
    print("🔬 目标：量化分析各种优化方法对推理性能的影响")
    print("📊 重点：性能与效率的权衡分析")

    # 导入必要的模块
    from ultralytics import YOLO
    from utils import benchmark_inference

    # 步骤1：准备四种不同的模型配置
    print("\n🔧 步骤1：准备模型配置")
    models = {
        'baseline': YOLO('yolov8n.yaml'),      # 原始模型作为基准
        'dcn': YOLO('yolov8n.yaml'),          # DCN优化
        'scc': YOLO('yolov8n.yaml'),          # SCC优化
        'combined': YOLO('yolov8n.yaml')      # 组合优化
    }

    print("   创建四种模型配置：")
    print("   • baseline: 原始YOLOv8n (基准)")
    print("   • dcn: 仅DCN优化")
    print("   • scc: 仅SCC优化")
    print("   • combined: DCN+SCC组合优化")

    # 步骤2：应用相应的优化策略
    print("\n⚙️ 步骤2：应用优化策略")

    from dcn.dcn_c2f import replace_c2f_with_dcn
    from scc.scc_c2f import replace_c2f_with_scc
    from dcn_scc_combined.dcn_scc_c2f import replace_c2f_with_dcn_scc

    # 应用DCN优化
    models['dcn'] = replace_c2f_with_dcn(models['dcn'], target_channels=[256, 512])
    print("   ✅ DCN优化应用完成")

    # 应用SCC优化
    models['scc'] = replace_c2f_with_scc(models['scc'], target_channels=[256, 512], enhanced=False)
    print("   ✅ SCC优化应用完成")

    # 应用组合优化
    models['combined'] = replace_c2f_with_dcn_scc(models['combined'], target_channels=[256, 512], mode='standard')
    print("   ✅ DCN+SCC组合优化应用完成")

    # 步骤3：执行推理性能基准测试
    print("\n📈 步骤3：执行推理性能基准测试")
    print("   测试配置：")
    print("   • 输入尺寸: (1, 3, 320, 320)")
    print("   • 测试轮数: 20次")
    print("   • 指标: 平均推理时间和FPS")

    benchmark_results = {}

    for name, model in models.items():
        try:
            print(f"\n🔍 测试 {name.upper()} 模型:")
            avg_time, fps = benchmark_inference(
                model.model,
                model.device,
                input_size=(1, 3, 320, 320),  # 标准输入尺寸
                num_runs=20                     # 足够统计样本
            )

            benchmark_results[name] = {'time': avg_time, 'fps': fps}
            print(f"   📊 {name.upper():8} | 时间: {avg_time:.2f}ms | FPS: {fps:.2f}")

        except Exception as e:
            print(f"   ❌ {name.upper():8} | 测试失败: {e}")
            benchmark_results[name] = {'time': float('inf'), 'fps': 0}

    # 步骤4：结果汇总和分析
    print("\n📊 推理性能对比结果汇总")
    print("=" * 50)

    if benchmark_results:
        baseline_fps = benchmark_results['baseline']['fps']
        print("方法名称    | 推理时间 | FPS    | 相对性能")
        print("-" * 45)

        for name, result in benchmark_results.items():
            time_str = f"{result['time']:.2f}ms" if result['time'] != float('inf') else "N/A"
            fps_str = f"{result['fps']:.2f}" if result['fps'] > 0 else "N/A"

            if baseline_fps > 0 and result['fps'] > 0:
                relative_perf = result['fps'] / baseline_fps
                perf_str = f"{relative_perf:.2f}x"
            else:
                perf_str = "N/A"

            print("10")

    print("\n💡 性能分析总结:")
    print("   • baseline: 最快速度，适合对实时性要求极高的应用")
    print("   • dcn: 轻微速度损失，显著提升几何建模能力")
    print("   • scc: 最小速度影响，增强特征表达质量")
    print("   • combined: 最大性能提升，但有一定速度开销")
    print("   🎯 建议：根据应用场景平衡性能和效率需求")

    return models

def example_5_custom_training():
    """
    示例5：自定义训练配置演示

    本示例展示如何使用高级训练配置和优化策略，
    结合DCN+SCC组合优化进行高质量的模型训练。

    技术要点：
    =========
    - 使用增强版的组合优化模式
    - 自定义训练超参数配置
    - 应用现代训练技巧和优化策略
    - 平衡训练速度和模型性能

    高级配置：
    =========
    - 较大的图像尺寸：416x416提升检测精度
    - 更大的批次大小：充分利用GPU并行能力
    - AdamW优化器：更好的权重衰减特性
    - 余弦学习率调度：平滑的学习率变化
    - 预热阶段：稳定的训练初期化

    训练策略：
    =========
    - warmup_epochs: 预热阶段避免梯度爆炸
    - cos_lr: 余弦退火学习率调度
    - weight_decay: L2正则化防止过拟合
    - save_period: 定期保存检查点

    适用场景：
    =========
    - 需要高质量模型训练的应用
    - 计算资源相对充足的环境
    - 对检测精度要求较高的任务

    预期效果：
    =========
    - 获得更好的模型收敛效果
    - 提升最终检测性能
    - 改善训练稳定性和鲁棒性
    """
    print("\n📋 示例5：自定义训练配置演示")
    print("=" * 50)
    print("🔧 目标：展示高级训练配置和优化策略的使用")
    print("🎯 重点：平衡训练质量和效率的完整配置方案")

    # 导入必要的模块
    from ultralytics import YOLO
    from dcn_scc_combined.dcn_scc_c2f import replace_c2f_with_dcn_scc

    # 步骤1：加载并优化模型
    print("\n📥 步骤1：加载并优化模型")
    model = YOLO('yolov8n.yaml')
    print("   使用增强版DCN+SCC组合优化")
    model = replace_c2f_with_dcn_scc(model, target_channels=[256, 512], mode='enhanced')
    print("   ✅ 增强版组合优化应用成功")

    # 步骤2：配置高级训练参数
    print("\n⚙️ 步骤2：配置高级训练参数")
    training_config = {
        'data': 'coco8.yaml',

        # 训练周期配置
        'epochs': 10,              # 训练轮数（演示用，实际可增加）
        'warmup_epochs': 2,        # 预热轮数，避免初期梯度爆炸

        # 数据配置
        'imgsz': 416,              # 图像尺寸（比标准320更大以提升精度）
        'batch': 8,                # 批次大小（根据GPU内存调整）

        # 优化器配置
        'optimizer': 'AdamW',      # 使用AdamW优化器，更好的权重衰减
        'lr0': 0.001,              # 初始学习率
        'weight_decay': 0.0005,    # 权重衰减系数

        # 学习率调度
        'cos_lr': True,            # 启用余弦学习率调度

        # 保存配置
        'save': True,              # 保存模型权重
        'save_period': 5,          # 每5轮保存检查点

        # 项目配置
        'project': 'results_custom',     # 结果保存目录
        'name': 'custom_optimized_training'  # 实验名称
    }

    print("   高级训练配置详情：")
    print("   📊 训练设置:")
    print(f"   • 训练轮数: {training_config['epochs']} (预热: {training_config['warmup_epochs']}轮)")
    print(f"   • 图像尺寸: {training_config['imgsz']}x{training_config['imgsz']} (提升精度)")
    print(f"   • 批次大小: {training_config['batch']} (GPU并行优化)")

    print("   🎯 优化器配置:")
    print(f"   • 优化器: {training_config['optimizer']} (自适应优化)")
    print(f"   • 初始学习率: {training_config['lr0']}")
    print(f"   • 权重衰减: {training_config['weight_decay']}")

    print("   📈 学习率调度:")
    print(f"   • 余弦调度: {training_config['cos_lr']} (平滑收敛)")

    # 步骤3：执行自定义训练
    print("\n🏃 步骤3：执行自定义训练")
    print("   应用增强版组合优化 + 高级训练策略")
    print("   这将获得更好的模型性能和训练效果")

    results = model.train(**training_config)

    print("\n✅ 自定义训练演示完成！")
    print("📊 训练特色总结：")
    print("   • 增强版DCN+SCC组合优化提供最佳性能")
    print("   • 大尺寸图像训练提升检测精度")
    print("   • AdamW优化器改善收敛特性")
    print("   • 余弦学习率调度实现平滑训练")
    print("   • 预热阶段确保训练稳定性")

    print("\n💡 实际应用建议：")
    print("   • 根据GPU内存情况调整batch_size")
    print("   • 对于大数据集可增加训练轮数")
    print("   • 监控验证集性能避免过拟合")
    print("   • 定期保存最佳模型权重")
    print("   • 对于大数据集可增加训练轮数")
    print("   • 监控验证集性能避免过拟合")
    print("   • 定期保存最佳模型权重")

    return model

def main():
    """
    主函数：运行所有使用示例演示

    本函数按顺序执行所有5个示例，展示YOLOv8优化项目的完整功能。
    每个示例都独立运行，可以根据需要选择性地执行。

    执行流程：
    =========
    1. DCN基础优化示例
    2. SCC基础注意力优化示例
    3. DCN+SCC组合优化示例
    4. 推理性能对比分析
    5. 自定义高级训练配置

    输出结果：
    =========
    - 每个示例的详细执行日志
    - 训练过程中的性能指标
    - 结果文件保存路径
    - 完整的错误处理和异常报告

    注意事项：
    =========
    - 示例执行需要一定的计算资源
    - GPU环境可以显著提升执行速度
    - 每个示例都会生成结果目录
    - 可以根据需要修改示例参数

    预期时长：
    =========
    - CPU环境：约30-60分钟
    - GPU环境：约10-20分钟
    - 具体时间取决于硬件配置和训练参数
    """
    print("🚀 YOLOv8优化示例项目 - 完整使用演示")
    print("=" * 60)
    print("📚 本演示包含5个循序渐进的使用示例")
    print("🎯 涵盖从基础到高级的完整优化流程")
    print("⚡ 建议在GPU环境下运行以获得更好体验")

    try:
        print("\n📋 执行示例列表:")
        print("   1️⃣ DCN基础优化 - 几何建模能力演示")
        print("   2️⃣ SCC注意力优化 - 特征增强效果演示")
        print("   3️⃣ DCN+SCC组合 - 协同优化效果演示")
        print("   4️⃣ 性能对比分析 - 效率评估和选择指南")
        print("   5️⃣ 高级训练配置 - 最佳实践演示")

        # 依次执行所有示例
        print("\n🏁 开始执行示例演示...")

        example_1_dcn_basic()
        print("\n" + "="*60)

        example_2_scc_basic()
        print("\n" + "="*60)

        example_3_dcn_scc_combined()
        print("\n" + "="*60)

        example_4_inference_comparison()
        print("\n" + "="*60)

        example_5_custom_training()

        # 演示完成总结
        print("\n" + "=" * 60)
        print("🎉 所有示例演示执行成功完成！")
        print("📊 训练结果汇总:")
        print("   📁 results_dcn_demo/          - DCN优化结果")
        print("   📁 results_scc_demo/          - SCC优化结果")
        print("   📁 results_combined_demo/      - 组合优化结果")
        print("   📁 results_custom/             - 高级训练结果")

        print("\n📈 结果文件包含:")
        print("   • 训练日志和性能曲线")
        print("   • 模型权重文件（best.pt, last.pt）")
        print("   • 验证结果和指标数据")
        print("   • 可视化图表和分析报告")

        print("\n💡 使用建议:")
        print("   • 查看tensorboard日志了解详细训练过程")
        print("   • 使用最佳权重进行后续推理测试")
        print("   • 根据应用需求选择合适的优化策略")
        print("   • 参考性能对比结果进行部署决策")

        print("\n🚀 接下来可以:")
        print("   • 运行 run_all_examples.py 体验更多功能")
        print("   • 使用 create_optimized_model() 函数快速创建优化模型")
        print("   • 查看各模块文档了解更多技术细节")
        print("   • 在自己的数据集上应用这些优化方法")

    except KeyboardInterrupt:
        print("\n⏹️  用户中断执行")
        print("💡 可以使用 --help 参数查看更多运行选项")

    except Exception as e:
        print(f"\n❌ 示例执行失败: {e}")
        print("🔍 错误详情:")
        import traceback
        traceback.print_exc()

        print("\n🔧 故障排除建议:")
        print("   1. 检查PyTorch和Ultralytics是否正确安装")
        print("   2. 确认GPU驱动和CUDA版本兼容性")
        print("   3. 检查系统内存和磁盘空间是否充足")
        print("   4. 尝试使用更小的batch_size或imgsz参数")
        print("   5. 查看项目GitHub上的常见问题解答")

        print("\n📞 如需帮助，请查看:")
        print("   • 项目文档: README.md")
        print("   • GitHub Issues: 提交问题报告")
        print("   • 技术支持: 查看项目Wiki页面")

if __name__ == "__main__":
    main()
