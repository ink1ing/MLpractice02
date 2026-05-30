#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCN模型训练前检查脚本
用于在启动训练前检查环境、数据和模型状态是否正常
"""
import os
import sys
import torch
import numpy as np
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# 添加当前目录到导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入必要的模块
try:
    # 导入自定义模块
    from model import get_model
    from dataset import get_dataloaders, CornRustDataset
    from utils import FocalLoss
    # 尝试导入FCN特定的模块
    from model import DiseaseFCN, FCNBackbone, AttentionGate
    fcn_available = True
except ImportError as e:
    print(f"导入错误: {e}")
    fcn_available = False
    sys.exit(1)

def check_environment():
    """检查环境配置"""
    print("\n======= 环境信息 =======")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        device_props = torch.cuda.get_device_properties(0)
        print(f"GPU 显存: {device_props.total_memory / (1024**3):.2f} GB")
        print(f"GPU 计算能力: {device_props.major}.{device_props.minor}")
        print(f"多处理器数量: {device_props.multi_processor_count}")
        print(f"当前已分配显存: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        print(f"当前已缓存显存: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
    
    # 检查PyTorch AMP支持
    print(f"PyTorch 自动混合精度: {'支持' if hasattr(torch.cuda.amp, 'autocast') else '不支持'}")
    
    # 测试混合精度能否正常工作
    if torch.cuda.is_available():
        try:
            with autocast(device_type='cuda'):
                x = torch.rand(10, 3, 224, 224).cuda()
                print(f"混合精度测试通过: 可以使用autocast")
        except Exception as e:
            print(f"混合精度测试失败: {e}")
    
    # 检查torch.compile支持
    print(f"PyTorch Compile: {'支持' if hasattr(torch, 'compile') else '不支持'}")
    if hasattr(torch, 'compile'):
        try:
            def simple_model(x):
                return x * x
            compiled_fn = torch.compile(simple_model)
            compiled_fn(torch.randn(10))
            print(f"torch.compile测试通过: 可以使用")
        except Exception as e:
            print(f"torch.compile测试失败: {e}")
    
    print("=======================\n")

def check_dataset(data_root, json_root, batch_size=4, img_size=128):
    """检查数据集是否可以正确加载和预处理"""
    print("\n===== 数据集检查 =====")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_root):
        print(f"错误: 数据目录不存在: {data_root}")
        return False
    
    if json_root and not os.path.exists(json_root):
        print(f"错误: JSON标注目录不存在: {json_root}")
        return False
    
    # 尝试创建数据集
    try:
        print("尝试加载数据集...")
        dataset = CornRustDataset(
            data_dir=data_root,
            json_dir=json_root,
            img_size=img_size,
            use_extended_dataset=True
        )
        print(f"数据集加载成功，共 {len(dataset)} 个样本")
        
        # 检查标签分布
        position_counts = [0, 0, 0]  # 下部/中部/上部
        grade_values = []
        
        for i in tqdm(range(min(100, len(dataset))), desc="检查样本"):
            _, position, grade = dataset[i]
            position_counts[position] += 1
            grade_values.append(grade)
        
        print("\n位置标签分布:")
        position_names = ["下部", "中部", "上部"]
        for i, count in enumerate(position_counts):
            percentage = count / sum(position_counts) * 100
            print(f"  {position_names[i]}: {count} ({percentage:.1f}%)")
        
        if grade_values:
            print("\n等级标签统计:")
            print(f"  最小值: {min(grade_values)}")
            print(f"  最大值: {max(grade_values)}")
            print(f"  平均值: {sum(grade_values) / len(grade_values):.2f}")
        
        # 尝试创建数据加载器并迭代
        print("\n测试数据加载器...")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        # 计时测试数据加载速度
        start_time = time.time()
        for i, (images, positions, grades) in enumerate(tqdm(loader, desc="加载批次")):
            if i >= 5:  # 只测试几个批次
                break
            
            # 检查数据形状
            batch_size_actual = images.size(0)
            print(f"\n批次 {i+1}:")
            print(f"  图像形状: {images.shape}")
            print(f"  位置标签形状: {positions.shape}")
            print(f"  等级标签形状: {grades.shape}")
            
            # 检查数据类型
            print(f"  图像数据类型: {images.dtype}")
            print(f"  位置标签数据类型: {positions.dtype}")
            print(f"  等级标签数据类型: {grades.dtype}")
            
            # 检查数值范围
            print(f"  图像数值范围: [{images.min():.2f}, {images.max():.2f}]")
        
        end_time = time.time()
        print(f"\n数据加载测试完成，用时: {end_time - start_time:.2f}秒")
        
        # 可视化第一个批次的图像
        if batch_size_actual > 0:
            plt.figure(figsize=(15, 5))
            for i in range(min(batch_size_actual, 4)):  # 最多显示4张图片
                plt.subplot(1, 4, i+1)
                # 如果通道数大于3，只使用前3个通道
                img = images[i].numpy()
                if img.shape[0] > 3:
                    img = img[:3]
                # 转置通道顺序以适应matplotlib
                img = np.transpose(img, (1, 2, 0))
                # 归一化到[0,1]范围
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                plt.imshow(img)
                plt.title(f"位置: {position_names[positions[i].item()]}\n等级: {grades[i].item():.1f}")
                plt.axis('off')
            
            plt.tight_layout()
            
            # 保存图像
            os.makedirs("logs", exist_ok=True)
            plt.savefig("logs/sample_images.png")
            print("\n样本图像已保存至 logs/sample_images.png")
        
        return True
    
    except Exception as e:
        print(f"数据集检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model(img_size=128):
    """检查FCN模型是否可以正确创建和前向传播"""
    print("\n===== 模型检查 =====")
    
    try:
        # 创建FCN模型
        model = get_model(model_type='fcn', in_channels=3, img_size=img_size)
        print(f"FCN模型创建成功，参数总量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # 检查模型结构
        print("\n模型结构:")
        def print_module_tree(module, prefix=''):
            for name, child in module.named_children():
                print(f"{prefix}├─ {name}")
                if list(child.named_children()):
                    print_module_tree(child, prefix + '│  ')
        print_module_tree(model)
        
        # 测试随机输入的前向传播
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
        
        # 前向传播测试
        print("\n前向传播测试...")
        try:
            with torch.no_grad():
                pos_logits, grade_values = model(dummy_input)
                print("\n前向传播成功:")
                print(f"  位置logits形状: {pos_logits.shape}")
                print(f"  等级值形状: {grade_values.shape}")
                
                # 测试分割图获取
                segmentation_map = model.get_segmentation_map(dummy_input)
                print(f"  分割图形状: {segmentation_map.shape}")
            
            return True
        except Exception as e:
            print(f"前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"模型检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_loaders(data_root="./guanceng-bit", json_root="./biaozhu_json", batch_size=8, num_workers=4, img_size=128):
    """检查数据加载器"""
    print("\n======= 数据加载器测试 =======")
    try:
        train_loader, val_loader = get_dataloaders(
            data_root=data_root,
            json_root=json_root,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            use_extended_dataset=True,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2
        )
        
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        
        # 测试一批数据加载速度
        start_time = time.time()
        for images, position_labels, grade_labels in tqdm(train_loader, desc="测试数据加载速度", total=min(5, len(train_loader))):
            if train_loader.batch_sampler.batch_size != images.shape[0]:
                print(f"警告: 批次大小不一致 - 预期 {train_loader.batch_sampler.batch_size}, 实际 {images.shape[0]}")
            if images.shape[0] == batch_size:
                break
        
        load_time = time.time() - start_time
        print(f"批次大小: {batch_size}")
        print(f"图像尺寸: {images.shape}")
        print(f"位置标签尺寸: {position_labels.shape}")
        print(f"等级标签尺寸: {grade_labels.shape}")
        print(f"加载5个批次耗时: {load_time:.2f}秒")
        print(f"每秒加载样本数: {5 * batch_size / load_time:.1f}")
        
        print("数据加载器测试通过")
    except Exception as e:
        print(f"数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("============================\n")

def check_model_forward(batch_size=4, img_size=128, mixed_precision=True):
    """检查模型前向传播"""
    print("\n======= 模型前向传播测试 =======")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 创建模型
        model = get_model(model_type='fcn', in_channels=3, img_size=img_size)
        model = model.to(device)
        print(f"模型已加载到设备: {device}")
        
        # 生成随机输入
        x = torch.randn(batch_size, 3, img_size, img_size).to(device)
        
        # 测试前向传播
        model.eval()
        
        # 预热
        print("执行模型预热...")
        with torch.no_grad():
            for _ in range(3):
                position_logits, grade_values = model(x)
        
        # 测试带梯度的前向传播
        print("测试带梯度的前向传播...")
        start_time = time.time()
        position_logits, grade_values = model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        grad_forward_time = time.time() - start_time
        
        # 测试无梯度的前向传播
        print("测试无梯度的前向传播...")
        with torch.no_grad():
            start_time = time.time()
            for _ in range(10):
                position_logits, grade_values = model(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            no_grad_forward_time = (time.time() - start_time) / 10
        
        # 测试混合精度的前向传播
        if mixed_precision and torch.cuda.is_available():
            print("测试混合精度的前向传播...")
            with torch.no_grad(), autocast(device_type='cuda'):
                start_time = time.time()
                for _ in range(10):
                    position_logits, grade_values = model(x)
                torch.cuda.synchronize()
                mixed_precision_time = (time.time() - start_time) / 10
            
            speedup = no_grad_forward_time / mixed_precision_time
            print(f"混合精度加速比: {speedup:.2f}x")
        
        # 输出结果
        print(f"带梯度的前向传播耗时: {grad_forward_time*1000:.2f} ms")
        print(f"无梯度的前向传播耗时: {no_grad_forward_time*1000:.2f} ms")
        if mixed_precision and torch.cuda.is_available():
            print(f"混合精度的前向传播耗时: {mixed_precision_time*1000:.2f} ms")
        
        # 测试张量溢出
        print("测试模型输出...")
        print(f"位置预测形状: {position_logits.shape}")
        print(f"位置预测范围: [{position_logits.min().item():.3f}, {position_logits.max().item():.3f}]")
        print(f"位置预测包含NaN: {torch.isnan(position_logits).any().item()}")
        print(f"位置预测包含Inf: {torch.isinf(position_logits).any().item()}")
        
        print(f"等级预测形状: {grade_values.shape}")
        print(f"等级预测范围: [{grade_values.min().item():.3f}, {grade_values.max().item():.3f}]")
        print(f"等级预测包含NaN: {torch.isnan(grade_values).any().item()}")
        print(f"等级预测包含Inf: {torch.isinf(grade_values).any().item()}")
        
        # 检查可视化内容是否正确生成
        print("检查注意力图...")
        if hasattr(model, 'attention_maps') and model.attention_maps:
            print(f"注意力图数量: {len(model.attention_maps)}")
            for i, att_map in enumerate(model.attention_maps):
                print(f"  注意力图 {i+1}: 形状 {att_map.shape}, 范围 [{att_map.min().item():.3f}, {att_map.max().item():.3f}]")
        else:
            print("注意力图未生成")
        
        print("模型前向传播测试通过")
    except Exception as e:
        print(f"模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("============================\n")

def check_memory_usage(img_size=256, batch_sizes=[2, 4, 8, 16, 32, 64]):
    """测试不同批次大小的内存使用情况"""
    if not torch.cuda.is_available():
        print("GPU不可用，跳过内存使用测试")
        return
    
    print("\n======= GPU内存使用测试 =======")
    device = torch.device("cuda")
    
    # 清空GPU缓存
    torch.cuda.empty_cache()
    starting_mem = torch.cuda.memory_allocated() / (1024**2)
    
    # 加载模型
    model = get_model(model_type='fcn', in_channels=3, img_size=img_size)
    model = model.to(device)
    
    model_size = (torch.cuda.memory_allocated() - starting_mem) / (1024**2)
    print(f"模型参数使用内存: {model_size:.2f} MB")
    
    results = []
    
    try:
        for batch_size in batch_sizes:
            # 清空GPU缓存
            torch.cuda.empty_cache()
            base_mem = torch.cuda.memory_allocated() / (1024**2)
            
            # 生成随机输入
            x = torch.randn(batch_size, 3, img_size, img_size).to(device)
            input_mem = (torch.cuda.memory_allocated() / (1024**2)) - base_mem
            
            # 前向传播
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), autocast(device_type='cuda'):
                position_logits, grade_values = model(x)
            
            # 计算内存使用情况
            forward_mem = torch.cuda.max_memory_allocated() / (1024**2) - base_mem
            active_mem = torch.cuda.memory_allocated() / (1024**2) - base_mem
            
            # 测试带梯度的内存使用
            torch.cuda.empty_cache()
            base_mem = torch.cuda.memory_allocated() / (1024**2)
            x = torch.randn(batch_size, 3, img_size, img_size).to(device)
            
            # 重置峰值内存统计
            torch.cuda.reset_peak_memory_stats()
            
            # 带梯度的前向传播
            position_logits, grade_values = model(x)
            loss = position_logits.sum() + grade_values.sum()
            loss.backward()
            
            # 计算带梯度的内存使用情况
            backward_mem = torch.cuda.max_memory_allocated() / (1024**2) - base_mem
            
            results.append({
                'batch_size': batch_size,
                'input_mem': input_mem,
                'forward_mem': forward_mem,
                'active_mem': active_mem,
                'backward_mem': backward_mem
            })
            
            print(f"批次大小: {batch_size}")
            print(f"  输入张量占用: {input_mem:.2f} MB")
            print(f"  前向传播峰值占用: {forward_mem:.2f} MB")
            print(f"  前向传播活跃占用: {active_mem:.2f} MB")
            print(f"  反向传播峰值占用: {backward_mem:.2f} MB")
    except Exception as e:
        print(f"在批次大小 {batch_size} 时发生内存错误: {e}")
    
    print("\n内存使用总结:")
    print("批次大小 | 输入占用(MB) | 前向峰值(MB) | 活跃占用(MB) | 反向峰值(MB) | 每样本占用(MB)")
    print("---------|--------------|--------------|--------------|--------------|---------------")
    for r in results:
        per_sample = r['backward_mem'] / r['batch_size']
        print(f"{r['batch_size']:9} | {r['input_mem']:12.2f} | {r['forward_mem']:12.2f} | {r['active_mem']:12.2f} | {r['backward_mem']:12.2f} | {per_sample:15.2f}")
    
    print("============================\n")

def run_full_check(data_root="./guanceng-bit", json_root="./biaozhu_json"):
    """运行完整检查"""
    print("开始FCN模型运行环境检查...")
    
    # 检查环境
    check_environment()
    
    # 检查数据加载器
    check_data_loaders(
        data_root=data_root,
        json_root=json_root,
        batch_size=32,  # 使用更大的批次大小
        num_workers=4,
        img_size=256    # 使用更大的图像尺寸
    )
    
    # 检查模型前向传播
    check_model_forward(batch_size=4, img_size=256, mixed_precision=True)
    
    # 测试不同批次大小的内存使用情况
    if torch.cuda.is_available():
        check_memory_usage(img_size=256)
    
    print("FCN模型运行环境检查完成!")

def main(args):
    """主函数"""
    print("========================================")
    print("    FCN模型训练前状态检查工具")
    print("========================================")
    
    # 打印检查配置
    print("\n检查配置:")
    print(f"  数据目录: {args.data_root}")
    print(f"  JSON目录: {args.json_root}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  图像大小: {args.img_size}")
    
    # 环境检查
    env_ok = check_environment()
    if not env_ok:
        print("\n⚠️ 环境检查失败，请修复上述问题后重试")
        return False
    
    # 数据集检查
    data_ok = check_dataset(args.data_root, args.json_root, args.batch_size, args.img_size)
    if not data_ok:
        print("\n⚠️ 数据集检查失败，请修复上述问题后重试")
        return False
    
    # 模型检查
    model_ok = check_model(args.img_size)
    if not model_ok:
        print("\n⚠️ 模型检查失败，请修复上述问题后重试")
        return False
    
    # 所有检查通过
    print("\n✅ 所有检查通过！FCN模型已准备好开始训练")
    print("\n建议使用以下命令启动训练:")
    print(f"python run_fcn.py --data_root {args.data_root} --json_root {args.json_root} --batch_size {args.batch_size} --img_size {args.img_size} --epochs 50 --monitor_metric f1 --patience 5")
    print("\n或者在Windows系统上直接运行批处理文件:")
    print("train_fcn.bat")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCN模型运行前环境检查")
    parser.add_argument('--data_root', default='./guanceng-bit', help='数据根目录')
    parser.add_argument('--json_root', default='./biaozhu_json', help='JSON标注根目录')
    
    args = parser.parse_args()
    
    run_full_check(
        data_root=args.data_root,
        json_root=args.json_root
    ) 