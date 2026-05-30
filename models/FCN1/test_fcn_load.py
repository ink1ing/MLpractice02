#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCN数据加载测试脚本
用于测试数据加载过程中的问题
"""
import os
import sys
import torch
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加当前目录到导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入必要的模块
try:
    # 导入自定义模块
    from dataset import get_dataloaders, CornRustDataset
    print("成功导入数据集模块")
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

def test_dataset_loading(data_root="./guanceng-bit", json_root="./biaozhu_json", 
                         verbose=True, sample_limit=5):
    """测试数据集加载，逐步进行以便发现问题"""
    print(f"\n开始测试数据集加载: {data_root}, {json_root}")
    
    try:
        print("步骤1: 创建简单数据集实例...")
        dataset = CornRustDataset(
            data_dir=data_root,
            json_dir=json_root,
            img_size=128,
            transform=None
        )
        print(f"数据集创建成功，总样本数: {len(dataset)}")
        
        # 尝试访问几个样本，但不全部加载
        print("\n步骤2: 测试访问单个样本...")
        sample_count = min(sample_limit, len(dataset))
        for i in range(sample_count):
            print(f"  加载第 {i+1}/{sample_count} 个样本...")
            start_time = time.time()
            sample = dataset[i]
            load_time = time.time() - start_time
            
            if verbose:
                img, pos, grade = sample
                print(f"  样本 {i}: 图像形状={img.shape}, 位置={pos}, 等级={grade}")
                print(f"  加载时间: {load_time:.2f}秒")
        
        print("\n所有测试样本加载成功!")
        return True
    
    except Exception as e:
        print(f"数据集加载错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader(data_root="./guanceng-bit", json_root="./biaozhu_json", 
                    batch_size=1, num_workers=0, prefetch_factor=None):
    """测试DataLoader的创建和迭代"""
    print(f"\n开始测试DataLoader: batch_size={batch_size}, num_workers={num_workers}")
    
    try:
        # 使用最小的配置创建数据加载器
        print("步骤1: 创建数据加载器...")
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
        }
        
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = prefetch_factor
        
        # 创建简单数据集
        dataset = CornRustDataset(
            data_dir=data_root,
            json_dir=json_root,
            img_size=128,
            transform=None
        )
        
        print(f"数据集创建成功，样本数: {len(dataset)}")
        dataloader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
        print(f"DataLoader创建成功，批次数: {len(dataloader)}")
        
        # 尝试加载几个批次
        print("\n步骤2: 测试加载批次...")
        batch_limit = 3  # 只测试前几个批次
        
        for i, batch in enumerate(dataloader):
            if i >= batch_limit:
                break
            
            images, position_labels, grade_labels = batch
            print(f"批次 {i+1}: 图像形状={images.shape}, 位置标签形状={position_labels.shape}")
            
            # 打印批次内第一个样本的信息
            print(f"  第一个样本位置={position_labels[0]}, 等级={grade_labels[0]}")
            
        print("\n所有测试批次加载成功!")
        return True
    
    except Exception as e:
        print(f"DataLoader错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_get_dataloaders(data_root="./guanceng-bit", json_root="./biaozhu_json", 
                         batch_size=1, num_workers=0, prefetch_factor=2):
    """测试get_dataloaders函数"""
    print(f"\n开始测试get_dataloaders函数: batch_size={batch_size}, num_workers={num_workers}")
    
    try:
        print("创建训练和验证数据加载器...")
        train_loader, val_loader = get_dataloaders(
            data_root=data_root,
            json_root=json_root,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=128,
            use_extended_dataset=True,
            pin_memory=False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        
        print(f"数据加载器创建成功！")
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        
        # 尝试加载一个批次
        print("\n获取训练集第一个批次...")
        train_iter = iter(train_loader)
        train_batch = next(train_iter)
        images, position_labels, grade_labels = train_batch
        
        print(f"训练批次加载成功: 图像形状={images.shape}")
        
        # 尝试加载验证集批次
        print("\n获取验证集第一个批次...")
        val_iter = iter(val_loader)
        val_batch = next(val_iter)
        
        print("所有批次加载成功!")
        return True
    
    except Exception as e:
        print(f"get_dataloaders错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("========== FCN数据加载测试 ==========")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 执行测试
    print("\n=== 测试1: 基本数据集加载 ===")
    test_dataset_loading(verbose=True, sample_limit=3)
    
    print("\n=== 测试2: 简单DataLoader ===")
    test_dataloader(batch_size=1, num_workers=0)
    
    print("\n=== 测试3: 多线程DataLoader ===")
    test_dataloader(batch_size=2, num_workers=1, prefetch_factor=2)
    
    print("\n=== 测试4: get_dataloaders函数 ===")
    test_get_dataloaders(batch_size=2, num_workers=0)
    
    print("\n所有测试完成!") 