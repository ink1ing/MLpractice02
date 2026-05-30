# 旧版本文件：修复版数据集测试脚本，验证数据集修复效果
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 测试修复版本数据集

import os
import sys
import torch
import time
from dataset_fixed import CornRustDataset, get_dataloaders

def test_fixed_dataset():
    """测试修复版本的数据集"""
    print("=" * 60)
    print("测试修复版本数据集")
    print("=" * 60)
    
    try:
        print("1. 测试数据集创建...")
        start_time = time.time()
        
        dataset = CornRustDataset(
            data_root='./guanceng-bit',
            json_root='./biaozhu_json',
            img_size=128,
            use_extended_dataset=True
        )
        
        create_time = time.time() - start_time
        print(f"   数据集创建成功，耗时: {create_time:.2f}秒")
        print(f"   数据集大小: {len(dataset)}")
        
        print("\n2. 测试单个样本获取...")
        start_time = time.time()
        
        sample = dataset[0]
        
        load_time = time.time() - start_time
        print(f"   第一个样本获取成功，耗时: {load_time:.2f}秒")
        
        if isinstance(sample, tuple) and len(sample) == 3:
            image, position_label, grade_label = sample
            print(f"   图像形状: {image.shape}")
            print(f"   位置标签: {position_label}")
            print(f"   等级标签: {grade_label}")
        
        print("\n3. 测试多个样本获取...")
        start_time = time.time()
        
        for i in range(5):
            sample = dataset[i]
            
        multi_load_time = time.time() - start_time
        print(f"   5个样本获取成功，耗时: {multi_load_time:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"   数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fixed_dataloader():
    """测试修复版本的数据加载器"""
    print("\n" + "=" * 60)
    print("测试修复版本数据加载器")
    print("=" * 60)
    
    try:
        print("1. 创建数据加载器...")
        start_time = time.time()
        
        train_loader, val_loader = get_dataloaders(
            data_root='./guanceng-bit',
            json_root='./biaozhu_json',
            batch_size=4,
            num_workers=0,  # 使用单线程避免问题
            img_size=128,
            train_ratio=0.8,
            use_extended_dataset=True
        )
        
        create_time = time.time() - start_time
        print(f"   数据加载器创建成功，耗时: {create_time:.2f}秒")
        print(f"   训练批次数: {len(train_loader)}")
        print(f"   验证批次数: {len(val_loader)}")
        
        print("\n2. 测试第一个批次...")
        start_time = time.time()
        
        data_iter = iter(train_loader)
        images, position_labels, grade_labels = next(data_iter)
        
        batch_time = time.time() - start_time
        print(f"   第一个批次获取成功，耗时: {batch_time:.2f}秒")
        print(f"   批次图像形状: {images.shape}")
        print(f"   批次位置标签形状: {position_labels.shape}")
        print(f"   批次等级标签形状: {grade_labels.shape}")
        
        print("\n3. 测试第二个批次...")
        start_time = time.time()
        
        images2, position_labels2, grade_labels2 = next(data_iter)
        
        batch_time2 = time.time() - start_time
        print(f"   第二个批次获取成功，耗时: {batch_time2:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"   数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始测试修复版本数据集...")
    
    # 1. 测试数据集
    if not test_fixed_dataset():
        print("数据集测试失败")
        return
    
    # 2. 测试数据加载器
    if not test_fixed_dataloader():
        print("数据加载器测试失败")
        return
    
    print("\n" + "=" * 60)
    print("所有测试通过！修复版本工作正常")
    print("=" * 60)

if __name__ == "__main__":
    main() 
