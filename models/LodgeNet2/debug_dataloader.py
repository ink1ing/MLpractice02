#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 旧版本文件：数据加载器调试脚本，逐步检查数据加载问题

import os
import sys
import torch
import time
import threading
from dataset import get_dataloaders

def timeout_handler():
    print("\n超时警告: 操作超过30秒")
    print("强制退出...")
    os._exit(1)

def test_dataloader_step_by_step():
    """逐步测试数据加载器"""
    print("=" * 80)
    print("数据加载器逐步调试")
    print("=" * 80)
    
    # 测试不同的配置
    test_configs = [
        {"batch_size": 1, "num_workers": 0, "img_size": 128},
        {"batch_size": 2, "num_workers": 0, "img_size": 128},
        {"batch_size": 4, "num_workers": 1, "img_size": 128},
        {"batch_size": 8, "num_workers": 2, "img_size": 128},
        {"batch_size": 16, "num_workers": 4, "img_size": 128},
        {"batch_size": 32, "num_workers": 4, "img_size": 256},
        {"batch_size": 64, "num_workers": 8, "img_size": 256},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n测试配置 {i+1}: {config}")
        
        # 设置30秒超时
        timer = threading.Timer(30.0, timeout_handler)
        timer.start()
        
        try:
            start_time = time.time()
            
            # 创建数据加载器
            print("  创建数据加载器...")
            train_loader, val_loader = get_dataloaders(
                data_root='./guanceng-bit',
                json_root='./biaozhu_json',
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                img_size=config['img_size'],
                train_ratio=0.8,
                use_extended_dataset=True
            )
            
            create_time = time.time() - start_time
            print(f"  数据加载器创建成功，耗时: {create_time:.2f}秒")
            
            # 测试获取第一个批次
            print("  获取第一个批次...")
            iter_start_time = time.time()
            
            data_iter = iter(train_loader)
            images, position_labels, grade_labels = next(data_iter)
            
            iter_time = time.time() - iter_start_time
            print(f"  第一个批次获取成功，耗时: {iter_time:.2f}秒")
            print(f"    图像形状: {images.shape}")
            print(f"    位置标签形状: {position_labels.shape}")
            print(f"    等级标签形状: {grade_labels.shape}")
            
            # 测试获取第二个批次
            print("  获取第二个批次...")
            second_start_time = time.time()
            
            images2, position_labels2, grade_labels2 = next(data_iter)
            
            second_time = time.time() - second_start_time
            print(f"  第二个批次获取成功，耗时: {second_time:.2f}秒")
            
            total_time = time.time() - start_time
            print(f"  配置测试成功，总耗时: {total_time:.2f}秒")
            
            timer.cancel()
            
            # 如果这个配置成功，继续测试下一个
            del train_loader, val_loader, data_iter
            torch.cuda.empty_cache()
            
        except Exception as e:
            timer.cancel()
            print(f"  配置测试失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 清理资源
            try:
                del train_loader, val_loader
                torch.cuda.empty_cache()
            except:
                pass
            
            print(f"  跳过此配置，继续测试...")
            continue

def test_simple_dataloader():
    """测试最简单的数据加载器配置"""
    print("\n" + "=" * 80)
    print("测试最简单配置")
    print("=" * 80)
    
    # 设置60秒超时
    timer = threading.Timer(60.0, timeout_handler)
    timer.start()
    
    try:
        print("配置: batch_size=1, num_workers=0, img_size=128")
        
        start_time = time.time()
        train_loader, val_loader = get_dataloaders(
            data_root='./guanceng-bit',
            json_root='./biaozhu_json',
            batch_size=1,
            num_workers=0,
            img_size=128,
            train_ratio=0.8,
            use_extended_dataset=True
        )
        
        create_time = time.time() - start_time
        print(f"数据加载器创建成功，耗时: {create_time:.2f}秒")
        
        print("开始获取第一个批次...")
        iter_start_time = time.time()
        
        data_iter = iter(train_loader)
        print("迭代器创建成功")
        
        images, position_labels, grade_labels = next(data_iter)
        
        iter_time = time.time() - iter_start_time
        print(f"第一个批次获取成功！耗时: {iter_time:.2f}秒")
        print(f"图像形状: {images.shape}")
        print(f"位置标签形状: {position_labels.shape}")
        print(f"等级标签形状: {grade_labels.shape}")
        
        timer.cancel()
        return True
        
    except Exception as e:
        timer.cancel()
        print(f"简单配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_directly():
    """直接测试数据集，不使用DataLoader"""
    print("\n" + "=" * 80)
    print("直接测试数据集")
    print("=" * 80)
    
    try:
        from dataset import CornRustDataset
        
        print("创建数据集...")
        dataset = CornRustDataset(
            data_root='./guanceng-bit',
            json_root='./biaozhu_json',
            img_size=128,
            use_extended_dataset=True
        )
        
        print(f"数据集创建成功，样本数: {len(dataset)}")
        
        print("获取第一个样本...")
        start_time = time.time()
        
        sample = dataset[0]
        
        load_time = time.time() - start_time
        print(f"第一个样本获取成功，耗时: {load_time:.2f}秒")
        
        if isinstance(sample, tuple) and len(sample) == 3:
            image, position_label, grade_label = sample
            print(f"图像形状: {image.shape}")
            print(f"位置标签: {position_label}")
            print(f"等级标签: {grade_label}")
        else:
            print(f"样本格式: {type(sample)}")
        
        print("获取第二个样本...")
        start_time = time.time()
        
        sample2 = dataset[1]
        
        load_time2 = time.time() - start_time
        print(f"第二个样本获取成功，耗时: {load_time2:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"数据集直接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始数据加载器调试...")
    
    # 1. 直接测试数据集
    if not test_dataset_directly():
        print("数据集直接测试失败，无法继续")
        return
    
    # 2. 测试最简单的数据加载器
    if not test_simple_dataloader():
        print("简单数据加载器测试失败")
        return
    
    # 3. 逐步测试不同配置
    test_dataloader_step_by_step()
    
    print("\n" + "=" * 80)
    print("调试完成")
    print("=" * 80)

if __name__ == "__main__":
    main() 