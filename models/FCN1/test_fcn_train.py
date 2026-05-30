#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCN模型训练测试脚本
用于验证FCN模型训练过程，只训练少量批次
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler
import time

# 导入自定义模块
from dataset import get_dataloaders, CornRustDataset
from model import get_model
from utils import FocalLoss

def test_train_fcn(args):
    """
    测试FCN模型训练过程
    
    参数:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        json_root=args.json_root,
        batch_size=args.batch_size,
        num_workers=1,  # 使用1个工作线程进行测试
        img_size=args.img_size,
        use_extended_dataset=True,
        pin_memory=torch.cuda.is_available()
    )
    
    # 创建FCN模型
    model = get_model(
        model_type='fcn',
        in_channels=3,
        img_size=args.img_size
    ).to(device)
    
    # 定义损失函数
    position_criterion = FocalLoss(gamma=2.0)
    grade_criterion = nn.MSELoss()
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 定义混合精度训练的Grad Scaler
    scaler = GradScaler()
    
    # 任务权重 - 位置分类和等级回归的权重
    task_weights = [0.7, 0.3]
    
    # 开始测试训练
    print("开始测试训练...")
    model.train()
    
    # 训练计数器
    position_correct = 0
    total_samples = 0
    
    # 获取少量批次进行测试
    max_batches = min(5, len(train_loader))
    
    # 使用tqdm创建进度条
    train_pbar = tqdm(enumerate(train_loader), total=max_batches, desc="训练中")
    
    # 计时器
    data_time = 0.0
    compute_time = 0.0
    batch_times = []
    data_times = []
    compute_times = []
    
    end = time.time()
    for i, (images, position_labels, grade_labels) in train_pbar:
        if i >= max_batches:
            break
        
        # 记录数据加载时间
        data_time = time.time() - end
        data_times.append(data_time)
        
        # 将数据移至设备
        images = images.to(device)
        position_labels = position_labels.to(device).long()
        grade_labels = grade_labels.float().unsqueeze(1).to(device)
        
        # 标记计算开始时间
        compute_start = time.time()
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 使用混合精度训练
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            # 前向传播
            position_logits, grade_values = model(images)
            
            # 计算损失
            pos_loss = position_criterion(position_logits, position_labels)
            grade_loss = grade_criterion(grade_values, grade_labels)
            loss = task_weights[0] * pos_loss + task_weights[1] * grade_loss
        
        # 反向传播和优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 记录计算时间
        compute_time = time.time() - compute_start
        compute_times.append(compute_time)
        
        # 计算训练指标
        batch_size = images.size(0)
        
        _, position_preds = torch.max(position_logits, 1)
        position_correct += (position_preds == position_labels).sum().item()
        
        grade_mae = torch.abs(grade_values - grade_labels).mean().item()
        
        total_samples += batch_size
        
        # 计算总批次时间
        batch_time = time.time() - end
        batch_times.append(batch_time)
        
        # 更新进度条
        train_pbar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'pos_acc': f"{position_correct / total_samples:.3f}",
            'data_time': f"{data_time:.3f}s",
            'compute_time': f"{compute_time:.3f}s"
        })
        
        # 准备下一批次的计时
        end = time.time()
        
        # 每次批次后都显示一次性能分析
        if len(batch_times) >= 1:
            recent_batch_times = batch_times[-min(len(batch_times), 3):]
            recent_data_times = data_times[-min(len(data_times), 3):]
            recent_compute_times = compute_times[-min(len(compute_times), 3):]
            
            avg_batch_time = sum(recent_batch_times) / len(recent_batch_times)
            avg_data_time = sum(recent_data_times) / len(recent_data_times)
            avg_compute_time = sum(recent_compute_times) / len(recent_compute_times)
            
            samples_per_sec = batch_size / avg_batch_time
            
            print("\n性能分析 (最近批次):")
            print(f"  平均数据加载时间: {avg_data_time:.3f}秒")
            print(f"  平均计算时间: {avg_compute_time:.3f}秒")
            print(f"  平均总批次时间: {avg_batch_time:.3f}秒")
            print(f"  当前性能: {samples_per_sec:.1f} 样本/秒")
    
    # 在测试训练结束后显示性能统计
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_data_time = sum(data_times) / len(data_times)
        avg_compute_time = sum(compute_times) / len(compute_times)
        
        data_percentage = avg_data_time / avg_batch_time * 100
        compute_percentage = avg_compute_time / avg_batch_time * 100
        
        samples_per_sec = batch_size / avg_batch_time
        
        print("\n训练性能统计:")
        print(f"  平均批次时间: {avg_batch_time:.3f}秒")
        print(f"  - 数据加载: {avg_data_time:.3f}秒 ({data_percentage:.1f}%)")
        print(f"  - 计算: {avg_compute_time:.3f}秒 ({compute_percentage:.1f}%)")
        print(f"  样本吞吐量: {samples_per_sec:.1f} 样本/秒")
    
    # 测试验证过程
    print("\n测试验证过程...")
    model.eval()
    
    # 验证计数器
    val_position_correct = 0
    val_total_samples = 0
    
    # 获取少量批次进行测试
    max_val_batches = min(3, len(val_loader))
    
    # 使用tqdm创建进度条
    val_pbar = tqdm(enumerate(val_loader), total=max_val_batches, desc="验证中")
    
    with torch.no_grad():
        for i, (images, position_labels, grade_labels) in val_pbar:
            if i >= max_val_batches:
                break
            
            # 将数据移至设备
            images = images.to(device)
            position_labels = position_labels.to(device).long()
            grade_labels = grade_labels.float().unsqueeze(1).to(device)
            
            # 使用混合精度，但不计算梯度
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # 前向传播
                position_logits, grade_values = model(images)
                
                # 计算损失
                pos_loss = position_criterion(position_logits, position_labels)
                grade_loss = grade_criterion(grade_values, grade_labels)
                loss = task_weights[0] * pos_loss + task_weights[1] * grade_loss
            
            # 计算验证指标
            batch_size = images.size(0)
            
            _, position_preds = torch.max(position_logits, 1)
            val_position_correct += (position_preds == position_labels).sum().item()
            
            val_total_samples += batch_size
            
            # 更新进度条
            val_pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'pos_acc': f"{val_position_correct / val_total_samples:.3f}"
            })
    
    print("\n测试训练和验证过程完成！")
    print(f"训练准确率: {position_correct / total_samples:.4f}")
    print(f"验证准确率: {val_position_correct / val_total_samples:.4f}")
    
    # 返回测试结果
    return {
        'train_accuracy': position_correct / total_samples,
        'val_accuracy': val_position_correct / val_total_samples
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FCN模型训练测试脚本')
    parser.add_argument('--data_root', type=str, default='./guanceng-bit', help='多光谱图像数据目录')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json', help='标注JSON文件目录')
    parser.add_argument('--output_dir', type=str, default='./output_fcn_test', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--img_size', type=int, default=128, help='图像大小')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU索引')
    
    args = parser.parse_args()
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行测试
    result = test_train_fcn(args)
    
    # 计算总运行时间
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    
    # 输出最终结果
    print("\n" + "="*50)
    print("测试完成!")
    print(f"总运行时间: {int(minutes)}分 {seconds:.2f}秒")
    print("="*50) 