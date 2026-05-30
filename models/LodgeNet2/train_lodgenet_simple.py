#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 参考文件：简化版LodgeNet训练脚本，作为完整版训练脚本的基础
# LodgeNet简化训练脚本：类似README中的命令格式，专注核心训练流程

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
from tqdm import tqdm
import time
import json
from datetime import datetime

# 导入自定义模块
from dataset import get_dataloaders
from lodgenet_model import get_lodgenet_model, count_parameters
from utils import FocalLoss

def create_dummy_segmentation_labels(position_labels, grade_labels, img_size=128):
    """创建虚拟分割标签"""
    batch_size = position_labels.size(0)
    seg_labels = torch.zeros(batch_size, img_size, img_size, dtype=torch.long)
    
    for i in range(batch_size):
        grade = grade_labels[i].item()
        if grade > 0:  # 有感染
            center_size = int(img_size * 0.3 * (grade / 4.0))  # 根据等级调整大小
            if center_size > 0:
                start = (img_size - center_size) // 2
                end = start + center_size
                seg_labels[i, start:end, start:end] = 1
    
    return seg_labels

def train_epoch(model, train_loader, optimizer, device, epoch, total_epochs):
    """训练一个epoch"""
    model.train()
    
    # 定义损失函数
    seg_criterion = nn.CrossEntropyLoss()
    position_criterion = FocalLoss(alpha=None, gamma=2.0)
    grade_criterion = nn.MSELoss()
    
    total_loss = 0.0
    position_correct = 0
    total_samples = 0
    grade_mae_sum = 0.0
    
    # 创建进度条
    progress_bar = tqdm(train_loader, desc=f"训练中 Epoch {epoch}/{total_epochs}")
    
    for batch_idx, (images, position_labels, grade_labels) in enumerate(progress_bar):
        # 数据移动到设备
        images = images.to(device, non_blocking=True)
        position_labels = position_labels.view(-1).long().to(device, non_blocking=True)
        grade_labels = grade_labels.float().unsqueeze(1).to(device, non_blocking=True)
        
        # 创建虚拟分割标签
        seg_labels = create_dummy_segmentation_labels(
            position_labels, grade_labels.squeeze(1), images.size(-1)
        ).to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 前向传播
        seg_output, position_logits, grade_output = model(images)
        
        # 计算损失
        loss_seg = seg_criterion(seg_output, seg_labels)
        loss_position = position_criterion(position_logits, position_labels)
        loss_grade = grade_criterion(grade_output, grade_labels)
        
        # 组合损失
        total_task_loss = 0.4 * loss_seg + 0.3 * loss_position + 0.3 * loss_grade
        
        # 反向传播
        total_task_loss.backward()
        optimizer.step()
        
        # 统计指标
        batch_size = images.size(0)
        total_loss += total_task_loss.item() * batch_size
        
        # 位置分类准确率
        _, position_preds = torch.max(position_logits, 1)
        position_correct += (position_preds == position_labels).sum().item()
        
        # 等级回归MAE
        grade_mae = torch.abs(grade_output - grade_labels).mean().item()
        grade_mae_sum += grade_mae * batch_size
        
        total_samples += batch_size
        
        # 更新进度条
        current_acc = position_correct / total_samples
        current_mae = grade_mae_sum / total_samples
        progress_bar.set_postfix({
            'loss': f'{total_task_loss.item():.3f}',
            'pos_acc': f'{current_acc:.3f}',
            'grade_mae': f'{current_mae:.3f}'
        })
    
    return {
        'loss': total_loss / total_samples,
        'position_accuracy': position_correct / total_samples,
        'grade_mae': grade_mae_sum / total_samples
    }

def validate_epoch(model, val_loader, device):
    """验证一个epoch"""
    model.eval()
    
    # 定义损失函数
    seg_criterion = nn.CrossEntropyLoss()
    position_criterion = FocalLoss(alpha=None, gamma=2.0)
    grade_criterion = nn.MSELoss()
    
    total_loss = 0.0
    position_correct = 0
    total_samples = 0
    grade_mae_sum = 0.0
    
    with torch.no_grad():
        for images, position_labels, grade_labels in val_loader:
            images = images.to(device, non_blocking=True)
            position_labels = position_labels.view(-1).long().to(device, non_blocking=True)
            grade_labels = grade_labels.float().unsqueeze(1).to(device, non_blocking=True)
            
            # 创建虚拟分割标签
            seg_labels = create_dummy_segmentation_labels(
                position_labels, grade_labels.squeeze(1), images.size(-1)
            ).to(device, non_blocking=True)
            
            # 前向传播
            seg_output, position_logits, grade_output = model(images)
            
            # 计算损失
            loss_seg = seg_criterion(seg_output, seg_labels)
            loss_position = position_criterion(position_logits, position_labels)
            loss_grade = grade_criterion(grade_output, grade_labels)
            
            total_task_loss = 0.4 * loss_seg + 0.3 * loss_position + 0.3 * loss_grade
            
            batch_size = images.size(0)
            total_loss += total_task_loss.item() * batch_size
            
            # 位置分类准确率
            _, position_preds = torch.max(position_logits, 1)
            position_correct += (position_preds == position_labels).sum().item()
            
            # 等级回归MAE
            grade_mae = torch.abs(grade_output - grade_labels).mean().item()
            grade_mae_sum += grade_mae * batch_size
            
            total_samples += batch_size
    
    return {
        'loss': total_loss / total_samples,
        'position_accuracy': position_correct / total_samples,
        'grade_mae': grade_mae_sum / total_samples
    }

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='LodgeNet训练脚本')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='TIF图像数据根目录')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注数据根目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作进程数')
    
    # 模型参数
    parser.add_argument('--img_size', type=int, default=128,
                        help='输入图像尺寸')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./output_lodgenet_simple',
                        help='输出目录')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LodgeNet训练开始")
    print("=" * 60)
    print(f"数据根目录: {args.data_root}")
    print(f"JSON根目录: {args.json_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"图像尺寸: {args.img_size}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 获取数据加载器
    print("\n加载数据集...")
    try:
        train_loader, val_loader = get_dataloaders(
            data_root=args.data_root,
            json_root=args.json_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            train_ratio=0.8,
            use_extended_dataset=True
        )
        
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模型
    print("\n创建LodgeNet模型...")
    try:
        model = get_lodgenet_model(
            n_channels=3,
            n_classes=2,
            img_size=args.img_size,
            bilinear=True
        ).to(device)
        
        print(f"模型参数数量: {count_parameters(model):,}")
        
    except Exception as e:
        print(f"模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # 训练历史记录
    train_history = []
    val_history = []
    best_val_loss = float('inf')
    
    print(f"\n开始训练 {args.num_epochs} 轮...")
    print("=" * 60)
    
    for epoch in range(1, args.num_epochs + 1):
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, args.num_epochs)
        
        # 验证
        print("验证中...")
        val_metrics = validate_epoch(model, val_loader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        train_history.append(train_metrics)
        val_history.append(val_metrics)
        
        # 打印结果
        print(f"\n轮次 {epoch} 完成")
        print(f"训练指标: 损失={train_metrics['loss']:.4f}, "
              f"位置准确率={train_metrics['position_accuracy']:.4f}, "
              f"等级MAE={train_metrics['grade_mae']:.4f}")
        print(f"验证指标: 损失={val_metrics['loss']:.4f}, "
              f"位置准确率={val_metrics['position_accuracy']:.4f}, "
              f"等级MAE={val_metrics['grade_mae']:.4f}")
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"保存最佳模型 (验证损失: {best_val_loss:.4f})")
        
        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }, os.path.join(args.output_dir, 'last_model.pth'))
        
        print("-" * 60)
    
    # 保存训练历史
    history = {
        'train': train_history,
        'val': val_history,
        'args': vars(args)
    }
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"训练结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 