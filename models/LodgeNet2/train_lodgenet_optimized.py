#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 参考文件：优化版LodgeNet训练脚本，含性能优化，被最终版本替代
# LodgeNet优化训练脚本：解决训练卡住问题，增强性能监控

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
import threading
import signal

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
            center_size = int(img_size * 0.3 * (grade / 4.0))
            if center_size > 0:
                start = (img_size - center_size) // 2
                end = start + center_size
                seg_labels[i, start:end, start:end] = 1
    
    return seg_labels

class TimeoutHandler:
    """超时处理器，防止训练卡死"""
    def __init__(self, timeout_seconds=300):  # 5分钟超时
        self.timeout_seconds = timeout_seconds
        self.timer = None
        
    def start_timer(self, message="操作超时"):
        def timeout_handler():
            print(f"\n警告: {message}")
            print("可能的原因：")
            print("1. 数据加载器卡住")
            print("2. GPU内存不足")
            print("3. 模型前向传播问题")
            print("4. 网络I/O问题")
            
        self.timer = threading.Timer(self.timeout_seconds, timeout_handler)
        self.timer.start()
        
    def stop_timer(self):
        if self.timer:
            self.timer.cancel()

def test_single_batch(model, train_loader, device):
    """测试单个批次，确保训练循环能正常工作"""
    print("\n检查: 测试单个批次...")
    timeout_handler = TimeoutHandler(60)  # 1分钟超时
    
    try:
        timeout_handler.start_timer("单批次测试超时")
        
        # 获取第一个批次
        data_iter = iter(train_loader)
        images, position_labels, grade_labels = next(data_iter)
        
        print(f"成功: 数据加载成功")
        print(f"   图像形状: {images.shape}")
        print(f"   位置标签形状: {position_labels.shape}")
        print(f"   等级标签形状: {grade_labels.shape}")
        
        # 移动到设备
        images = images.to(device, non_blocking=True)
        position_labels = position_labels.view(-1).long().to(device, non_blocking=True)
        grade_labels = grade_labels.float().unsqueeze(1).to(device, non_blocking=True)
        
        print(f"成功: 数据移动到设备成功")
        
        # 创建虚拟分割标签
        seg_labels = create_dummy_segmentation_labels(
            position_labels, grade_labels.squeeze(1), images.size(-1)
        ).to(device, non_blocking=True)
        
        print(f"成功: 分割标签创建成功，形状: {seg_labels.shape}")
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            seg_output, position_logits, grade_output = model(images)
            
        print(f"成功: 前向传播成功")
        print(f"   分割输出形状: {seg_output.shape}")
        print(f"   位置输出形状: {position_logits.shape}")
        print(f"   等级输出形状: {grade_output.shape}")
        
        # 测试损失计算
        seg_criterion = nn.CrossEntropyLoss()
        position_criterion = FocalLoss(alpha=None, gamma=2.0)
        grade_criterion = nn.MSELoss()
        
        loss_seg = seg_criterion(seg_output, seg_labels)
        loss_position = position_criterion(position_logits, position_labels)
        loss_grade = grade_criterion(grade_output, grade_labels)
        
        total_loss = 0.4 * loss_seg + 0.3 * loss_position + 0.3 * loss_grade
        
        print(f"成功: 损失计算成功")
        print(f"   分割损失: {loss_seg.item():.4f}")
        print(f"   位置损失: {loss_position.item():.4f}")
        print(f"   等级损失: {loss_grade.item():.4f}")
        print(f"   总损失: {total_loss.item():.4f}")
        
        timeout_handler.stop_timer()
        return True
        
    except Exception as e:
        timeout_handler.stop_timer()
        print(f"错误: 单批次测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_epoch_with_monitoring(model, train_loader, optimizer, device, epoch, total_epochs):
    """带监控的训练epoch"""
    model.train()
    
    # 定义损失函数
    seg_criterion = nn.CrossEntropyLoss()
    position_criterion = FocalLoss(alpha=None, gamma=2.0)
    grade_criterion = nn.MSELoss()
    
    total_loss = 0.0
    position_correct = 0
    total_samples = 0
    grade_mae_sum = 0.0
    
    # 性能监控
    batch_times = []
    data_times = []
    compute_times = []
    
    print(f"\n开始训练 Epoch {epoch}/{total_epochs}")
    print(f"   批次数: {len(train_loader)}")
    print(f"   批次大小: {train_loader.batch_size}")
    
    # 创建进度条
    progress_bar = tqdm(train_loader, desc=f"训练中 Epoch {epoch}/{total_epochs}")
    
    data_start_time = time.time()
    
    for batch_idx, (images, position_labels, grade_labels) in enumerate(progress_bar):
        batch_start_time = time.time()
        data_load_time = batch_start_time - data_start_time
        data_times.append(data_load_time)
        
        try:
            # 数据预处理
            images = images.to(device, non_blocking=True)
            position_labels = position_labels.view(-1).long().to(device, non_blocking=True)
            grade_labels = grade_labels.float().unsqueeze(1).to(device, non_blocking=True)
            
            # 创建虚拟分割标签
            seg_labels = create_dummy_segmentation_labels(
                position_labels, grade_labels.squeeze(1), images.size(-1)
            ).to(device, non_blocking=True)
            
            # 前向传播和反向传播
            compute_start_time = time.time()
            
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
            
            compute_time = time.time() - compute_start_time
            compute_times.append(compute_time)
            
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
            
            # 记录批次时间
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # 更新进度条
            current_acc = position_correct / total_samples
            current_mae = grade_mae_sum / total_samples
            progress_bar.set_postfix({
                'loss': f'{total_task_loss.item():.3f}',
                'pos_acc': f'{current_acc:.3f}',
                'grade_mae': f'{current_mae:.3f}',
                'data_time': f'{data_load_time:.3f}s'
            })
            
            # 每10个批次输出性能统计
            if (batch_idx + 1) % 10 == 0:
                recent_batches = min(10, len(batch_times))
                avg_data_time = sum(data_times[-recent_batches:]) / recent_batches
                avg_compute_time = sum(compute_times[-recent_batches:]) / recent_batches
                avg_batch_time = sum(batch_times[-recent_batches:]) / recent_batches
                throughput = recent_batches * batch_size / sum(batch_times[-recent_batches:])
                
                print(f"\n性能分析 (最近{recent_batches}个批次):")
                print(f"  平均数据加载时间: {avg_data_time:.3f}秒")
                print(f"  平均计算时间: {avg_compute_time:.3f}秒")
                print(f"  平均批次时间: {avg_batch_time:.3f}秒")
                print(f"  样本吞吐量: {throughput:.1f} 样本/秒")
                
                # GPU内存使用情况
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"  GPU内存: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
            
        except Exception as e:
            print(f"\n错误: 批次 {batch_idx} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 重置数据加载计时器
        data_start_time = time.time()
    
    # 计算epoch平均指标
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_accuracy = position_correct / total_samples if total_samples > 0 else 0
    avg_mae = grade_mae_sum / total_samples if total_samples > 0 else 0
    
    # 输出epoch统计
    if batch_times:
        total_epoch_time = sum(batch_times)
        avg_batch_time = total_epoch_time / len(batch_times)
        avg_data_time = sum(data_times) / len(data_times)
        avg_compute_time = sum(compute_times) / len(compute_times)
        
        print(f"\n轮次 {epoch} 完成，耗时: {total_epoch_time:.2f}秒")
        print(f"训练指标: 损失={avg_loss:.4f}, 位置准确率={avg_accuracy:.4f}, 等级MAE={avg_mae:.4f}")
        print(f"性能统计:")
        print(f"  平均批次时间: {avg_batch_time:.3f}秒")
        print(f"  - 数据加载: {avg_data_time:.3f}秒 ({avg_data_time/avg_batch_time*100:.1f}%)")
        print(f"  - 计算: {avg_compute_time:.3f}秒 ({avg_compute_time/avg_batch_time*100:.1f}%)")
        print(f"  样本吞吐量: {total_samples/total_epoch_time:.1f} 样本/秒")
    
    return {
        'loss': avg_loss,
        'position_accuracy': avg_accuracy,
        'grade_mae': avg_mae
    }

def validate_epoch(model, val_loader, device):
    """验证epoch"""
    model.eval()
    
    seg_criterion = nn.CrossEntropyLoss()
    position_criterion = FocalLoss(alpha=None, gamma=2.0)
    grade_criterion = nn.MSELoss()
    
    total_loss = 0.0
    position_correct = 0
    total_samples = 0
    grade_mae_sum = 0.0
    
    print("验证中...")
    with torch.no_grad():
        for images, position_labels, grade_labels in tqdm(val_loader, desc="验证中"):
            images = images.to(device, non_blocking=True)
            position_labels = position_labels.view(-1).long().to(device, non_blocking=True)
            grade_labels = grade_labels.float().unsqueeze(1).to(device, non_blocking=True)
            
            seg_labels = create_dummy_segmentation_labels(
                position_labels, grade_labels.squeeze(1), images.size(-1)
            ).to(device, non_blocking=True)
            
            seg_output, position_logits, grade_output = model(images)
            
            loss_seg = seg_criterion(seg_output, seg_labels)
            loss_position = position_criterion(position_logits, position_labels)
            loss_grade = grade_criterion(grade_output, grade_labels)
            
            total_task_loss = 0.4 * loss_seg + 0.3 * loss_position + 0.3 * loss_grade
            
            batch_size = images.size(0)
            total_loss += total_task_loss.item() * batch_size
            
            _, position_preds = torch.max(position_logits, 1)
            position_correct += (position_preds == position_labels).sum().item()
            
            grade_mae = torch.abs(grade_output - grade_labels).mean().item()
            grade_mae_sum += grade_mae * batch_size
            
            total_samples += batch_size
    
    return {
        'loss': total_loss / total_samples if total_samples > 0 else 0,
        'position_accuracy': position_correct / total_samples if total_samples > 0 else 0,
        'grade_mae': grade_mae_sum / total_samples if total_samples > 0 else 0
    }

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='LodgeNet优化训练脚本')
    
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
    parser.add_argument('--output_dir', type=str, default='./output_lodgenet_optimized',
                        help='输出目录')
    
    # 调试参数
    parser.add_argument('--test_mode', action='store_true',
                        help='测试模式，只运行单批次测试')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LodgeNet优化训练脚本")
    print("=" * 80)
    print(f"配置信息:")
    print(f"  数据根目录: {args.data_root}")
    print(f"  JSON根目录: {args.json_root}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  图像尺寸: {args.img_size}")
    print(f"  工作进程数: {args.num_workers}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备信息:")
    print(f"  使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  CUDA版本: {torch.version.cuda}")
    
    # 获取数据加载器
    print(f"\n数据集加载...")
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
        
        print(f"成功: 数据加载成功")
        print(f"  训练集大小: {len(train_loader.dataset)}")
        print(f"  验证集大小: {len(val_loader.dataset)}")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")
        
    except Exception as e:
        print(f"错误: 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模型
    print(f"\n模型创建...")
    try:
        model = get_lodgenet_model(
            n_channels=3,
            n_classes=2,
            img_size=args.img_size,
            bilinear=True
        ).to(device)
        
        print(f"成功: 模型创建成功")
        print(f"  模型参数数量: {count_parameters(model):,}")
        
    except Exception as e:
        print(f"错误: 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试单批次
    if not test_single_batch(model, train_loader, device):
        print("错误: 单批次测试失败，无法继续训练")
        return
    
    if args.test_mode:
        print("成功: 测试模式完成")
        return
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # 训练历史记录
    train_history = []
    val_history = []
    best_val_loss = float('inf')
    
    print(f"\n开始训练 {args.num_epochs} 轮...")
    print("=" * 80)
    
    for epoch in range(1, args.num_epochs + 1):
        # 训练
        train_metrics = train_epoch_with_monitoring(
            model, train_loader, optimizer, device, epoch, args.num_epochs
        )
        
        # 验证
        val_metrics = validate_epoch(model, val_loader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        train_history.append(train_metrics)
        val_history.append(val_metrics)
        
        # 打印验证结果
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
            print(f"已保存检查点: {args.output_dir}/checkpoint_epoch_{epoch}.pth")
            print(f"发现新的最佳模型! 已保存到: {args.output_dir}/best_model.pth")
        
        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }, os.path.join(args.output_dir, 'last_model.pth'))
        
        print("=" * 80)
    
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