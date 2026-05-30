#!/usr/bin/env python
# 核心文件，完整运行项目不可缺少
# 优化版训练脚本：使用所有优化后的代码，解决训练卡住的问题

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import sys
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from torch.amp import autocast, GradScaler
import random
import json
from datetime import datetime

# 导入自定义模块
from dataset import CornRustDataset, get_dataloaders
from model import get_model
from utils import save_checkpoint, load_checkpoint, calculate_metrics, plot_metrics, FocalLoss, calculate_class_weights

def set_seed(seed=42):
    """
    设置随机种子以确保实验可重复性
    
    参数:
        seed: 随机种子，默认为42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='玉米南方锈病识别系统训练脚本')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit', help='数据集根目录，包含.tif文件')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json', help='标注数据根目录，包含.json文件')
    parser.add_argument('--img_size', type=int, default=128, help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='aud', choices=['simple', 'resnet', 'resnet+', 'aud'], 
                        help='模型类型: simple, resnet, resnet+, aud')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--pretrained', type=str, default=None, help='预训练模型路径')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--task_weights', type=float, nargs=2, default=[0.7, 0.3], 
                        help='任务权重 [位置权重, 等级权重]')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./output_aud', help='输出目录')
    parser.add_argument('--log_interval', type=int, default=10, help='日志输出间隔（批次）')
    parser.add_argument('--save_interval', type=int, default=5, help='模型保存间隔（轮次）')
    
    # 特殊功能
    parser.add_argument('--amp', action='store_true', default=True, help='启用混合精度训练')
    parser.add_argument('--focal_loss', action='store_true', default=True, help='使用Focal Loss替代交叉熵损失')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal Loss的gamma参数')
    
    return parser.parse_args()

def train_one_epoch(model, train_loader, optimizer, position_criterion, grade_criterion, device, 
                   task_weights, scaler=None, log_interval=10):
    """
    训练模型一个epoch
    
    参数:
        model: 模型实例
        train_loader: 训练数据加载器
        optimizer: 优化器实例
        position_criterion: 位置分类的损失函数
        grade_criterion: 等级预测的损失函数
        device: 计算设备(CPU/GPU)
        task_weights: 任务权重，表示位置任务和等级任务的权重
        scaler: 混合精度训练的GradScaler实例
        log_interval: 日志输出间隔（批次）
        
    返回:
        dict: 包含训练指标的字典
    """
    model.train()
    total_loss = 0.0
    position_loss_sum = 0.0
    grade_loss_sum = 0.0
    position_correct = 0
    total_samples = 0
    grade_mae_sum = 0.0
    
    # 收集预测和标签用于计算F1分数
    position_preds_all = []
    position_labels_all = []
    
    progress_bar = tqdm(train_loader, desc="训练中")
    
    for batch_idx, (images, position_labels, grade_labels) in enumerate(progress_bar):
        # 将数据移动到指定设备
        images = images.to(device)
        position_labels = position_labels.to(device).long()
        grade_labels = grade_labels.float().unsqueeze(1).to(device)
        
        # 清零梯度
        optimizer.zero_grad()  
        
        # 使用混合精度训练
        if scaler is not None:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # 前向传播
                position_logits, grade_values = model(images)
                
                # 计算损失
                loss_position = position_criterion(position_logits, position_labels)
                loss_grade = grade_criterion(grade_values, grade_labels)
                
                # 组合损失
                loss = task_weights[0] * loss_position + task_weights[1] * loss_grade
            
            # 反向传播和参数更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规训练流程
            position_logits, grade_values = model(images)
            loss_position = position_criterion(position_logits, position_labels)
            loss_grade = grade_criterion(grade_values, grade_labels)
            loss = task_weights[0] * loss_position + task_weights[1] * loss_grade
            loss.backward()
            optimizer.step()
        
        # 统计指标
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        position_loss_sum += loss_position.item() * batch_size
        grade_loss_sum += loss_grade.item() * batch_size
        
        # 计算位置分类准确率
        _, position_preds = torch.max(position_logits, 1)
        position_correct += (position_preds == position_labels).sum().item()
        
        # 收集用于计算F1分数的数据
        position_preds_all.extend(position_preds.cpu().numpy())
        position_labels_all.extend(position_labels.cpu().numpy())
        
        # 计算等级预测MAE
        grade_mae = torch.abs(grade_values - grade_labels).mean().item()
        grade_mae_sum += grade_mae * batch_size
        
        total_samples += batch_size
        
        # 更新进度条
        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
            # 计算F1分数
            position_f1 = f1_score(position_labels_all, position_preds_all, average='macro')
            position_recall = recall_score(position_labels_all, position_preds_all, average='macro')
            position_precision = precision_score(position_labels_all, position_preds_all, average='macro')
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'pos_acc': position_correct / total_samples,
                'pos_f1': position_f1,
                'grade_mae': grade_mae_sum / total_samples
            })
    
    # 计算整体指标
    avg_loss = total_loss / total_samples
    avg_position_loss = position_loss_sum / total_samples
    avg_grade_loss = grade_loss_sum / total_samples
    position_accuracy = position_correct / total_samples
    grade_mae = grade_mae_sum / total_samples
    
    # 计算F1分数
    position_f1 = f1_score(position_labels_all, position_preds_all, average='macro')
    position_recall = recall_score(position_labels_all, position_preds_all, average='macro')
    position_precision = precision_score(position_labels_all, position_preds_all, average='macro')
    
    return {
        'loss': avg_loss,
        'position_loss': avg_position_loss,
        'grade_loss': avg_grade_loss,
        'position_accuracy': position_accuracy,
        'position_f1': position_f1,
        'position_recall': position_recall,
        'position_precision': position_precision,
        'grade_mae': grade_mae
    }

def evaluate(model, val_loader, position_criterion, grade_criterion, device, task_weights):
    """
    评估模型在验证集上的性能
    
    参数:
        model: 模型实例
        val_loader: 验证数据加载器
        position_criterion: 位置分类的损失函数
        grade_criterion: 等级预测的损失函数
        device: 计算设备(CPU/GPU)
        task_weights: 任务权重
        
    返回:
        dict: 包含详细评估指标的字典
    """
    model.eval()
    total_loss = 0.0
    position_loss_sum = 0.0
    grade_loss_sum = 0.0
    
    # 收集预测和标签
    position_preds_all = []
    position_labels_all = []
    grade_values_all = []
    grade_labels_all = []
    
    with torch.no_grad():
        for images, position_labels, grade_labels in val_loader:
            # 将数据移动到指定设备
            images = images.to(device)
            position_labels = position_labels.to(device).long()
            grade_labels = grade_labels.float().unsqueeze(1).to(device)
            
            # 使用混合精度计算
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                # 前向传播
                position_logits, grade_values = model(images)
                
                # 计算损失
                loss_position = position_criterion(position_logits, position_labels)
                loss_grade = grade_criterion(grade_values, grade_labels)
                
                # 组合损失
                loss = task_weights[0] * loss_position + task_weights[1] * loss_grade
            
            # 统计指标
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            position_loss_sum += loss_position.item() * batch_size
            grade_loss_sum += loss_grade.item() * batch_size
            
            # 获取位置预测类别
            _, position_preds = torch.max(position_logits, 1)
            
            # 收集预测和标签
            position_preds_all.extend(position_preds.cpu().numpy())
            position_labels_all.extend(position_labels.cpu().numpy())
            grade_values_all.extend(grade_values.cpu().numpy())
            grade_labels_all.extend(grade_labels.cpu().numpy())
    
    # 计算平均指标
    total_samples = len(val_loader.dataset)
    avg_loss = total_loss / total_samples
    avg_position_loss = position_loss_sum / total_samples
    avg_grade_loss = grade_loss_sum / total_samples
    
    # 计算位置分类详细指标
    position_accuracy = accuracy_score(position_labels_all, position_preds_all)
    position_f1 = f1_score(position_labels_all, position_preds_all, average='macro')
    position_f1_per_class = f1_score(position_labels_all, position_preds_all, average=None)
    position_cm = confusion_matrix(position_labels_all, position_preds_all)
    position_precision = precision_score(position_labels_all, position_preds_all, average='macro')
    position_recall = recall_score(position_labels_all, position_preds_all, average='macro')
    
    # 计算等级预测MAE
    grade_values_all = np.array(grade_values_all).flatten()
    grade_labels_all = np.array(grade_labels_all).flatten()
    grade_mae = np.mean(np.abs(grade_values_all - grade_labels_all))
    
    # 计算±2误差容忍率
    grade_tolerance_2 = np.mean(np.abs(grade_values_all - grade_labels_all) <= 2.0)
    
    # 构建包含所有指标的字典
    metrics = {
        'loss': avg_loss,
        'position_loss': avg_position_loss,
        'grade_loss': avg_grade_loss,
        'position_accuracy': position_accuracy,
        'position_f1': position_f1,
        'position_precision': position_precision,
        'position_recall': position_recall,
        'position_f1_per_class': position_f1_per_class.tolist(),
        'position_confusion_matrix': position_cm.tolist(),
        'grade_mae': grade_mae,
        'grade_tolerance_2': grade_tolerance_2
    }
    
    return metrics

def save_training_log(metrics_history, args, output_dir):
    """
    保存训练日志和最终指标
    
    参数:
        metrics_history: 训练历史指标
        args: 命令行参数
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存最终指标
    final_metrics = {
        'train': metrics_history['train'][-1],
        'val': metrics_history['val'][-1],
        'training_args': vars(args),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    # 保存完整训练历史
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(metrics_history, f, indent=4)
    
    # 绘制并保存指标图表
    plot_metrics(metrics_history, output_dir)

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        json_root=args.json_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        use_extended_dataset=True,
        pin_memory=True
    )
    
    print(f"训练集样本数: {len(train_loader.dataset)}, 验证集样本数: {len(val_loader.dataset)}")
    
    # 获取模型实例
    model = get_model(model_type=args.model_type, in_channels=3, img_size=args.img_size)
    model.to(device)
    
    # 打印模型结构和参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型类型: {args.model_type}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 设置学习率调度器 - 使用ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 设置损失函数
    if args.focal_loss:
        # 计算类别权重
        dataset = train_loader.dataset
        # 获取原始数据集(非Subset)对象
        if hasattr(dataset, 'dataset'):
            # 如果是Subset，获取原始数据集
            original_dataset = dataset.dataset
            position_weights, _ = original_dataset.get_class_weights()
        else:
            # 如果不是Subset，直接使用
            position_weights, _ = dataset.get_class_weights()
            
        position_weights = torch.tensor(position_weights, device=device)
        position_criterion = FocalLoss(alpha=position_weights, gamma=args.gamma)
        print(f"使用Focal Loss (gamma={args.gamma}) 位置类别权重: {position_weights}")
    else:
        position_criterion = nn.CrossEntropyLoss()
        print("使用标准交叉熵损失")
    
    # 等级回归使用MSE损失
    grade_criterion = nn.MSELoss()
    
    # 设置混合精度训练
    scaler = GradScaler() if args.amp and torch.cuda.is_available() else None
    if scaler is not None:
        print("启用混合精度训练 (AMP)")
    
    # 如果指定了预训练模型，加载它
    start_epoch = 0
    if args.pretrained and os.path.exists(args.pretrained):
        start_epoch = load_checkpoint(model, optimizer, args.pretrained)
        print(f"从预训练模型加载权重: {args.pretrained}")
    
    # 如果需要恢复训练，加载最新的检查点
    if args.resume:
        latest_checkpoint = os.path.join(args.output_dir, 'last_model.pth')
        if os.path.exists(latest_checkpoint):
            start_epoch = load_checkpoint(model, optimizer, latest_checkpoint)
            print(f"恢复训练，从轮次 {start_epoch} 开始")
    
    # 记录训练历史指标
    metrics_history = {'train': [], 'val': []}
    best_metrics = {'f1': 0.0, 'mae': float('inf')}
    
    # 训练循环
    print(f"开始训练，共 {args.epochs} 轮...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n轮次 {epoch+1}/{args.epochs}")
        
        # 训练一个epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, position_criterion, grade_criterion,
            device, args.task_weights, scaler, args.log_interval
        )
        
        # 评估模型
        val_metrics = evaluate(
            model, val_loader, position_criterion, grade_criterion,
            device, args.task_weights
        )
        
        # 更新学习率
        scheduler.step(val_metrics['loss'])
        
        # 保存指标
        metrics_history['train'].append(train_metrics)
        metrics_history['val'].append(val_metrics)
        
        # 打印当前性能
        print(f"训练集 - 损失: {train_metrics['loss']:.4f}, 位置准确率: {train_metrics['position_accuracy']:.4f}, " 
              f"F1: {train_metrics['position_f1']:.4f}, 等级MAE: {train_metrics['grade_mae']:.4f}")
        print(f"验证集 - 损失: {val_metrics['loss']:.4f}, 位置准确率: {val_metrics['position_accuracy']:.4f}, " 
              f"F1: {val_metrics['position_f1']:.4f}, 等级MAE: {val_metrics['grade_mae']:.4f}")
        
        # 保存最后一个模型
        save_checkpoint(
            model, optimizer, epoch + 1,
            os.path.join(args.output_dir, 'last_model.pth')
        )
        
        # 根据指标保存最佳模型
        if val_metrics['position_f1'] > best_metrics['f1']:
            best_metrics['f1'] = val_metrics['position_f1']
            save_checkpoint(
                model, optimizer, epoch + 1,
                os.path.join(args.output_dir, 'best_model_f1.pth')
            )
            print(f"保存最佳F1模型 (F1={best_metrics['f1']:.4f})")
        
        if val_metrics['grade_mae'] < best_metrics['mae']:
            best_metrics['mae'] = val_metrics['grade_mae']
            save_checkpoint(
                model, optimizer, epoch + 1,
                os.path.join(args.output_dir, 'best_model_mae.pth')
            )
            print(f"保存最佳MAE模型 (MAE={best_metrics['mae']:.4f})")
        
        # 按指定间隔保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch + 1,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    # 保存训练日志和指标
    save_training_log(metrics_history, args, args.output_dir)
    
    print(f"训练完成! 最终结果:")
    print(f"最佳F1: {best_metrics['f1']:.4f}")
    print(f"最佳MAE: {best_metrics['mae']:.4f}")
    print(f"所有模型和指标已保存到 {args.output_dir}")

if __name__ == "__main__":
    main() 