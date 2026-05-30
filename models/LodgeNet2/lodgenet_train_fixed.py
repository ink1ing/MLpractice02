#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 核心文件：修复版本的LodgeNet训练脚本，使用修复后的数据加载器
# LodgeNet训练脚本：专门用于玉米锈病识别的多任务学习训练
# 实现图像分割、位置分类和病害等级回归的联合训练
# 支持混合精度训练、完整的评估指标监控和模型保存
# 优化配置以充分利用RTX6000 22.5GB GPU性能

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from torch.amp import autocast, GradScaler
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from dataset_fixed import CornRustDataset, get_dataloaders
from lodgenet_model import get_lodgenet_model, count_parameters
from utils import save_checkpoint, load_checkpoint, calculate_metrics, plot_metrics, FocalLoss, calculate_class_weights

class DiceLoss(nn.Module):
    """
    Dice损失函数：用于图像分割任务
    特别适用于处理类别不平衡问题
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # 应用softmax获取概率
        inputs = torch.softmax(inputs, dim=1)
        
        # 将targets转换为one-hot编码
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # 计算Dice系数
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss

class CombinedSegmentationLoss(nn.Module):
    """
    组合分割损失：结合CrossEntropy和Dice损失
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedSegmentationLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice

def get_data_transforms(train=True):
    """
    获取数据增强变换
    
    参数:
        train: 是否为训练模式
    
    返回:
        transforms: 数据增强变换组合
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        ])
    else:
        return None

def create_dummy_segmentation_labels(position_labels, grade_labels, img_size=128):
    """
    创建虚拟分割标签（因为当前数据集没有像素级标注）
    基于位置和等级信息生成简单的分割掩码
    
    参数:
        position_labels: 位置标签 [batch_size]
        grade_labels: 等级标签 [batch_size]
        img_size: 图像尺寸
        
    返回:
        seg_labels: 分割标签 [batch_size, img_size, img_size]
    """
    batch_size = position_labels.size(0)
    seg_labels = torch.zeros(batch_size, img_size, img_size, dtype=torch.long)
    
    for i in range(batch_size):
        # 根据等级创建不同强度的感染区域
        grade = grade_labels[i].item()
        if grade > 0:  # 有感染
            # 创建中心区域作为感染区域
            center_size = int(img_size * 0.3 * (grade / 9.0))  # 根据等级调整大小
            if center_size > 0:
                start = (img_size - center_size) // 2
                end = start + center_size
                seg_labels[i, start:end, start:end] = 1
    
    return seg_labels

def train_one_epoch(model, train_loader, optimizer, seg_criterion, position_criterion, 
                   grade_criterion, device, task_weights=[0.4, 0.3, 0.3], scaler=None):
    """
    训练模型一个epoch
    
    参数:
        model: LodgeNet模型实例
        train_loader: 训练数据加载器
        optimizer: 优化器
        seg_criterion: 分割损失函数
        position_criterion: 位置分类损失函数
        grade_criterion: 等级回归损失函数
        device: 计算设备
        task_weights: 任务权重 [分割, 位置, 等级]
        scaler: 混合精度训练的GradScaler
        
    返回:
        dict: 训练指标字典
    """
    model.train()
    total_loss = 0.0
    seg_loss_sum = 0.0
    position_loss_sum = 0.0
    grade_loss_sum = 0.0
    position_correct = 0
    total_samples = 0
    grade_mae_sum = 0.0
    
    progress_bar = tqdm(train_loader, desc="训练中")
    
    for images, position_labels, grade_labels in progress_bar:
        # 将数据移动到设备
        images = images.to(device, non_blocking=True)
        position_labels = position_labels.view(-1).long().to(device, non_blocking=True)
        grade_labels = grade_labels.float().unsqueeze(1).to(device, non_blocking=True)
        
        # 创建虚拟分割标签
        seg_labels = create_dummy_segmentation_labels(
            position_labels, grade_labels.squeeze(1), images.size(-1)
        ).to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 使用混合精度训练
        if scaler is not None:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # 前向传播
                seg_output, position_logits, grade_output = model(images)
                
                # 计算各任务损失
                loss_seg = seg_criterion(seg_output, seg_labels)
                loss_position = position_criterion(position_logits, position_labels)
                loss_grade = grade_criterion(grade_output, grade_labels)
                
                # 组合损失
                total_task_loss = (task_weights[0] * loss_seg + 
                                 task_weights[1] * loss_position + 
                                 task_weights[2] * loss_grade)
            
            # 反向传播
            scaler.scale(total_task_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规训练
            seg_output, position_logits, grade_output = model(images)
            
            loss_seg = seg_criterion(seg_output, seg_labels)
            loss_position = position_criterion(position_logits, position_labels)
            loss_grade = grade_criterion(grade_output, grade_labels)
            
            total_task_loss = (task_weights[0] * loss_seg + 
                             task_weights[1] * loss_position + 
                             task_weights[2] * loss_grade)
            
            total_task_loss.backward()
            optimizer.step()
        
        # 统计指标
        batch_size = images.size(0)
        total_loss += total_task_loss.item() * batch_size
        seg_loss_sum += loss_seg.item() * batch_size
        position_loss_sum += loss_position.item() * batch_size
        grade_loss_sum += loss_grade.item() * batch_size
        
        # 位置分类准确率
        _, position_preds = torch.max(position_logits, 1)
        position_correct += (position_preds == position_labels).sum().item()
        
        # 等级回归MAE
        grade_mae = torch.abs(grade_output - grade_labels).mean().item()
        grade_mae_sum += grade_mae * batch_size
        
        total_samples += batch_size
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': total_task_loss.item(),
            'pos_acc': position_correct / total_samples,
            'grade_mae': grade_mae_sum / total_samples
        })
    
    return {
        'loss': total_loss / total_samples,
        'seg_loss': seg_loss_sum / total_samples,
        'position_loss': position_loss_sum / total_samples,
        'grade_loss': grade_loss_sum / total_samples,
        'position_accuracy': position_correct / total_samples,
        'grade_mae': grade_mae_sum / total_samples
    }

def evaluate(model, val_loader, seg_criterion, position_criterion, grade_criterion, 
            device, task_weights=[0.4, 0.3, 0.3]):
    """
    评估模型性能
    
    参数:
        model: LodgeNet模型实例
        val_loader: 验证数据加载器
        seg_criterion: 分割损失函数
        position_criterion: 位置分类损失函数
        grade_criterion: 等级回归损失函数
        device: 计算设备
        task_weights: 任务权重
        
    返回:
        dict: 评估指标字典
    """
    model.eval()
    total_loss = 0.0
    seg_loss_sum = 0.0
    position_loss_sum = 0.0
    grade_loss_sum = 0.0
    
    position_preds_all = []
    position_labels_all = []
    grade_values_all = []
    grade_labels_all = []
    
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
            
            total_task_loss = (task_weights[0] * loss_seg + 
                             task_weights[1] * loss_position + 
                             task_weights[2] * loss_grade)
            
            batch_size = images.size(0)
            total_loss += total_task_loss.item() * batch_size
            seg_loss_sum += loss_seg.item() * batch_size
            position_loss_sum += loss_position.item() * batch_size
            grade_loss_sum += loss_grade.item() * batch_size
            
            # 收集预测结果
            _, position_preds = torch.max(position_logits, 1)
            position_preds_all.extend(position_preds.cpu().numpy())
            position_labels_all.extend(position_labels.cpu().numpy())
            grade_values_all.extend(grade_output.cpu().numpy().flatten())
            grade_labels_all.extend(grade_labels.cpu().numpy().flatten())
    
    # 计算详细指标
    total_samples = len(position_labels_all)
    
    # 位置分类指标
    position_accuracy = accuracy_score(position_labels_all, position_preds_all)
    position_f1 = f1_score(position_labels_all, position_preds_all, average='weighted')
    position_precision = precision_score(position_labels_all, position_preds_all, average='weighted')
    position_recall = recall_score(position_labels_all, position_preds_all, average='weighted')
    
    # 等级回归指标
    grade_mae = np.mean(np.abs(np.array(grade_values_all) - np.array(grade_labels_all)))
    grade_rmse = np.sqrt(np.mean((np.array(grade_values_all) - np.array(grade_labels_all))**2))
    
    return {
        'loss': total_loss / total_samples,
        'seg_loss': seg_loss_sum / total_samples,
        'position_loss': position_loss_sum / total_samples,
        'grade_loss': grade_loss_sum / total_samples,
        'position_accuracy': position_accuracy,
        'position_f1': position_f1,
        'position_precision': position_precision,
        'position_recall': position_recall,
        'grade_mae': grade_mae,
        'grade_rmse': grade_rmse
    }

def plot_training_metrics(metrics_history, save_dir):
    """
    绘制训练指标图表
    
    参数:
        metrics_history: 指标历史记录
        save_dir: 保存目录
    """
    epochs = range(1, len(metrics_history['train']) + 1)
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LodgeNet训练指标', fontsize=16)
    
    # 损失曲线
    axes[0, 0].plot(epochs, [m['loss'] for m in metrics_history['train']], 'b-', label='训练')
    axes[0, 0].plot(epochs, [m['loss'] for m in metrics_history['val']], 'r-', label='验证')
    axes[0, 0].set_title('总损失')
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 位置分类准确率
    axes[0, 1].plot(epochs, [m['position_accuracy'] for m in metrics_history['train']], 'b-', label='训练')
    axes[0, 1].plot(epochs, [m['position_accuracy'] for m in metrics_history['val']], 'r-', label='验证')
    axes[0, 1].set_title('位置分类准确率')
    axes[0, 1].set_xlabel('轮次')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 位置分类F1分数
    axes[0, 2].plot(epochs, [m.get('position_f1', 0) for m in metrics_history['val']], 'r-', label='验证F1')
    axes[0, 2].set_title('位置分类F1分数')
    axes[0, 2].set_xlabel('轮次')
    axes[0, 2].set_ylabel('F1分数')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # 等级回归MAE
    axes[1, 0].plot(epochs, [m['grade_mae'] for m in metrics_history['train']], 'b-', label='训练')
    axes[1, 0].plot(epochs, [m['grade_mae'] for m in metrics_history['val']], 'r-', label='验证')
    axes[1, 0].set_title('等级预测MAE')
    axes[1, 0].set_xlabel('轮次')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 分割损失
    axes[1, 1].plot(epochs, [m['seg_loss'] for m in metrics_history['train']], 'b-', label='训练')
    axes[1, 1].plot(epochs, [m['seg_loss'] for m in metrics_history['val']], 'r-', label='验证')
    axes[1, 1].set_title('分割损失')
    axes[1, 1].set_xlabel('轮次')
    axes[1, 1].set_ylabel('损失')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 位置分类损失
    axes[1, 2].plot(epochs, [m['position_loss'] for m in metrics_history['train']], 'b-', label='训练')
    axes[1, 2].plot(epochs, [m['position_loss'] for m in metrics_history['val']], 'r-', label='验证')
    axes[1, 2].set_title('位置分类损失')
    axes[1, 2].set_xlabel('轮次')
    axes[1, 2].set_ylabel('损失')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lodgenet_training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        title: 图表标题
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    """
    主训练函数
    """
    print("====== LodgeNet训练开始 ======")
    print(f"训练配置:")
    print(f"  - 数据根目录: {args.data_root}")
    print(f"  - JSON根目录: {args.json_root}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 学习率: {args.learning_rate}")
    print(f"  - 训练轮数: {args.num_epochs}")
    print(f"  - 图像尺寸: {args.img_size}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  - 使用设备: {device}")
    
    # 优化GPU设置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # 优化卷积性能
        torch.backends.cudnn.deterministic = False  # 允许非确定性算法以提高性能
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 获取数据加载器
    print("\n加载数据集...")
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
    
    # 创建模型
    print("\n创建LodgeNet模型...")
    model = get_lodgenet_model(
        n_channels=3,
        n_classes=2,  # 背景 + 感染区域
        img_size=args.img_size,
        bilinear=True
    ).to(device)
    
    print(f"模型参数数量: {count_parameters(model):,}")
    
    # 定义损失函数
    seg_criterion = CombinedSegmentationLoss(ce_weight=0.5, dice_weight=0.5)
    position_criterion = FocalLoss(alpha=None, gamma=2.0)
    grade_criterion = nn.MSELoss()
    
    # 定义优化器 - 使用AdamW优化器，更好的权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    # 学习率调度器 - 使用CosineAnnealingLR获得更好的收敛
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    
    # 混合精度训练
    scaler = GradScaler() if torch.cuda.is_available() else None
    if scaler:
        print("  - 启用混合精度训练")
    
    # 训练历史记录
    metrics_history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    best_f1 = 0.0
    best_mae = float('inf')
    
    print(f"\n开始训练 {args.num_epochs} 轮...")
    
    for epoch in range(args.num_epochs):
        print(f"\n轮次 {epoch+1}/{args.num_epochs}")
        print("-" * 50)
        
        # 训练一个epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, seg_criterion, 
            position_criterion, grade_criterion, device, 
            task_weights=[0.4, 0.3, 0.3], scaler=scaler
        )
        
        # 验证
        val_metrics = evaluate(
            model, val_loader, seg_criterion, position_criterion, 
            grade_criterion, device, task_weights=[0.4, 0.3, 0.3]
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录指标
        metrics_history['train'].append(train_metrics)
        metrics_history['val'].append(val_metrics)
        
        # 打印指标
        print(f"训练 - 损失: {train_metrics['loss']:.4f}, "
              f"位置准确率: {train_metrics['position_accuracy']:.4f}, "
              f"等级MAE: {train_metrics['grade_mae']:.4f}")
        print(f"验证 - 损失: {val_metrics['loss']:.4f}, "
              f"位置准确率: {val_metrics['position_accuracy']:.4f}, "
              f"位置F1: {val_metrics['position_f1']:.4f}, "
              f"等级MAE: {val_metrics['grade_mae']:.4f}")
        
        # 检查是否达到目标指标
        target_met = (val_metrics['position_accuracy'] > 0.90 and 
                     val_metrics['position_f1'] > 0.85 and
                     val_metrics['position_recall'] > 0.85 and
                     val_metrics['position_precision'] > 0.85 and
                     val_metrics['grade_mae'] < 0.15 and
                     val_metrics['loss'] < 0.2)
        
        if target_met:
            print("达到目标指标!")
        
        # 保存最佳模型
        is_best_loss = val_metrics['loss'] < best_val_loss
        is_best_f1 = val_metrics['position_f1'] > best_f1
        is_best_mae = val_metrics['grade_mae'] < best_mae
        
        if is_best_loss:
            best_val_loss = val_metrics['loss']
            
        if is_best_f1:
            best_f1 = val_metrics['position_f1']
            
        if is_best_mae:
            best_mae = val_metrics['grade_mae']
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_f1': best_f1,
            'best_mae': best_mae,
            'metrics_history': metrics_history,
            'args': vars(args)
        }
        
        # 保存最新模型
        torch.save(checkpoint, os.path.join(args.output_dir, 'last_model.pth'))
        
        # 保存最佳模型
        if is_best_f1:
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"保存最佳F1模型 (F1: {best_f1:.4f})")
        
        # 每10轮保存一次
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(args.output_dir, f'epoch_{epoch+1}.pth'))
    
    print(f"\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳F1分数: {best_f1:.4f}")
    print(f"最佳MAE: {best_mae:.4f}")
    
    # 绘制训练曲线
    print("\n生成训练指标图表...")
    plot_training_metrics(metrics_history, args.output_dir)
    
    # 保存训练历史
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_history, f, indent=2, ensure_ascii=False)
    
    print(f"训练结果已保存到: {args.output_dir}")
    print("====== LodgeNet训练完成 ======")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LodgeNet训练脚本')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='TIF图像数据根目录')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注数据根目录')
    
    # 训练参数 - 优化以充分利用RTX6000性能
    parser.add_argument('--batch_size', type=int, default=32,  # 增大批次大小
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,  # 稍微增大学习率
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=8,  # 增加工作进程数
                        help='数据加载器工作进程数')
    
    # 模型参数
    parser.add_argument('--img_size', type=int, default=256,  # 增大图像尺寸以提高精度
                        help='输入图像尺寸')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./output_lodgenet',
                        help='输出目录')
    
    args = parser.parse_args()
    main(args) 