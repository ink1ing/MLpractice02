#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCN模型训练脚本
使用混合精度训练和EarlyStopping策略，优化训练效率和性能
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
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_absolute_error
import time
import torch.nn.functional as F

# 导入自定义模块
from dataset import get_dataloaders
from model import get_model
from utils import save_checkpoint, load_checkpoint, FocalLoss, plot_metrics

class EarlyStopping:
    """早停策略实现，监控指定指标，在指标不再改善时提前结束训练"""
    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        """
        初始化早停策略
        
        参数:
            patience: 容忍多少个epoch指标没有改善
            min_delta: 指标改善的最小变化量
            mode: 'min'表示监控指标越小越好，'max'表示监控指标越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, current_score):
        """
        判断是否应该早停
        
        参数:
            current_score: 当前指标值
            
        返回:
            bool: 是否应该早停
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
            
        if self.mode == 'min':
            # 如果监控指标是越小越好（如损失）
            delta = self.best_score - current_score
            score_improved = delta > self.min_delta
        else:
            # 如果监控指标是越大越好（如准确率）
            delta = current_score - self.best_score
            score_improved = delta > self.min_delta
            
        if score_improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop

def train_fcn(args):
    """
    训练FCN模型的主函数
    
    参数:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置随机种子保证可复现性
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置CUDA优化
    if torch.cuda.is_available():
        # 确保使用最佳算法以提高性能
        if args.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            print("已启用cudnn.benchmark以优化性能")
        
        # 清空GPU缓存
        torch.cuda.empty_cache()
        
        # 输出GPU信息
        print(f"使用GPU: {torch.cuda.get_device_name(args.gpu)}")
        print(f"GPU显存总量: {torch.cuda.get_device_properties(args.gpu).total_memory / (1024**3):.2f} GB")
        print(f"当前分配显存: {torch.cuda.memory_allocated(args.gpu) / (1024**3):.2f} GB")
        print(f"当前缓存显存: {torch.cuda.memory_reserved(args.gpu) / (1024**3):.2f} GB")
    
    # 获取数据加载器
    print("\n=== DataLoader 配置信息 ===")
    print(f"批次大小: {args.batch_size}")
    print(f"加载线程数: {args.num_workers}")
    print(f"是否使用pin_memory: {args.pin_memory or torch.cuda.is_available()}")
    print(f"prefetch_factor: {args.prefetch_factor}")
    
    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        json_root=args.json_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        use_extended_dataset=True,
        pin_memory=args.pin_memory or torch.cuda.is_available(),
        prefetch_factor=args.prefetch_factor
    )
    print("=== DataLoader配置完成 ===\n")
    
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
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 学习率调度器
    if args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True
        )
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    
    # 定义混合精度训练的Grad Scaler
    scaler = GradScaler() if args.amp or torch.cuda.is_available() else None
    if scaler:
        print("已启用混合精度训练")
    
    # 定义早停策略
    if args.monitor_metric == 'f1':
        early_stopping = EarlyStopping(patience=args.patience, mode='max')
    else:  # mae
        early_stopping = EarlyStopping(patience=args.patience, mode='min')
    
    # 训练历史记录
    history = {
        'train_loss': [], 'val_loss': [],
        'train_position_acc': [], 'val_position_acc': [],
        'train_grade_mae': [], 'val_grade_mae': [],
        'val_f1': [], 'val_precision': [], 'val_recall': []
    }
    
    # 最佳模型指标和路径
    best_metric = float('inf') if args.monitor_metric == 'mae' else 0
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    
    # 训练循环
    print(f"开始训练，总共 {args.epochs} 轮...")
    for epoch in range(args.epochs):
        # 记录轮次开始时间
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        position_correct = 0
        total_samples = 0
        grade_mae_sum = 0.0
        task_weights = [args.pos_weight, 1.0 - args.pos_weight]  # 位置分类和等级回归的任务权重
        
        train_pbar = tqdm(train_loader, desc=f"训练中")
        
        # 计时器，用于计算数据加载和计算时间
        data_time = 0.0
        compute_time = 0.0
        batch_times = []
        data_times = []
        compute_times = []
        
        # 梯度累积初始化
        optimizer.zero_grad()  # 开始时清零梯度
        accumulated_step = 0
        
        end = time.time()
        for images, position_labels, grade_labels in train_pbar:
            # 记录数据加载时间
            data_time = time.time() - end
            data_times.append(data_time)
            
            # 将数据移至设备
            images = images.to(device, non_blocking=True)  # 使用non_blocking=True加速
            position_labels = position_labels.to(device, non_blocking=True).long()
            grade_labels = grade_labels.float().unsqueeze(1).to(device, non_blocking=True)
            
            # 标记计算开始时间
            compute_start = time.time()
            
            # 使用混合精度训练
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=scaler is not None):
                # 前向传播
                position_logits, grade_values = model(images)
                
                # 对FCN输出进行池化处理，将[N, C, H, W]转换为[N, C]进行分类
                if len(position_logits.shape) == 4:  # 如果是FCN输出
                    # 全局平均池化，获得类别得分
                    position_scores = F.adaptive_avg_pool2d(position_logits, 1).view(position_logits.size(0), -1)
                else:
                    position_scores = position_logits
                
                # 计算损失
                pos_loss = position_criterion(position_logits, position_labels)
                grade_loss = grade_criterion(grade_values, grade_labels)
                loss = task_weights[0] * pos_loss + task_weights[1] * grade_loss
                
                # 如果使用梯度累积，对损失进行归一化
                if args.grad_accumulation > 1:
                    loss = loss / args.grad_accumulation
            
            # 反向传播和优化
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            accumulated_step += 1
            if accumulated_step == args.grad_accumulation:
                # 更新参数
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # 清零梯度并重置累积计数器
                optimizer.zero_grad()
                accumulated_step = 0
            
            # 记录计算时间
            compute_time = time.time() - compute_start
            compute_times.append(compute_time)
            
            # 计算训练指标
            batch_size = images.size(0)
            train_loss += loss.item() * batch_size * (args.grad_accumulation if args.grad_accumulation > 1 else 1)
            
            # 从池化后的分数中获取预测类别
            _, position_preds = torch.max(position_scores, 1)
            position_correct += (position_preds == position_labels).sum().item()
            
            grade_mae = torch.abs(grade_values - grade_labels).mean().item()
            grade_mae_sum += grade_mae * batch_size
            
            total_samples += batch_size
            
            # 计算总批次时间
            batch_time = time.time() - end
            batch_times.append(batch_time)
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': f"{loss.item() * (args.grad_accumulation if args.grad_accumulation > 1 else 1):.3f}",
                'pos_acc': f"{position_correct / total_samples:.3f}",
                'data_time': f"{data_time:.3f}s",
                'compute_time': f"{compute_time:.3f}s"
            })
            
            # 准备下一批次的计时
            end = time.time()
            
            # 每10个批次显示一次性能分析
            if len(batch_times) % 10 == 0:
                recent_batch_times = batch_times[-10:]
                recent_data_times = data_times[-10:]
                recent_compute_times = compute_times[-10:]
                
                avg_batch_time = sum(recent_batch_times) / len(recent_batch_times)
                avg_data_time = sum(recent_data_times) / len(recent_data_times)
                avg_compute_time = sum(recent_compute_times) / len(recent_compute_times)
                
                samples_per_sec = batch_size / avg_batch_time
                
                print("\n性能分析 (最近10个批次):")
                print(f"  平均数据加载时间: {avg_data_time:.3f}秒")
                print(f"  平均计算时间: {avg_compute_time:.3f}秒")
                print(f"  平均总批次时间: {avg_batch_time:.3f}秒")
                print(f"  当前性能: {samples_per_sec:.1f} 样本/秒")
        
        # 确保最后一批次的梯度被应用（如果使用梯度累积）
        if accumulated_step > 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # 在整个epoch结束后显示性能统计
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
        
        # 计算训练平均指标
        train_loss = train_loss / total_samples
        train_position_acc = position_correct / total_samples
        train_grade_mae = grade_mae_sum / total_samples
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_position_correct = 0
        val_total_samples = 0
        val_grade_mae_sum = 0.0
        
        # 收集所有预测和真实标签用于计算详细指标
        all_position_preds = []
        all_position_labels = []
        all_grade_values = []
        all_grade_labels = []
        
        val_pbar = tqdm(val_loader, desc=f"验证中")
        with torch.no_grad():
            for images, position_labels, grade_labels in val_pbar:
                # 将数据移至设备
                images = images.to(device, non_blocking=True)
                position_labels = position_labels.to(device, non_blocking=True).long()
                grade_labels = grade_labels.float().unsqueeze(1).to(device, non_blocking=True)
                
                # 使用混合精度，但不计算梯度
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=scaler is not None):
                    # 前向传播
                    position_logits, grade_values = model(images)
                    
                    # 对FCN输出进行池化处理，将[N, C, H, W]转换为[N, C]进行分类
                    if len(position_logits.shape) == 4:  # 如果是FCN输出
                        # 全局平均池化，获得类别得分
                        position_scores = F.adaptive_avg_pool2d(position_logits, 1).view(position_logits.size(0), -1)
                    else:
                        position_scores = position_logits
                    
                    # 计算损失
                    pos_loss = position_criterion(position_logits, position_labels)
                    grade_loss = grade_criterion(grade_values, grade_labels)
                    loss = task_weights[0] * pos_loss + task_weights[1] * grade_loss
                
                # 计算验证指标
                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                
                # 从池化后的分数中获取预测类别
                _, position_preds = torch.max(position_scores, 1)
                val_position_correct += (position_preds == position_labels).sum().item()
                
                val_grade_mae = torch.abs(grade_values - grade_labels).mean().item()
                val_grade_mae_sum += val_grade_mae * batch_size
                
                val_total_samples += batch_size
                
                # 收集预测和标签
                all_position_preds.extend(position_preds.cpu().numpy())
                all_position_labels.extend(position_labels.cpu().numpy())
                all_grade_values.extend(grade_values.cpu().numpy())
                all_grade_labels.extend(grade_labels.cpu().numpy())
                
                # 更新进度条
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.3f}",
                    'pos_acc': f"{val_position_correct / val_total_samples:.3f}"
                })
        
        # 计算验证平均指标
        val_loss = val_loss / val_total_samples
        val_position_acc = val_position_correct / val_total_samples
        val_grade_mae = val_grade_mae_sum / val_total_samples
        
        # 计算详细指标
        val_f1 = f1_score(all_position_labels, all_position_preds, average='macro')
        val_precision = precision_score(all_position_labels, all_position_preds, average='macro')
        val_recall = recall_score(all_position_labels, all_position_preds, average='macro')
        
        # 更新历史记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_position_acc'].append(train_position_acc)
        history['val_position_acc'].append(val_position_acc)
        history['train_grade_mae'].append(train_grade_mae)
        history['val_grade_mae'].append(val_grade_mae)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        
        # 更新学习率
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # 打印当前epoch的指标
        print(f"\n轮次 {epoch+1} 完成，耗时: {time.time() - epoch_start_time:.2f}秒")
        print(f"训练指标: 损失={train_loss:.4f}, 位置准确率={train_position_acc:.4f}, 等级MAE={train_grade_mae:.4f}")
        print(f"验证指标: 损失={val_loss:.4f}, 位置准确率={val_position_acc:.4f}, F1={val_f1:.4f}, 等级MAE={val_grade_mae:.4f}")
        print(f"精确率: {val_precision:.4f}, 召回率: {val_recall:.4f}")
        
        # 显示各类别详细指标
        position_f1_per_class = f1_score(all_position_labels, all_position_preds, average=None)
        position_precision_per_class = precision_score(all_position_labels, all_position_preds, average=None)
        position_recall_per_class = recall_score(all_position_labels, all_position_preds, average=None)
        
        print(f"各类别精确率: {', '.join([f'{p:.4f}' for p in position_precision_per_class])}")
        print(f"各类别召回率: {', '.join([f'{r:.4f}' for r in position_recall_per_class])}")
        print(f"各类别F1: {', '.join([f'{f1:.4f}' for f1 in position_f1_per_class])}")
        
        # 保存检查点 - 使用异常处理防止保存故障中断训练
        try:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            # 创建完整检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler': scaler.state_dict() if scaler else None,
                'loss': val_loss,
                'f1': val_f1,
                'mae': val_grade_mae,
                'args': vars(args)
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"已保存检查点: {checkpoint_path}")
        except Exception as e:
            print(f"保存检查点时出错: {e}")
            # 尝试仅保存模型权重
            try:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"weights_epoch_{epoch+1}.pth"))
                print(f"已保存模型权重")
            except Exception as e2:
                print(f"保存模型权重也失败: {e2}")
        
        # 检查是否为最佳模型
        monitor_value = val_grade_mae if args.monitor_metric == 'mae' else val_f1
        is_best = (monitor_value < best_metric) if args.monitor_metric == 'mae' else (monitor_value > best_metric)
        
        if is_best:
            best_metric = monitor_value
            try:
                # 保存最佳模型
                best_model_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"发现新的最佳模型! 已保存到: {best_model_path}")
            except Exception as e:
                print(f"保存最佳模型时出错: {e}")
        
        # 检查早停条件
        if early_stopping(monitor_value):
            print(f"Early stopping 触发，{args.patience} 轮内 {args.monitor_metric} 没有改善")
            break
    
    # 保存最后一个模型
    try:
        last_model_path = os.path.join(args.output_dir, 'last_model.pth')
        torch.save(model.state_dict(), last_model_path)
    except Exception as e:
        print(f"保存最终模型时出错: {e}")
    
    # 绘制训练历史
    try:
        plot_metrics(history, args.output_dir)
    except Exception as e:
        print(f"绘制训练历史图时出错: {e}")
    
    # 返回最终指标
    return {
        'final_loss': history['val_loss'][-1] if history['val_loss'] else float('nan'),
        'final_accuracy': history['val_position_acc'][-1] if history['val_position_acc'] else float('nan'),
        'final_f1': history['val_f1'][-1] if history['val_f1'] else float('nan'),
        'final_precision': history['val_precision'][-1] if history['val_precision'] else float('nan'),
        'final_recall': history['val_recall'][-1] if history['val_recall'] else float('nan'),
        'final_mae': history['val_grade_mae'][-1] if history['val_grade_mae'] else float('nan'),
        'best_metric': best_metric,
        'best_metric_name': args.monitor_metric
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FCN模型训练脚本')
    parser.add_argument('--data_root', type=str, required=True, help='多光谱图像数据目录')
    parser.add_argument('--json_root', type=str, default=None, help='标注JSON文件目录')
    parser.add_argument('--output_dir', type=str, default='./output_fcn', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--img_size', type=int, default=128, help='图像大小')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU索引')
    parser.add_argument('--pos_weight', type=float, default=0.7, help='位置任务权重，等级任务权重为1-pos_weight')
    parser.add_argument('--patience', type=int, default=5, help='早停策略的耐心轮数')
    parser.add_argument('--monitor_metric', type=str, default='f1', choices=['f1', 'mae'], help='监控指标，f1或mae')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='数据加载器的预取因子')
    parser.add_argument('--pin_memory', action='store_true', help='使用pin_memory提高数据传输速度')
    parser.add_argument('--grad_accumulation', type=int, default=1, help='梯度累积步数，可以使用更大的批次大小')
    parser.add_argument('--cudnn_benchmark', action='store_true', help='启用cudnn.benchmark以优化性能')
    parser.add_argument('--amp', action='store_true', help='启用自动混合精度训练')
    parser.add_argument('--lr_scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'step'], help='学习率调度器')
    
    args = parser.parse_args()
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型并获取指标
    metrics = train_fcn(args)
    
    # 计算总训练时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 输出最终结果
    print("\n" + "="*50)
    print("训练完成！")
    print(f"总训练时间: {int(hours)}时 {int(minutes)}分 {seconds:.2f}秒")
    print(f"最终验证损失: {metrics['final_loss']:.4f}")
    print(f"位置分类准确率: {metrics['final_accuracy']:.4f}")
    print(f"位置分类F1分数: {metrics['final_f1']:.4f}")
    print(f"位置分类精确率: {metrics['final_precision']:.4f}")
    print(f"位置分类召回率: {metrics['final_recall']:.4f}")
    print(f"等级预测MAE: {metrics['final_mae']:.4f}")
    print(f"最佳{metrics['best_metric_name']}: {metrics['best_metric']:.4f}")
    print("="*50)
    
    print(f"\n模型保存在: {args.output_dir}")
    print(f"最佳模型: {os.path.join(args.output_dir, 'best_model.pth')}")
    print(f"最终模型: {os.path.join(args.output_dir, 'last_model.pth')}") 