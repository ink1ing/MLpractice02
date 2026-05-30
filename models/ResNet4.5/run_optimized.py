#!/usr/bin/env python
# 优化版训练脚本：使用所有优化后的代码，解决训练卡住的问题

import os
import argparse
import torch
import time
import sys
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# 导入自定义模块
from dataset import get_dataloaders
from model import get_model
from utils import FocalLoss

def train_epoch_optimized(model, train_loader, optimizer, position_criterion, grade_criterion, 
                          device, task_weights=[0.7, 0.3], scaler=None):
    """优化版训练循环，增强性能监控和错误恢复"""
    model.train()
    
    total_loss = 0.0
    position_loss_sum = 0.0
    grade_loss_sum = 0.0
    position_correct = 0
    total_samples = 0
    grade_mae_sum = 0.0
    
    batch_times = []  # 记录每个批次的处理时间
    data_times = []   # 记录数据加载时间
    compute_times = [] # 记录计算时间
    
    print(f"开始训练循环，批次大小: {train_loader.batch_size}, 批次数: {len(train_loader)}")
    progress_bar = tqdm(train_loader, desc="训练中")
    
    data_start = time.time()  # 开始计时
    
    for batch_idx, (images, position_labels, grade_labels) in enumerate(progress_bar):
        batch_start = time.time()
        data_time = batch_start - data_start  # 数据加载时间
        data_times.append(data_time)
        
        try:
            # 将数据移动到设备
            images = images.to(device, non_blocking=True)  # 使用non_blocking=True加速
            position_labels = position_labels.to(device, non_blocking=True)
            position_labels = position_labels.view(-1).long()
            grade_labels = grade_labels.float().unsqueeze(1).to(device, non_blocking=True)
            
            # 清零梯度
            optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True可以稍微加速
            
            # 使用混合精度训练
            if scaler is not None:
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    # 前向传播
                    position_logits, grade_values = model(images)
                    loss_position = position_criterion(position_logits, position_labels)
                    loss_grade = grade_criterion(grade_values, grade_labels)
                    loss = task_weights[0] * loss_position + task_weights[1] * loss_grade
                
                # 使用scaler进行反向传播和参数更新
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 前向传播
                position_logits, grade_values = model(images)
                loss_position = position_criterion(position_logits, position_labels)
                loss_grade = grade_criterion(grade_values, grade_labels)
                loss = task_weights[0] * loss_position + task_weights[1] * loss_grade
                
                # 反向传播和优化
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
            
            # 计算等级预测MAE
            grade_mae = torch.abs(grade_values - grade_labels).mean().item()
            grade_mae_sum += grade_mae * batch_size
            
            total_samples += batch_size
            
            # 记录计算时间
            compute_time = time.time() - batch_start
            compute_times.append(compute_time)
            batch_times.append(data_time + compute_time)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'pos_acc': position_correct / total_samples,
                'data_time': f"{data_time:.3f}s",
                'compute_time': f"{compute_time:.3f}s"
            })
            
            # 每10个批次输出性能分析
            if (batch_idx + 1) % 10 == 0:
                recent_batches = min(10, len(batch_times))
                print(f"\n性能分析 (最近{recent_batches}个批次):")
                print(f"  平均数据加载时间: {sum(data_times[-recent_batches:]) / recent_batches:.3f}秒")
                print(f"  平均计算时间: {sum(compute_times[-recent_batches:]) / recent_batches:.3f}秒")
                print(f"  平均总批次时间: {sum(batch_times[-recent_batches:]) / recent_batches:.3f}秒")
                print(f"  当前性能: {recent_batches * batch_size / sum(batch_times[-recent_batches:]):.1f} 样本/秒")
                
        except Exception as e:
            print(f"批次 {batch_idx} 处理出错: {e}")
            print("继续处理下一批次...")
            
        # 为下一批次重置计时器
        data_start = time.time()
    
    # 计算整个epoch的平均指标
    avg_loss = total_loss / total_samples
    avg_position_loss = position_loss_sum / total_samples
    avg_grade_loss = grade_loss_sum / total_samples
    position_accuracy = position_correct / total_samples
    grade_mae = grade_mae_sum / total_samples
    
    # 输出性能统计
    if batch_times:
        print("\n训练性能统计:")
        print(f"  平均批次时间: {sum(batch_times) / len(batch_times):.3f}秒")
        print(f"  - 数据加载: {sum(data_times) / len(data_times):.3f}秒 ({sum(data_times) / sum(batch_times) * 100:.1f}%)")
        print(f"  - 计算: {sum(compute_times) / len(compute_times):.3f}秒 ({sum(compute_times) / sum(batch_times) * 100:.1f}%)")
        print(f"  样本吞吐量: {total_samples / sum(batch_times):.1f} 样本/秒")
    
    return {
        'loss': avg_loss,
        'position_loss': avg_position_loss,
        'grade_loss': avg_grade_loss,
        'position_accuracy': position_accuracy,
        'grade_mae': grade_mae
    }

def evaluate_model(model, val_loader, position_criterion, grade_criterion, device, task_weights=[0.5, 0.5]):
    """评估模型性能并计算详细指标，包括精确率和召回率"""
    model.eval()
    
    total_loss = 0.0
    position_loss_sum = 0.0
    grade_loss_sum = 0.0
    position_correct = 0
    total_samples = 0
    grade_mae_sum = 0.0
    
    all_position_preds = []
    all_position_labels = []
    all_grade_preds = []
    all_grade_labels = []
    
    with torch.no_grad():
        for images, position_labels, grade_labels in val_loader:
            # 将数据移动到设备
            images = images.to(device, non_blocking=True)
            position_labels = position_labels.to(device, non_blocking=True).view(-1).long()
            grade_labels = grade_labels.float().unsqueeze(1).to(device, non_blocking=True)
            
            # 前向传播
            position_logits, grade_values = model(images)
            
            # 计算损失
            loss_position = position_criterion(position_logits, position_labels)
            loss_grade = grade_criterion(grade_values, grade_labels)
            loss = task_weights[0] * loss_position + task_weights[1] * loss_grade
            
            # 统计指标
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            position_loss_sum += loss_position.item() * batch_size
            grade_loss_sum += loss_grade.item() * batch_size
            
            # 计算位置分类准确率
            _, position_preds = torch.max(position_logits, 1)
            position_correct += (position_preds == position_labels).sum().item()
            
            # 计算等级预测MAE
            grade_mae = torch.abs(grade_values - grade_labels).mean().item()
            grade_mae_sum += grade_mae * batch_size
            
            # 收集预测结果用于计算F1分数、精确率和召回率
            all_position_preds.extend(position_preds.cpu().numpy())
            all_position_labels.extend(position_labels.cpu().numpy())
            all_grade_preds.extend(grade_values.cpu().numpy())
            all_grade_labels.extend(grade_labels.cpu().numpy())
            
            total_samples += batch_size
    
    # 计算整体指标
    avg_loss = total_loss / total_samples
    avg_position_loss = position_loss_sum / total_samples
    avg_grade_loss = grade_loss_sum / total_samples
    position_accuracy = position_correct / total_samples
    grade_mae = grade_mae_sum / total_samples
    
    # 计算F1分数、精确率和召回率 (宏平均和每类别)
    position_f1 = f1_score(all_position_labels, all_position_preds, average='macro')
    position_precision = precision_score(all_position_labels, all_position_preds, average='macro')
    position_recall = recall_score(all_position_labels, all_position_preds, average='macro')
    
    # 计算每个类别的精确率和召回率
    position_precision_per_class = precision_score(all_position_labels, all_position_preds, average=None)
    position_recall_per_class = recall_score(all_position_labels, all_position_preds, average=None)
    position_f1_per_class = f1_score(all_position_labels, all_position_preds, average=None)
    
    # 返回详细指标
    return {
        'loss': avg_loss,
        'position_loss': avg_position_loss,
        'grade_loss': avg_grade_loss,
        'position_accuracy': position_accuracy,
        'position_f1': position_f1,
        'position_precision': position_precision,
        'position_recall': position_recall,
        'position_precision_per_class': position_precision_per_class,
        'position_recall_per_class': position_recall_per_class,
        'position_f1_per_class': position_f1_per_class,
        'grade_mae': grade_mae
    }

def main():
    parser = argparse.ArgumentParser(description='玉米南方锈病模型优化训练脚本')
    
    # 基本参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='数据根目录路径')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注根目录路径')
    parser.add_argument('--output_dir', type=str, default='./output_optimized',
                        help='输出目录')
    parser.add_argument('--model_type', type=str, default='simple',
                        choices=['simple', 'resnet', 'resnet_plus'],
                        help='模型类型，simple速度最快，适合调试')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小，根据显存调整')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载线程数，0表示使用主线程')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA')
    parser.add_argument('--img_size', type=int, default=128,
                        help='图像大小')
    parser.add_argument('--no_extended', action='store_true',
                        help='不使用扩展数据集')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点文件路径，用于继续训练')
    
    args = parser.parse_args()
    
    # 打印系统信息
    print(f"\n=== 系统信息 ===")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 提前清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据加载器
    try:
        # 使用固定随机种子确保可重复性
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        # 获取数据加载器
        train_loader, val_loader, test_loader = get_dataloaders(
            data_root=args.data_root,
            json_root=args.json_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            train_ratio=0.8,  # 固定为8:1:1的比例
            val_ratio=0.1,
            test_ratio=0.1,
            aug_prob=0.5,  # 应用数据增强
            use_extended_dataset=not args.no_extended,
            pin_memory=device.type == 'cuda',  # 在GPU上时启用pin_memory
            prefetch_factor=2  # 默认预取2个批次
        )
        
        # 打印数据分割信息
        print(f"数据集分割: 训练集={len(train_loader.dataset)}样本, 验证集={len(val_loader.dataset)}样本, 测试集={len(test_loader.dataset)}样本")
    except Exception as e:
        print(f"创建数据加载器时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模型
    try:
        model = get_model(model_type=args.model_type, in_channels=3, img_size=args.img_size)
        model = model.to(device)
        print(f"模型创建成功: {args.model_type}")
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数总量: {total_params:,}")
    except Exception as e:
        print(f"创建模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 定义损失函数
    position_criterion = torch.nn.CrossEntropyLoss()
    grade_criterion = torch.nn.MSELoss()
    
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 加载检查点（如果提供）
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            print(f"加载检查点: {args.checkpoint}")
            # 使用weights_only=False来处理PyTorch 2.6的兼容性问题
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            
            # 尝试加载模型状态
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("模型状态加载成功")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("模型状态加载成功")
            else:
                # 尝试直接加载权重
                model.load_state_dict(checkpoint)
                print("模型权重加载成功")
            
            # 可选: 尝试加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("优化器状态加载成功")
            
            # 可选: 获取之前训练的轮次
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"继续从轮次 {start_epoch} 开始")
            
            print("检查点加载成功")
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            print("将从头开始训练")
    
    # 设置混合精度训练
    try:
        if torch.cuda.is_available() and not args.no_cuda:
            # 使用新的推荐方式创建GradScaler
            scaler = torch.amp.GradScaler(device_type='cuda')
            print("启用混合精度训练")
        else:
            scaler = None
    except Exception as e:
        print(f"设置混合精度训练时出错: {e}")
        # 回退到旧版本的创建方式
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() and not args.no_cuda else None
        if scaler:
            print("启用混合精度训练 (旧版本API)")
    
    # 训练循环
    print(f"\n=== 开始训练 ({args.epochs} 轮) ===")
    best_val_loss = float('inf')
    best_f1 = 0.0
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"\n轮次 {epoch+1}/{start_epoch + args.epochs}")
        epoch_start = time.time()
        
        # 训练一个轮次
        train_metrics = train_epoch_optimized(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            position_criterion=position_criterion,
            grade_criterion=grade_criterion,
            device=device,
            task_weights=[0.5, 0.5],  # 位置和等级任务权重
            scaler=scaler
        )
        
        # 评估模型
        val_metrics = evaluate_model(
            model=model,
            val_loader=val_loader,
            position_criterion=position_criterion,
            grade_criterion=grade_criterion,
            device=device,
            task_weights=[0.5, 0.5]
        )
        
        epoch_time = time.time() - epoch_start
        
        # 打印训练结果
        print(f"\n轮次 {epoch+1} 完成，耗时: {epoch_time:.2f}秒")
        print(f"训练指标: 损失={train_metrics['loss']:.4f}, 位置准确率={train_metrics['position_accuracy']:.4f}, 等级MAE={train_metrics['grade_mae']:.4f}")
        print(f"验证指标: 损失={val_metrics['loss']:.4f}, 位置准确率={val_metrics['position_accuracy']:.4f}, F1={val_metrics['position_f1']:.4f}, 等级MAE={val_metrics['grade_mae']:.4f}")
        print(f"精确率: {val_metrics['position_precision']:.4f}, 召回率: {val_metrics['position_recall']:.4f}")
        print(f"各类别精确率: {', '.join([f'{p:.4f}' for p in val_metrics['position_precision_per_class']])}")
        print(f"各类别召回率: {', '.join([f'{r:.4f}' for r in val_metrics['position_recall_per_class']])}")
        print(f"各类别F1: {', '.join([f'{f1:.4f}' for f1 in val_metrics['position_f1_per_class']])}")
        
        # 是否保存为最佳模型
        is_best = False
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            is_best = True
        
        if val_metrics['position_f1'] > best_f1:
            best_f1 = val_metrics['position_f1']
            is_best = True
        
        # 每轮保存一次模型
        try:
            # 保存完整检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'position_accuracy': val_metrics['position_accuracy'],
                'position_f1': val_metrics['position_f1'],
                'position_precision': val_metrics['position_precision'],
                'position_recall': val_metrics['position_recall'],
                'grade_mae': val_metrics['grade_mae'],
                'best_val_loss': best_val_loss,
                'best_f1': best_f1
            }
            
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"已保存检查点: {checkpoint_path}")
            
            # 同时保存仅权重版本，避免兼容性问题
            weights_path = os.path.join(args.output_dir, f'weights_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), weights_path)
        except Exception as e:
            print(f"保存模型时出错: {e}")
            try:
                # 尝试只保存权重
                weights_path = os.path.join(args.output_dir, f'weights_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), weights_path)
                print(f"已保存模型权重: {weights_path}")
            except Exception as e2:
                print(f"保存权重也失败: {e2}")
        
        # 保存最佳模型
        if is_best:
            try:
                best_model_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"发现新的最佳模型! 已保存到: {best_model_path}")
            except Exception as e:
                print(f"保存最佳模型失败: {e}")
    
    print("\n=== 训练完成 ===")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳F1分数: {best_f1:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"已保存最终模型: {final_model_path}")
    except Exception as e:
        print(f"保存最终模型时出错: {e}")
    
    # 显示训练统计
    print(f"\n训练统计:")
    print(f"  模型类型: {args.model_type}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  图像大小: {args.img_size}x{args.img_size}")
    print(f"  输出目录: {os.path.abspath(args.output_dir)}")
    
    # 在最终模型上评估测试集性能
    print("\n=== 测试集评估 ===")
    test_metrics = evaluate_model(
        model=model,
        val_loader=test_loader,  # 使用测试数据加载器
        position_criterion=position_criterion,
        grade_criterion=grade_criterion,
        device=device,
        task_weights=[0.5, 0.5]
    )
    
    # 打印测试结果
    print(f"测试集指标:")
    print(f"  位置准确率: {test_metrics['position_accuracy']:.4f}")
    print(f"  位置F1分数: {test_metrics['position_f1']:.4f}")
    print(f"  位置精确率: {test_metrics['position_precision']:.4f}")
    print(f"  位置召回率: {test_metrics['position_recall']:.4f}")
    print(f"  等级MAE: {test_metrics['grade_mae']:.4f}")
    print(f"  各类别F1: {', '.join([f'{f1:.4f}' for f1 in test_metrics['position_f1_per_class']])}")
    
    # 保存测试结果到文件
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write("====== 测试集评估结果 ======\n")
        f.write(f"位置分类准确率: {test_metrics['position_accuracy']:.4f}\n")
        f.write(f"位置分类F1分数: {test_metrics['position_f1']:.4f}\n")
        f.write(f"位置分类精确率: {test_metrics['position_precision']:.4f}\n")
        f.write(f"位置分类召回率: {test_metrics['position_recall']:.4f}\n")
        f.write(f"等级预测MAE: {test_metrics['grade_mae']:.4f}\n")
        f.write(f"各类别F1: {', '.join([f'{f1:.4f}' for f1 in test_metrics['position_f1_per_class']])}\n")

if __name__ == "__main__":
    main() 