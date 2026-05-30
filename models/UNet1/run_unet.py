#!/usr/bin/env python
# -*- coding: utf-8 -*-
# [核心文件] 运行脚本：用于执行UNet模型训练、验证和测试的主入口，支持命令行参数配置

import argparse
import os
import torch
import numpy as np
import random
from datetime import datetime
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt

# 导入自定义模块
from model import get_model
from dataset import get_dataloaders
from utils import FocalLoss
from train import evaluate, train_one_epoch, plot_metrics, plot_confusion_matrix

def setup_environment():
    """
    设置运行环境，包括环境变量、随机种子等
    """
    # 设置TIF图像相关环境变量，提高加载性能
    os.environ['GDAL_PAM_ENABLED'] = 'NO'  # 禁用GDAL PAM文件生成
    
    # 确保使用确定性算法，增强可重复性
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 针对CUDA 10.2及以上版本
    
    # 如果有NVDECODE和NVENCODE库，启用硬件加速图像处理
    os.environ['OPENCV_OPENCL_RUNTIME'] = ''
    
    # 设置PyTorch相关环境变量
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # 启用cuDNN v8 API
    os.environ['TORCH_USE_CUDA_DSA'] = '1'  # 使用CUDA设备端分配
    
    print("环境变量设置完成")

def set_seed(seed=42):
    """
    设置随机种子，确保实验可重复
    
    参数:
        seed: 随机种子值，默认为42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 关闭自动调优，确保可重复性
    print(f"随机种子已设置为: {seed}")

def parse_arguments():
    """
    解析命令行参数
    
    返回:
        args: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='UNet模型训练脚本 - 玉米南方锈病识别')
    
    # 基本参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='多光谱图像数据根目录')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注根目录')
    parser.add_argument('--output_dir', type=str, default='./unet_output',
                        help='输出目录，用于保存模型和结果')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'unet++'],
                        help='模型类型: unet或unet++')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='输入通道数')
    parser.add_argument('--out_channels', type=int, default=1,
                        help='分割输出通道数')
    parser.add_argument('--img_size', type=int, default=128,
                        help='图像大小')
    parser.add_argument('--bilinear', action='store_true',
                        help='使用双线性插值代替转置卷积进行上采样')
    parser.add_argument('--attention', action='store_true', default=True,
                        help='是否使用注意力机制')
    parser.add_argument('--features', type=str, default='64,128,256,512',
                        help='特征通道数，以逗号分隔')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='最小学习率，用于学习率调度')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减系数')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='优化器: adam或sgd')
    parser.add_argument('--lr_scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='学习率调度器: plateau, cosine或none')
    parser.add_argument('--task_weights', type=str, default='0.7,0.2,0.1',
                        help='任务权重: 分割,位置分类,等级回归，以逗号分隔')
    parser.add_argument('--no_early_stopping', action='store_true',
                        help='禁用早停策略')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值，连续多少轮无改善时停止训练')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 数据处理参数
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--aug_prob', type=float, default=0.7,
                        help='数据增强概率')
    
    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['ce', 'focal', 'dice'],
                        help='损失函数类型: ce(交叉熵), focal或dice')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss的gamma参数')
    
    # 性能相关参数
    parser.add_argument('--amp', action='store_true', default=True,
                        help='启用自动混合精度训练')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA，即使可用也使用CPU')
    
    # 加载和恢复参数
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--load_weights', type=str, default=None,
                        help='加载预训练权重的路径')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='开始训练的轮次')
    
    args = parser.parse_args()
    
    # 解析特征通道数
    args.features = [int(f) for f in args.features.split(',')]
    
    # 解析任务权重
    args.task_weights = [float(w) for w in args.task_weights.split(',')]
    
    # 检查必要目录是否存在，不存在则创建
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def dice_loss(pred, target, smooth=1.0):
    """
    计算Dice损失，用于分割任务
    
    参数:
        pred: 预测值
        target: 目标值
        smooth: 平滑项，防止分母为0
        
    返回:
        dice_loss: Dice损失值
    """
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def segmentation_loss(pred, target, loss_type='ce'):
    """
    计算分割损失，支持多种损失函数
    
    参数:
        pred: 预测值
        target: 目标值
        loss_type: 损失函数类型，支持'ce'(交叉熵)，'focal'和'dice'
        
    返回:
        loss: 损失值
    """
    if loss_type == 'dice':
        return dice_loss(pred, target)
    elif loss_type == 'focal':
        focal_loss = FocalLoss(gamma=2.0)
        return focal_loss(pred, target)
    else:  # 默认使用BCE损失
        return F.binary_cross_entropy_with_logits(pred, target)

def main():
    """主函数，包含完整的训练和评估流程"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置运行环境
    setup_environment()
    
    # 设置随机种子确保可重复性
    set_seed(args.seed)
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 混合精度训练设置
    use_amp = args.amp and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("启用混合精度训练 (AMP)")
        
    # 创建数据加载器
    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        json_root=args.json_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        train_ratio=args.train_ratio,
        aug_prob=args.aug_prob,
        use_extended_dataset=True,
        pin_memory=torch.cuda.is_available()
    )
    
    # 创建模型
    model = get_model(
        model_type=args.model_type,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=args.img_size,
        bilinear=args.bilinear,
        with_attention=args.attention,
        features=args.features
    )
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 定义优化器
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # 定义学习率调度器
    if args.lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=args.min_lr, verbose=True
        )
    elif args.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    else:
        scheduler = None
    
    # 恢复训练（如果提供了检查点）
    start_epoch = args.start_epoch
    best_val_loss = float('inf')
    best_f1 = 0.0
    metrics_history = {'train_loss': [], 'val_loss': [], 
                       'train_seg_loss': [], 'val_seg_loss': [],
                       'train_position_accuracy': [], 'val_position_accuracy': [],
                       'train_position_f1': [], 'val_position_f1': [],
                       'train_grade_mae': [], 'val_grade_mae': [],
                       'val_position_precision': [], 'val_position_recall': [],
                       'val_grade_tolerance': []}
    
    # 加载权重或检查点
    if args.load_weights and os.path.isfile(args.load_weights):
        try:
            model.load_state_dict(torch.load(args.load_weights, map_location=device))
            print(f"从权重文件加载参数: {args.load_weights}")
        except Exception as e:
            print(f"加载权重文件时出错: {e}")
    elif args.resume and os.path.isfile(args.resume):
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            if 'best_f1' in checkpoint:
                best_f1 = checkpoint['best_f1']
            if 'metrics_history' in checkpoint:
                metrics_history = checkpoint['metrics_history']
            print(f"从检查点恢复训练: {args.resume}")
            print(f"继续从轮次 {start_epoch} 开始")
        except Exception as e:
            print(f"加载检查点时出错: {e}")
    
    # 定义BCE损失函数用于分割
    bce_loss = torch.nn.BCEWithLogitsLoss()
    
    # 定义位置分类损失函数
    if args.loss_type == 'focal':
        position_criterion = FocalLoss(gamma=args.focal_gamma)
    else:
        position_criterion = torch.nn.CrossEntropyLoss()
    
    # 定义等级回归损失函数
    grade_criterion = torch.nn.MSELoss()
    
    # 训练循环
    print(f"\n开始训练，共 {args.epochs} 轮...")
    no_improvement_count = 0
    
    # 记录训练开始时间
    start_time = datetime.now()
    print(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n轮次 {epoch+1}/{args.epochs}")
        
        # 训练一个轮次
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            position_criterion=position_criterion,
            grade_criterion=grade_criterion,
            device=device,
            task_weights=args.task_weights,
            scaler=scaler,
            seg_criterion=bce_loss
        )
        
        # 验证
        val_metrics = evaluate(
            model=model,
            val_loader=val_loader,
            position_criterion=position_criterion,
            grade_criterion=grade_criterion,
            device=device,
            task_weights=args.task_weights,
            seg_criterion=bce_loss
        )
        
        # 更新指标历史
        metrics_history['train_loss'].append(train_metrics['loss'])
        metrics_history['val_loss'].append(val_metrics['loss'])
        metrics_history['train_seg_loss'].append(train_metrics.get('seg_loss', 0))
        metrics_history['val_seg_loss'].append(val_metrics.get('seg_loss', 0))
        metrics_history['train_position_accuracy'].append(train_metrics['position_accuracy'])
        metrics_history['val_position_accuracy'].append(val_metrics['position_accuracy'])
        metrics_history['train_position_f1'].append(train_metrics.get('position_f1', 0))
        metrics_history['val_position_f1'].append(val_metrics['position_f1'])
        metrics_history['train_grade_mae'].append(train_metrics['grade_mae'])
        metrics_history['val_grade_mae'].append(val_metrics['grade_mae'])
        metrics_history['val_position_precision'].append(val_metrics['position_precision'])
        metrics_history['val_position_recall'].append(val_metrics['position_recall'])
        metrics_history['val_grade_tolerance'].append(val_metrics['grade_tolerance_accuracy'])
        
        # 打印当前性能
        print(f"训练: 损失={train_metrics['loss']:.4f}, 位置准确率={train_metrics['position_accuracy']:.4f}, 等级MAE={train_metrics['grade_mae']:.4f}")
        print(f"验证: 损失={val_metrics['loss']:.4f}, 位置准确率={val_metrics['position_accuracy']:.4f}, 位置F1={val_metrics['position_f1']:.4f}, 等级MAE={val_metrics['grade_mae']:.4f}")
        
        # 更新学习率调度器
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # 保存最佳模型和最新模型
        current_val_loss = val_metrics['loss']
        current_f1 = val_metrics['position_f1']
        
        is_best = False
        
        # 根据验证损失或F1分数确定最佳模型
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            is_best = True
            no_improvement_count = 0
        elif current_f1 > best_f1:
            best_f1 = current_f1
            is_best = True
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_f1': best_f1,
            'metrics_history': metrics_history,
            'args': args
        }
        
        # 保存最后一轮模型
        torch.save(checkpoint, os.path.join(args.output_dir, 'last_model.pth'))
        
        # 保存最佳模型
        if is_best:
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print("发现最佳模型，已保存!")
        
        # 早停判断
        if not args.no_early_stopping and no_improvement_count >= args.patience:
            print(f"验证指标 {args.patience} 轮没有改善，提前停止训练")
            break
        
        # 每10轮绘制一次中间指标图表
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            plot_metrics(metrics_history, args.output_dir)
    
    # 训练结束时间
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"\n训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总训练时长: {training_duration}")
    
    # 加载最佳模型进行最终评估
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"加载最佳模型进行最终评估: {best_model_path}")
    
    # 最终验证评估
    final_metrics = evaluate(
        model=model,
        val_loader=val_loader,
        position_criterion=position_criterion,
        grade_criterion=grade_criterion,
        device=device,
        task_weights=args.task_weights,
        seg_criterion=bce_loss
    )
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        final_metrics['position_cm'],
        ['下部', '中部', '上部'],
        f"位置分类混淆矩阵 (准确率: {final_metrics['position_accuracy']:.4f})",
        os.path.join(args.output_dir, 'position_confusion_matrix.png')
    )
    
    # 绘制最终指标图表
    plot_metrics(metrics_history, args.output_dir)
    
    # 保存训练配置和最终结果到README.md
    with open(os.path.join(args.output_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(f"# 玉米南方锈病UNet模型训练结果\n\n")
        f.write(f"## 训练配置\n\n")
        f.write(f"- 模型类型: {args.model_type}\n")
        f.write(f"- 输入通道数: {args.in_channels}\n")
        f.write(f"- 输出通道数: {args.out_channels}\n")
        f.write(f"- 图像尺寸: {args.img_size}x{args.img_size}\n")
        f.write(f"- 使用双线性插值: {args.bilinear}\n")
        f.write(f"- 使用注意力机制: {args.attention}\n")
        f.write(f"- 特征通道数: {args.features}\n")
        f.write(f"- 批次大小: {args.batch_size}\n")
        f.write(f"- 训练轮数: {epoch+1}\n")
        f.write(f"- 学习率: {args.lr}\n")
        f.write(f"- 优化器: {args.optimizer}\n")
        f.write(f"- 学习率调度器: {args.lr_scheduler}\n")
        f.write(f"- 任务权重: {args.task_weights}\n")
        f.write(f"- 损失函数: {args.loss_type}\n")
        f.write(f"- 混合精度训练: {use_amp}\n")
        f.write(f"- 训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 总训练时长: {training_duration}\n\n")
        
        f.write(f"## 最终评估结果\n\n")
        f.write(f"- 验证损失: {final_metrics['loss']:.4f}\n")
        f.write(f"- 位置分类准确率: {final_metrics['position_accuracy']:.4f}\n")
        f.write(f"- 位置分类F1分数: {final_metrics['position_f1']:.4f}\n")
        f.write(f"- 位置分类精确率: {final_metrics['position_precision']:.4f}\n")
        f.write(f"- 位置分类召回率: {final_metrics['position_recall']:.4f}\n")
        f.write(f"- 等级预测平均绝对误差(MAE): {final_metrics['grade_mae']:.4f}\n")
        f.write(f"- 等级预测容忍率(±2): {final_metrics['grade_tolerance_accuracy']:.4f}\n\n")
        
        f.write(f"## 每类F1分数\n\n")
        f.write(f"- 下部: {final_metrics['position_f1_per_class'][0]:.4f}\n")
        f.write(f"- 中部: {final_metrics['position_f1_per_class'][1]:.4f}\n")
        f.write(f"- 上部: {final_metrics['position_f1_per_class'][2]:.4f}\n\n")
        
        f.write(f"## 模型使用方法\n\n")
        f.write(f"```python\n")
        f.write(f"import torch\n")
        f.write(f"from model import get_model\n\n")
        f.write(f"# 创建模型\n")
        f.write(f"model = get_model(\n")
        f.write(f"    model_type='{args.model_type}',\n")
        f.write(f"    in_channels={args.in_channels},\n") 
        f.write(f"    out_channels={args.out_channels},\n")
        f.write(f"    img_size={args.img_size},\n")
        f.write(f"    bilinear={args.bilinear},\n")
        f.write(f"    with_attention={args.attention},\n")
        f.write(f"    features={args.features}\n")
        f.write(f")\n\n")
        f.write(f"# 加载训练好的权重\n")
        f.write(f"model.load_state_dict(torch.load('best_model.pth'))\n")
        f.write(f"model.eval()\n\n")
        f.write(f"# 使用模型进行预测\n")
        f.write(f"with torch.no_grad():\n")
        f.write(f"    segmentation, position_logits, grade_values = model(input_image)\n")
        f.write(f"    \n")
        f.write(f"    # 处理分割结果\n")
        f.write(f"    segmentation = torch.sigmoid(segmentation) > 0.5  # 二值化\n")
        f.write(f"    \n")
        f.write(f"    # 处理位置分类结果\n")
        f.write(f"    position_pred = torch.argmax(position_logits, dim=1)  # 获取预测类别\n")
        f.write(f"    \n")
        f.write(f"    # 处理等级回归结果\n")
        f.write(f"    predicted_grade = grade_values.item()  # 获取预测等级值\n")
        f.write(f"```\n")
    
    # 打印最终结果
    print("\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳位置F1分数: {best_f1:.4f}")
    print(f"位置预测准确率: {final_metrics['position_accuracy']:.4f}")
    print(f"位置预测F1分数: {final_metrics['position_f1']:.4f}")
    print(f"位置预测精确率: {final_metrics['position_precision']:.4f}")
    print(f"位置预测召回率: {final_metrics['position_recall']:.4f}")
    print(f"等级预测MAE: {final_metrics['grade_mae']:.4f}")
    print(f"等级预测容忍率(±2): {final_metrics['grade_tolerance_accuracy']:.4f}")
    print(f"训练结果保存在: {args.output_dir}")
    
    return model, final_metrics

if __name__ == "__main__":
    main() 