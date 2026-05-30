#!/usr/bin/env python
# -*- coding: utf-8 -*-
# [核心文件] 测试脚本：用于评估UNet模型在测试集上的性能，并可视化分割结果

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from torch.cuda.amp import autocast
import random
from tqdm import tqdm
import seaborn as sns

# 导入自定义模块
from model import get_model
from dataset import CornRustDataset
from train import evaluate
from utils import FocalLoss

def set_seed(seed=42):
    """设置随机种子确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='UNet模型测试脚本 - 玉米南方锈病识别')
    
    # 数据路径参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='多光谱图像数据根目录')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注根目录')
    parser.add_argument('--output_dir', type=str, default='./unet_test_results',
                        help='输出目录，用于保存测试结果')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重文件路径，必须提供')
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
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA，即使可用也使用CPU')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='是否可视化分割结果')
    parser.add_argument('--num_visualize', type=int, default=10,
                        help='可视化样本数量')
    
    args = parser.parse_args()
    
    # 解析特征通道数
    args.features = [int(f) for f in args.features.split(',')]
    
    # 检查必要目录是否存在，不存在则创建
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def create_test_dataloader(args):
    """创建测试数据加载器"""
    print(f"正在创建测试数据加载器...")
    
    # 创建测试数据集 - 不应用数据增强
    test_dataset = CornRustDataset(
        data_dir=args.data_root,
        json_dir=args.json_root,
        transform=None,  # 测试不使用数据增强
        img_size=args.img_size,
        use_extended_dataset=True
    )
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 测试不需要打乱数据
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.no_cuda
    )
    
    print(f"测试集样本数: {len(test_dataset)}")
    return test_loader

def load_model(args, device):
    """加载模型"""
    print(f"正在加载模型: {args.model_path}")
    
    # 创建模型实例
    model = get_model(
        model_type=args.model_type,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=args.img_size,
        bilinear=args.bilinear,
        with_attention=args.attention,
        features=args.features
    )
    
    # 加载模型权重
    try:
        # 尝试直接加载状态字典
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"直接加载状态字典失败: {e}")
        try:
            # 尝试从检查点加载
            checkpoint = torch.load(args.model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("检查点不包含模型状态字典，尝试直接加载检查点作为状态字典")
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
    
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params:,}")
    
    return model

def visualize_results(images, segmentations, position_preds, grade_preds, position_labels, grade_labels, args, indices=None):
    """可视化分割结果和预测"""
    # 位置类别名称
    position_names = ['下部', '中部', '上部']
    
    # 如果没有提供索引，则随机选择样本
    if indices is None:
        num_samples = min(args.num_visualize, len(images))
        indices = random.sample(range(len(images)), num_samples)
    
    for i, idx in enumerate(indices):
        # 获取当前样本
        image = images[idx].cpu().numpy().transpose(1, 2, 0)  # 转为HWC格式
        segmentation = segmentations[idx].cpu().numpy().squeeze()  # 移除通道维度
        position_pred = position_preds[idx]
        grade_pred = grade_preds[idx]
        position_label = position_labels[idx]
        grade_label = grade_labels[idx]
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(np.clip(image, 0, 1))
        axes[0].set_title(f"原始图像\n真实位置: {position_names[position_label]}, 真实等级: {grade_label:.2f}")
        axes[0].axis('off')
        
        # 分割结果
        axes[1].imshow(segmentation, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f"分割结果\n预测位置: {position_names[position_pred]}, 预测等级: {grade_pred:.2f}")
        axes[1].axis('off')
        
        # 原始图像+分割结果叠加
        axes[2].imshow(np.clip(image, 0, 1))
        mask = segmentation > 0.5  # 二值化分割结果
        overlay = np.zeros_like(image)
        overlay[:, :, 0] = mask * 1.0  # 红色通道表示分割区域
        axes[2].imshow(overlay, alpha=0.5)  # 叠加分割结果
        axes[2].set_title("叠加显示")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'sample_{i+1}.png'), dpi=150)
        plt.close()

def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('位置分类混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_model(model, test_loader, device, args):
    """测试模型并评估性能"""
    print("\n开始测试模型...")
    
    # 定义损失函数
    position_criterion = FocalLoss(gamma=2.0)
    grade_criterion = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    
    # 任务权重 - 分割:位置分类:等级回归 = 0.7:0.2:0.1
    task_weights = [0.7, 0.2, 0.1]
    
    # 使用evaluate函数评估
    metrics = evaluate(
        model=model,
        val_loader=test_loader,
        position_criterion=position_criterion,
        grade_criterion=grade_criterion,
        device=device,
        task_weights=task_weights,
        seg_criterion=bce_loss
    )
    
    # 打印评估结果
    print("\n测试结果:")
    print(f"总损失: {metrics['loss']:.4f}")
    print(f"分割损失: {metrics.get('seg_loss', 'N/A')}")
    print(f"位置分类损失: {metrics['position_loss']:.4f}")
    print(f"等级回归损失: {metrics['grade_loss']:.4f}")
    print(f"位置分类准确率: {metrics['position_accuracy']:.4f}")
    print(f"位置分类F1分数: {metrics['position_f1']:.4f}")
    print(f"位置分类精确率: {metrics['position_precision']:.4f}")
    print(f"位置分类召回率: {metrics['position_recall']:.4f}")
    print(f"等级预测MAE: {metrics['grade_mae']:.4f}")
    print(f"等级预测容忍率(±2): {metrics['grade_tolerance_accuracy']:.4f}")
    
    # 保存混淆矩阵
    plot_confusion_matrix(
        metrics['position_cm'],
        ['下部', '中部', '上部'],
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # 如果需要可视化，收集一些样本
    if args.visualize:
        # 收集可视化所需的数据
        all_images = []
        all_segmentations = []
        all_position_preds = []
        all_grade_preds = []
        all_position_labels = []
        all_grade_labels = []
        
        print("\n收集可视化样本...")
        with torch.no_grad():
            for images, position_labels, grade_labels in tqdm(test_loader, desc="处理批次"):
                # 将数据移动到设备
                images = images.to(device)
                position_labels = position_labels.to(device)
                grade_labels = grade_labels.float().unsqueeze(1).to(device)
                
                # 前向传播
                with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    segmentations, position_logits, grade_values = model(images)
                    
                # 获取预测
                segmentations = torch.sigmoid(segmentations)  # 应用sigmoid获取概率值
                _, position_preds = torch.max(position_logits, 1)
                
                # 收集数据
                all_images.extend(images.cpu())
                all_segmentations.extend(segmentations.cpu())
                all_position_preds.extend(position_preds.cpu().numpy())
                all_grade_preds.extend(grade_values.cpu().numpy().flatten())
                all_position_labels.extend(position_labels.cpu().numpy())
                all_grade_labels.extend(grade_labels.cpu().numpy().flatten())
        
        # 可视化结果
        print(f"\n可视化 {args.num_visualize} 个样本的结果...")
        visualize_results(
            all_images,
            all_segmentations,
            all_position_preds,
            all_grade_preds,
            all_position_labels,
            all_grade_labels,
            args
        )
    
    # 保存测试结果到文本文件
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write("UNet模型测试结果\n\n")
        f.write(f"模型类型: {args.model_type}\n")
        f.write(f"模型路径: {args.model_path}\n\n")
        f.write(f"总损失: {metrics['loss']:.4f}\n")
        f.write(f"分割损失: {metrics.get('seg_loss', 'N/A')}\n")
        f.write(f"位置分类损失: {metrics['position_loss']:.4f}\n")
        f.write(f"等级回归损失: {metrics['grade_loss']:.4f}\n")
        f.write(f"位置分类准确率: {metrics['position_accuracy']:.4f}\n")
        f.write(f"位置分类F1分数: {metrics['position_f1']:.4f}\n")
        f.write(f"位置分类精确率: {metrics['position_precision']:.4f}\n")
        f.write(f"位置分类召回率: {metrics['position_recall']:.4f}\n")
        f.write(f"等级预测MAE: {metrics['grade_mae']:.4f}\n")
        f.write(f"等级预测容忍率(±2): {metrics['grade_tolerance_accuracy']:.4f}\n\n")
        
        f.write("每类F1分数:\n")
        for i, class_name in enumerate(['下部', '中部', '上部']):
            f.write(f"{class_name}: {metrics['position_f1_per_class'][i]:.4f}\n")
    
    return metrics

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据加载器
    test_loader = create_test_dataloader(args)
    
    # 加载模型
    model = load_model(args, device)
    
    # 测试模型
    metrics = test_model(model, test_loader, device, args)
    
    print(f"\n测试完成! 结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 