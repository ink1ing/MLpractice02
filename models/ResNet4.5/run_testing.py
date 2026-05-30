#!/usr/bin/env python
# 测试启动脚本：加载训练好的模型并在测试集上评估性能

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

from dataset import CornRustDataset, get_dataloaders
from model import get_model
from utils import calculate_metrics

def main():
    """
    使用训练好的模型在测试集上进行评估
    """
    parser = argparse.ArgumentParser(description='玉米南方锈病模型测试脚本')
    
    # 基本参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='数据根目录路径')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注根目录路径')
    parser.add_argument('--output_dir', type=str, default='./test_output',
                        help='测试结果输出目录')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--model_type', type=str, default='resnet_plus',
                        choices=['simple', 'resnet', 'resnet_plus'],
                        help='模型类型')
    parser.add_argument('--img_size', type=int, default=128,
                        help='图像大小')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='输入通道数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--no_cuda', action='store_true',
                        help='不使用CUDA')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n数据配置:")
    print(f"数据根目录: {os.path.abspath(args.data_root)}")
    print(f"JSON标注目录: {os.path.abspath(args.json_root)}")
    
    # 加载数据
    print("\n加载数据...")
    try:
        # 获取数据加载器，直接使用测试集部分
        train_loader, val_loader, test_loader = get_dataloaders(
            data_root=args.data_root,
            json_root=args.json_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            train_ratio=0.8,  # 使用标准8:1:1的数据分割
            val_ratio=0.1,
            test_ratio=0.1,
            aug_prob=0.0,     # 测试时不使用数据增强
            use_extended_dataset=True
        )
        print(f"使用测试集进行评估，样本数量: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    model = get_model(model_type=args.model_type, in_channels=args.in_channels, img_size=args.img_size)
    
    try:
        # 尝试加载完整检查点
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("已加载完整检查点")
        else:
            # 尝试加载仅权重
            model.load_state_dict(checkpoint)
            print("已加载模型权重")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # 收集预测和真实标签
    position_preds_all = []
    position_labels_all = []
    grade_values_all = []
    grade_labels_all = []
    
    print("\n开始测试...")
    with torch.no_grad():
        for images, position_labels, grade_labels in tqdm(test_loader, desc="测试中"):
            images = images.to(device)
            position_labels = position_labels.to(device)
            grade_labels = grade_labels.float().unsqueeze(1).to(device)
            
            # 前向传播
            position_logits, grade_values = model(images)
            
            # 获取位置预测
            _, position_preds = torch.max(position_logits, 1)
            
            # 收集预测和标签
            position_preds_all.extend(position_preds.cpu().numpy())
            position_labels_all.extend(position_labels.cpu().numpy())
            grade_values_all.extend(grade_values.cpu().numpy())
            grade_labels_all.extend(grade_labels.cpu().numpy())
    
    # 计算位置分类指标
    position_accuracy = accuracy_score(position_labels_all, position_preds_all)
    position_f1 = f1_score(position_labels_all, position_preds_all, average='macro')
    position_f1_per_class = f1_score(position_labels_all, position_preds_all, average=None)
    position_precision = precision_score(position_labels_all, position_preds_all, average='macro')
    position_recall = recall_score(position_labels_all, position_preds_all, average='macro')
    position_cm = confusion_matrix(position_labels_all, position_preds_all)
    
    # 计算等级预测指标
    grade_values_all = np.array(grade_values_all)
    grade_labels_all = np.array(grade_labels_all)
    grade_mae = np.mean(np.abs(grade_values_all - grade_labels_all))
    
    # 计算±2误差容忍率
    tolerance = 2.0
    grade_tolerance_accuracy = np.mean(np.abs(grade_values_all - grade_labels_all) <= tolerance)
    
    # 打印测试结果
    print("\n====== 测试结果 ======")
    print(f"位置分类准确率: {position_accuracy:.4f}")
    print(f"位置分类F1分数: {position_f1:.4f}")
    print(f"位置分类精确率: {position_precision:.4f}")
    print(f"位置分类召回率: {position_recall:.4f}")
    print(f"位置分类每类F1: {position_f1_per_class}")
    print(f"等级预测MAE: {grade_mae:.4f}")
    print(f"等级预测±2容忍率: {grade_tolerance_accuracy:.4f}")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(position_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['下部', '中部', '上部'],
                yticklabels=['下部', '中部', '上部'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f"位置分类混淆矩阵 (准确率: {position_accuracy:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'position_confusion_matrix.png'))
    
    # 绘制等级预测散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(grade_labels_all, grade_values_all, alpha=0.5)
    plt.plot([0, 9], [0, 9], 'r--')  # 理想预测线
    plt.plot([0, 9], [0, 9+tolerance], 'g--')  # 上容忍边界
    plt.plot([0, 9], [0, 9-tolerance], 'g--')  # 下容忍边界
    plt.xlabel('真实等级')
    plt.ylabel('预测等级')
    plt.title(f"等级预测散点图 (MAE: {grade_mae:.4f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'grade_prediction.png'))
    
    # 保存测试结果到文件
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write("====== 测试结果 ======\n")
        f.write(f"位置分类准确率: {position_accuracy:.4f}\n")
        f.write(f"位置分类F1分数: {position_f1:.4f}\n")
        f.write(f"位置分类精确率: {position_precision:.4f}\n")
        f.write(f"位置分类召回率: {position_recall:.4f}\n")
        f.write(f"位置分类每类F1: {position_f1_per_class}\n")
        f.write(f"等级预测MAE: {grade_mae:.4f}\n")
        f.write(f"等级预测±2容忍率: {grade_tolerance_accuracy:.4f}\n")
    
    print(f"\n测试结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 