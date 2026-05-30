# [核心文件] UNet+Attention模型测试文件：用于评估训练好的模型在测试数据上的性能，生成预测结果和可视化
import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from torch.amp import autocast
import seaborn as sns

# 导入自定义模块
from dataset import CornRustDataset, get_dataloaders
from unet_model import get_unet_model

def evaluate_model(model, test_loader, device):
    """
    评估UNet+Attention模型在测试集上的性能
    
    参数:
        model: 已训练的模型
        test_loader: 测试数据加载器
        device: 计算设备
        
    返回:
        dict: 包含评估指标的字典
    """
    model.eval()  # 设置为评估模式
    
    # 收集所有预测和真实标签
    position_preds_all = []
    position_labels_all = []
    grade_values_all = []
    grade_labels_all = []
    
    # 收集分割结果示例（用于可视化）
    seg_examples = []
    
    with torch.no_grad():
        for batch_idx, (images, position_labels, grade_labels) in enumerate(tqdm(test_loader, desc="测试中")):
            # 将数据移动到指定设备
            images = images.to(device)
            position_labels = position_labels.to(device).view(-1).long()
            grade_labels = grade_labels.float().unsqueeze(1).to(device)
            
            # 使用混合精度计算
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                # 前向传播
                seg_pred, position_logits, grade_values = model(images)
            
            # 获取位置预测类别
            _, position_preds = torch.max(position_logits, 1)
            
            # 收集预测和标签
            position_preds_all.extend(position_preds.cpu().numpy())
            position_labels_all.extend(position_labels.cpu().numpy())
            grade_values_all.extend(grade_values.cpu().numpy())
            grade_labels_all.extend(grade_labels.cpu().numpy())
            
            # 保存一些分割示例
            if batch_idx < 5:  # 只保存前5个批次的示例
                for i in range(min(5, images.size(0))):  # 每个批次最多保存5个样本
                    seg_examples.append({
                        'image': images[i].cpu().numpy(),
                        'mask': seg_pred[i].cpu().numpy(),
                        'position': position_labels[i].item(),
                        'position_pred': position_preds[i].item(),
                        'grade': grade_labels[i].item(),
                        'grade_pred': grade_values[i].item()
                    })
    
    # 计算位置分类详细指标
    position_accuracy = accuracy_score(position_labels_all, position_preds_all)
    position_f1 = f1_score(position_labels_all, position_preds_all, average='macro')
    position_f1_per_class = f1_score(position_labels_all, position_preds_all, average=None)
    position_cm = confusion_matrix(position_labels_all, position_preds_all)
    position_precision = precision_score(position_labels_all, position_preds_all, average='macro')
    position_recall = recall_score(position_labels_all, position_preds_all, average='macro')
    
    # 计算等级回归指标
    grade_values_all = np.array(grade_values_all)
    grade_labels_all = np.array(grade_labels_all)
    grade_mae = np.mean(np.abs(grade_values_all - grade_labels_all))
    
    # 计算±2误差容忍率
    tolerance = 2.0
    grade_tolerance_accuracy = np.mean(np.abs(grade_values_all - grade_labels_all) <= tolerance)
    
    # 返回包含所有评估指标的字典
    return {
        'position_accuracy': position_accuracy,
        'position_f1': position_f1,
        'position_f1_per_class': position_f1_per_class,
        'position_precision': position_precision,
        'position_recall': position_recall,
        'position_cm': position_cm,
        'grade_mae': grade_mae,
        'grade_tolerance_accuracy': grade_tolerance_accuracy,
        'seg_examples': seg_examples
    }

def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_segmentation(examples, save_dir):
    """可视化分割结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    position_names = ['下部', '中部', '上部']
    
    for i, example in enumerate(examples):
        # 提取样本数据
        image = example['image']
        mask = example['mask'][0]  # 获取单通道掩码
        position = example['position']
        position_pred = example['position_pred']
        grade = example['grade']
        grade_pred = example['grade_pred']
        
        # 创建图像
        plt.figure(figsize=(12, 4))
        
        # 原始图像
        plt.subplot(1, 3, 1)
        # 将通道顺序从(C,H,W)转换为(H,W,C)用于显示
        img_display = np.transpose(image, (1, 2, 0))
        # 标准化显示范围
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        plt.imshow(img_display)
        plt.title('原始图像')
        plt.axis('off')
        
        # 预测的分割掩码
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='viridis')
        plt.title('分割掩码')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # 融合显示
        plt.subplot(1, 3, 3)
        plt.imshow(img_display)
        plt.imshow(mask, alpha=0.4, cmap='viridis')
        plt.title('融合显示')
        plt.axis('off')
        
        # 添加标题
        plt.suptitle(f'样本 #{i+1} - 位置: {position_names[position]}(预测:{position_names[position_pred]}), 等级: {grade:.1f}(预测:{grade_pred:.1f})', fontsize=14)
        
        # 保存图像
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f'segmentation_example_{i+1}.png'))
        plt.close()

def main(args):
    """主测试函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载测试数据
    if args.test_only:
        # 仅使用测试集
        print(f"加载测试数据集...")
        test_dataset = CornRustDataset(
            data_dir=args.data_root,
            json_dir=args.json_root,
            transform=None,
            img_size=args.img_size,
            use_extended_dataset=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        # 使用与训练时相同的数据划分，但只评估验证集部分
        _, test_loader = get_dataloaders(
            data_root=args.data_root,
            json_root=args.json_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            train_ratio=args.train_ratio,
            aug_prob=0.0  # 测试时不使用数据增强
        )
    
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 加载模型
    model = get_unet_model(in_channels=args.in_channels, img_size=args.img_size)
    
    # 加载模型权重
    if os.path.exists(args.model_path):
        print(f"加载模型权重: {args.model_path}")
        # 尝试加载完整检查点
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("模型加载成功")
        except Exception as e:
            print(f"加载检查点出错: {e}")
            print("尝试直接加载模型权重...")
            model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    model = model.to(device)
    model.eval()
    
    # 评估模型
    print("开始评估模型...")
    metrics = evaluate_model(model, test_loader, device)
    
    # 打印评估结果
    print("\n评估结果:")
    print(f"位置分类准确率: {metrics['position_accuracy']:.4f}")
    print(f"位置分类宏平均F1: {metrics['position_f1']:.4f}")
    print(f"位置分类精确率: {metrics['position_precision']:.4f}")
    print(f"位置分类召回率: {metrics['position_recall']:.4f}")
    print(f"等级预测MAE: {metrics['grade_mae']:.4f}")
    print(f"等级预测±2容忍率: {metrics['grade_tolerance_accuracy']:.4f}")
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        metrics['position_cm'],
        ['下部', '中部', '上部'],
        f"位置分类混淆矩阵 (准确率: {metrics['position_accuracy']:.4f})",
        os.path.join(args.output_dir, 'position_confusion_matrix.png')
    )
    
    # 可视化分割结果
    if len(metrics['seg_examples']) > 0:
        print("可视化分割结果...")
        visualize_segmentation(
            metrics['seg_examples'],
            os.path.join(args.output_dir, 'segmentation_examples')
        )
    
    # 保存评估结果到文本文件
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("UNet+Attention模型评估结果\n")
        f.write("===========================\n\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"数据根目录: {args.data_root}\n")
        f.write(f"测试集大小: {len(test_loader.dataset)}\n\n")
        
        f.write("位置分类指标:\n")
        f.write(f"  准确率: {metrics['position_accuracy']:.4f}\n")
        f.write(f"  宏平均F1: {metrics['position_f1']:.4f}\n")
        f.write(f"  精确率: {metrics['position_precision']:.4f}\n")
        f.write(f"  召回率: {metrics['position_recall']:.4f}\n\n")
        
        f.write("等级预测指标:\n")
        f.write(f"  平均绝对误差(MAE): {metrics['grade_mae']:.4f}\n")
        f.write(f"  ±2容忍率: {metrics['grade_tolerance_accuracy']:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="玉米南方锈病UNet+Attention模型测试")
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='数据根目录路径')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注根目录路径')
    parser.add_argument('--img_size', type=int, default=128,
                        help='图像大小')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='输入图像通道数')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例，用于数据划分')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重文件路径')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--no_cuda', action='store_true',
                        help='不使用CUDA')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--test_only', action='store_true',
                        help='仅使用测试集评估，不使用训练/验证划分')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./unet_test_results',
                        help='输出目录路径')
    
    args = parser.parse_args()
    main(args) 