#!/usr/bin/env python
# 核心文件：测试脚本，验证LodgeNet模型的结构和性能
# LodgeNet测试脚本：用于测试模型架构、加载预训练模型和进行推理
# 支持模型验证、性能评估和可视化结果

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 导入自定义模块
from lodgenet_model import get_lodgenet_model, count_parameters
from dataset import CornRustDataset, get_dataloaders
from lodgenet_train import evaluate, create_dummy_segmentation_labels, CombinedSegmentationLoss
from utils import FocalLoss

def test_model_architecture():
    """
    测试LodgeNet模型架构
    """
    print("====== 测试LodgeNet模型架构 ======")
    
    # 创建模型
    model = get_lodgenet_model(n_channels=3, n_classes=2, img_size=128)
    
    print(f"模型参数数量: {count_parameters(model):,}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 128, 128)
    
    model.eval()
    with torch.no_grad():
        seg_output, pos_output, grade_output = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"分割输出形状: {seg_output.shape}")
    print(f"位置分类输出形状: {pos_output.shape}")
    print(f"病害等级回归输出形状: {grade_output.shape}")
    
    # 验证输出维度
    assert seg_output.shape == (batch_size, 2, 128, 128), f"分割输出形状错误: {seg_output.shape}"
    assert pos_output.shape == (batch_size, 3), f"位置分类输出形状错误: {pos_output.shape}"
    assert grade_output.shape == (batch_size, 1), f"病害等级输出形状错误: {grade_output.shape}"
    
    print("✅ 模型架构测试通过!")
    return model

def test_loss_functions():
    """
    测试损失函数
    """
    print("\n====== 测试损失函数 ======")
    
    batch_size = 4
    img_size = 128
    
    # 创建虚拟数据
    seg_output = torch.randn(batch_size, 2, img_size, img_size)
    seg_labels = torch.randint(0, 2, (batch_size, img_size, img_size))
    
    pos_output = torch.randn(batch_size, 3)
    pos_labels = torch.randint(0, 3, (batch_size,))
    
    grade_output = torch.randn(batch_size, 1)
    grade_labels = torch.rand(batch_size, 1) * 9  # 0-9范围
    
    # 测试损失函数
    seg_criterion = CombinedSegmentationLoss()
    pos_criterion = FocalLoss(alpha=None, gamma=2.0)
    grade_criterion = nn.MSELoss()
    
    seg_loss = seg_criterion(seg_output, seg_labels)
    pos_loss = pos_criterion(pos_output, pos_labels)
    grade_loss = grade_criterion(grade_output, grade_labels)
    
    print(f"分割损失: {seg_loss.item():.4f}")
    print(f"位置分类损失: {pos_loss.item():.4f}")
    print(f"病害等级回归损失: {grade_loss.item():.4f}")
    
    print("✅ 损失函数测试通过!")

def load_and_test_model(model_path, data_root, json_root, device='cpu'):
    """
    加载预训练模型并进行测试
    
    参数:
        model_path: 模型文件路径
        data_root: 数据根目录
        json_root: JSON标注根目录
        device: 计算设备
    """
    print(f"\n====== 加载并测试模型: {model_path} ======")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型实例
    model = get_lodgenet_model(n_channels=3, n_classes=2, img_size=128).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型加载成功，训练轮次: {checkpoint.get('epoch', 'N/A')}")
    
    # 加载测试数据
    print("加载测试数据...")
    try:
        _, val_loader = get_dataloaders(
            data_root=data_root,
            json_root=json_root,
            batch_size=8,
            num_workers=0,
            img_size=128,
            train_ratio=0.8,
            use_extended_dataset=True
        )
        
        print(f"测试集大小: {len(val_loader.dataset)}")
        
        # 定义损失函数
        seg_criterion = CombinedSegmentationLoss()
        pos_criterion = FocalLoss(alpha=None, gamma=2.0)
        grade_criterion = nn.MSELoss()
        
        # 评估模型
        print("评估模型性能...")
        metrics = evaluate(
            model, val_loader, seg_criterion, pos_criterion, 
            grade_criterion, device, task_weights=[0.4, 0.3, 0.3]
        )
        
        # 打印结果
        print(f"\n模型性能指标:")
        print(f"  总损失: {metrics['loss']:.4f}")
        print(f"  分割损失: {metrics['seg_loss']:.4f}")
        print(f"  位置分类损失: {metrics['position_loss']:.4f}")
        print(f"  病害等级回归损失: {metrics['grade_loss']:.4f}")
        print(f"  位置分类准确率: {metrics['position_accuracy']:.4f}")
        print(f"  位置分类F1分数: {metrics['position_f1']:.4f}")
        print(f"  位置分类精确率: {metrics['position_precision']:.4f}")
        print(f"  位置分类召回率: {metrics['position_recall']:.4f}")
        print(f"  病害等级MAE: {metrics['grade_mae']:.4f}")
        print(f"  病害等级RMSE: {metrics['grade_rmse']:.4f}")
        
        # 检查是否达到目标指标
        target_met = (metrics['position_accuracy'] > 0.90 and 
                     metrics['position_f1'] > 0.85 and
                     metrics['position_recall'] > 0.85 and
                     metrics['position_precision'] > 0.85 and
                     metrics['grade_mae'] < 0.15 and
                     metrics['loss'] < 0.2)
        
        if target_met:
            print("\n🎉 模型达到目标指标!")
        else:
            print("\n⚠️ 模型未达到目标指标")
            print("目标指标:")
            print("  - 准确率 > 90%")
            print("  - F1分数 > 0.85")
            print("  - 召回率 > 0.85")
            print("  - 精确率 > 0.85")
            print("  - MAE < 0.15")
            print("  - 总损失 < 0.2")
        
        return metrics
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        return None

def visualize_predictions(model, data_loader, device, num_samples=4, save_dir=None):
    """
    可视化模型预测结果
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        num_samples: 可视化样本数量
        save_dir: 保存目录
    """
    print(f"\n====== 可视化预测结果 ======")
    
    model.eval()
    samples_shown = 0
    
    # 位置类别名称
    position_names = ['下部', '中部', '上部']
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for batch_idx, (images, position_labels, grade_labels) in enumerate(data_loader):
            if samples_shown >= num_samples:
                break
                
            images = images.to(device)
            position_labels = position_labels.view(-1).long().to(device)
            grade_labels = grade_labels.float().to(device)
            
            # 模型预测
            seg_output, pos_output, grade_output = model(images)
            
            # 获取预测结果
            seg_pred = torch.softmax(seg_output, dim=1)
            pos_pred = torch.softmax(pos_output, dim=1)
            _, pos_pred_class = torch.max(pos_output, 1)
            
            # 可视化每个样本
            for i in range(min(images.size(0), num_samples - samples_shown)):
                row = samples_shown
                
                # 原始图像
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())  # 归一化到0-1
                axes[row, 0].imshow(img)
                axes[row, 0].set_title('原始图像')
                axes[row, 0].axis('off')
                
                # 分割预测（感染区域）
                seg_mask = seg_pred[i, 1].cpu().numpy()  # 感染区域概率
                axes[row, 1].imshow(seg_mask, cmap='hot')
                axes[row, 1].set_title('感染区域预测')
                axes[row, 1].axis('off')
                
                # 位置分类结果
                true_pos = position_labels[i].cpu().item()
                pred_pos = pos_pred_class[i].cpu().item()
                pos_conf = pos_pred[i, pred_pos].cpu().item()
                
                axes[row, 2].bar(range(3), pos_pred[i].cpu().numpy())
                axes[row, 2].set_title(f'位置分类\n真实: {position_names[true_pos]}\n预测: {position_names[pred_pos]} ({pos_conf:.2f})')
                axes[row, 2].set_xticks(range(3))
                axes[row, 2].set_xticklabels(position_names, rotation=45)
                
                # 病害等级回归结果
                true_grade = grade_labels[i].cpu().item()
                pred_grade = grade_output[i, 0].cpu().item()
                mae = abs(true_grade - pred_grade)
                
                axes[row, 3].bar(['真实', '预测'], [true_grade, pred_grade], color=['blue', 'orange'])
                axes[row, 3].set_title(f'病害等级\n真实: {true_grade:.1f}\n预测: {pred_grade:.1f}\nMAE: {mae:.2f}')
                axes[row, 3].set_ylim(0, 9)
                
                samples_shown += 1
                if samples_shown >= num_samples:
                    break
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'lodgenet_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果已保存到: {save_path}")
    
    plt.show()

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='LodgeNet测试脚本')
    
    parser.add_argument('--test_architecture', action='store_true',
                        help='测试模型架构')
    parser.add_argument('--test_loss', action='store_true',
                        help='测试损失函数')
    parser.add_argument('--model_path', type=str, default=None,
                        help='预训练模型路径')
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='数据根目录')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注根目录')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化预测结果')
    parser.add_argument('--save_dir', type=str, default='./test_results',
                        help='结果保存目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试模型架构
    if args.test_architecture:
        model = test_model_architecture()
    
    # 测试损失函数
    if args.test_loss:
        test_loss_functions()
    
    # 测试预训练模型
    if args.model_path:
        metrics = load_and_test_model(args.model_path, args.data_root, args.json_root, device)
        
        # 可视化预测结果
        if args.visualize and metrics:
            try:
                # 加载模型
                checkpoint = torch.load(args.model_path, map_location=device)
                model = get_lodgenet_model(n_channels=3, n_classes=2, img_size=128).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # 加载数据
                _, val_loader = get_dataloaders(
                    data_root=args.data_root,
                    json_root=args.json_root,
                    batch_size=4,
                    num_workers=0,
                    img_size=128,
                    train_ratio=0.8,
                    use_extended_dataset=True
                )
                
                # 可视化
                visualize_predictions(model, val_loader, device, num_samples=4, save_dir=args.save_dir)
                
            except Exception as e:
                print(f"可视化过程中出错: {e}")
    
    # 如果没有指定任何测试，默认测试架构
    if not any([args.test_architecture, args.test_loss, args.model_path]):
        print("未指定测试内容，默认测试模型架构...")
        test_model_architecture()
        test_loss_functions()
    
    print("\n====== 测试完成 ======")

if __name__ == "__main__":
    main() 