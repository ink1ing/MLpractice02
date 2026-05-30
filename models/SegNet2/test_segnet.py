#!/usr/bin/env python
# 核心文件，完整运行项目不可缺少
# SegNet模型测试脚本
# 使用示例: python test_segnet.py --data_root ./guanceng-bit --json_root ./biaozhu_json --model_path ./output_segnet/best_model.pth

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import seaborn as sns
from torch.amp import autocast
import time
import json
from scipy.ndimage import binary_dilation
from skimage.metrics import structural_similarity as ssim

from model import get_model
from dataset import CornRustDataset, get_dataloaders
from utils import FocalLoss
from train import HybridLoss

def visualize_segmentation(images, segmentations, targets, position_preds, position_labels, 
                           grade_preds, grade_labels, save_dir, batch_idx, max_samples=8):
    """可视化分割结果和分类预测"""
    # 只显示最多max_samples个样本
    num_samples = min(images.size(0), max_samples)
    
    for i in range(num_samples):
        plt.figure(figsize=(15, 10))
        
        # 原始图像
        plt.subplot(2, 3, 1)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        # 归一化到0-1范围以便显示
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imshow(img)
        plt.title("原始图像")
        plt.axis("off")
        
        # 分割预测结果
        plt.subplot(2, 3, 2)
        seg_pred = torch.sigmoid(segmentations[i, 0]).cpu().detach().numpy()
        plt.imshow(seg_pred, cmap='jet')
        plt.title("分割预测")
        plt.colorbar()
        plt.axis("off")
        
        # 分割目标（粗略标签）
        plt.subplot(2, 3, 3)
        seg_target = targets[i, 0].cpu().numpy()
        plt.imshow(seg_target, cmap='jet')
        plt.title("分割标签")
        plt.colorbar()
        plt.axis("off")
        
        # 分割叠加在原图上
        plt.subplot(2, 3, 4)
        seg_overlay = img.copy()
        seg_pred_binary = (seg_pred > 0.5).astype(np.float32)
        # 叠加红色通道
        seg_overlay[:, :, 0] = np.maximum(seg_overlay[:, :, 0], seg_pred_binary * 0.7)
        plt.imshow(seg_overlay)
        plt.title("分割叠加")
        plt.axis("off")
        
        # 文本信息：位置和等级预测
        plt.subplot(2, 3, 5)
        plt.axis("off")
        pos_classes = ["下部", "中部", "上部"]
        pos_pred = pos_classes[position_preds[i]]
        pos_true = pos_classes[position_labels[i]]
        grade_pred_val = grade_preds[i].item()
        grade_true_val = grade_labels[i].item()
        
        info_text = (f"位置预测: {pos_pred}\n"
                    f"真实位置: {pos_true}\n\n"
                    f"等级预测: {grade_pred_val:.2f}\n"
                    f"真实等级: {grade_true_val:.2f}\n\n"
                    f"位置正确: {'✓' if pos_pred == pos_true else '✗'}\n"
                    f"等级误差: {abs(grade_pred_val - grade_true_val):.2f}")
        
        plt.text(0.1, 0.5, info_text, fontsize=12)
        
        # 保存图像
        save_path = os.path.join(save_dir, f"sample_batch{batch_idx}_idx{i}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def calculate_rust_metrics(seg_preds, seg_targets, position_preds, position_labels, grade_preds, grade_labels):
    """
    计算锈病特定的评估指标
    
    参数:
        seg_preds: 分割预测 [batch, 1, H, W]
        seg_targets: 分割目标 [batch, 1, H, W]
        position_preds: 位置预测 [batch]
        position_labels: 位置标签 [batch]
        grade_preds: 等级预测 [batch, 1]
        grade_labels: 等级标签 [batch, 1]
        
    返回:
        dict: 包含锈病特定指标的字典
    """
    # 转为numpy数组
    seg_preds_np = (seg_preds > 0.5).float().cpu().numpy()
    seg_targets_np = seg_targets.cpu().numpy()
    
    # 病斑检测指标
    batch_size = seg_preds.size(0)
    lesion_metrics = {
        'lesion_detection_rate': 0.0,  # 病斑检出率
        'lesion_false_positive': 0.0,  # 假阳性率
        'lesion_area_error': 0.0,      # 病斑面积误差
        'boundary_iou': 0.0,           # 边界IoU
        'ssim': 0.0                    # 结构相似性
    }
    
    for i in range(batch_size):
        pred = seg_preds_np[i, 0]
        target = seg_targets_np[i, 0]
        
        # 病斑检出率：检测到的真实病斑/总真实病斑
        if np.sum(target) > 0:
            # 每个连通区域作为一个病斑
            from scipy import ndimage
            target_labels, target_count = ndimage.label(target)
            pred_labels, pred_count = ndimage.label(pred)
            
            detected = 0
            for j in range(1, target_count + 1):
                target_region = (target_labels == j)
                # 如果预测与该病斑有重叠，算作检出
                if np.sum(pred * target_region) > 0:
                    detected += 1
            
            lesion_metrics['lesion_detection_rate'] += detected / max(1, target_count)
            
            # 假阳性率：错误预测的病斑/总预测病斑
            false_positive = 0
            for j in range(1, pred_count + 1):
                pred_region = (pred_labels == j)
                if np.sum(target * pred_region) == 0:
                    false_positive += 1
            
            lesion_metrics['lesion_false_positive'] += false_positive / max(1, pred_count)
            
            # 病斑面积误差：|预测面积-真实面积|/真实面积
            area_error = abs(np.sum(pred) - np.sum(target)) / max(1, np.sum(target))
            lesion_metrics['lesion_area_error'] += area_error
            
            # 边界IoU：考虑病斑边界
            # 使用膨胀操作获取边界
            target_dilated = binary_dilation(target)
            target_boundary = target_dilated ^ target  # XOR得到边界
            
            pred_dilated = binary_dilation(pred)
            pred_boundary = pred_dilated ^ pred
            
            # 计算边界IoU
            boundary_intersection = np.sum(target_boundary * pred_boundary)
            boundary_union = np.sum(target_boundary) + np.sum(pred_boundary) - boundary_intersection
            boundary_iou = boundary_intersection / max(1, boundary_union)
            lesion_metrics['boundary_iou'] += boundary_iou
            
            # 结构相似性指数
            lesion_metrics['ssim'] += ssim(pred, target, data_range=1.0)
    
    # 计算平均值
    for key in lesion_metrics:
        lesion_metrics[key] /= batch_size
    
    # 等级估计精度：预测等级与真实等级的误差<1视为正确
    grade_preds_np = grade_preds.cpu().numpy().flatten()
    grade_labels_np = grade_labels.cpu().numpy().flatten()
    grade_accuracy = np.mean(np.abs(grade_preds_np - grade_labels_np) < 1.0)
    
    # 组合位置和等级的联合准确率
    joint_correct = ((position_preds == position_labels).cpu().numpy() & 
                    (np.abs(grade_preds_np - grade_labels_np) < 1.0))
    joint_accuracy = np.mean(joint_correct)
    
    # 构建并返回完整指标
    rust_metrics = {
        'grade_accuracy_tol1': grade_accuracy,  # 容忍误差为1的等级准确率
        'joint_accuracy': joint_accuracy,       # 位置和等级联合准确率
        **lesion_metrics                        # 病斑检测指标
    }
    
    return rust_metrics

def test_model(model, test_loader, device, save_dir=None, use_hybrid_loss=False):
    """测试模型性能并可视化结果，添加锈病特定指标"""
    model.eval()
    
    # 初始化指标
    position_preds_all = []
    position_labels_all = []
    grade_preds_all = []
    grade_labels_all = []
    loss_sum = 0.0
    
    # 分类和回归损失
    if use_hybrid_loss:
        hybrid_loss = HybridLoss()
        position_criterion = hybrid_loss.position_loss
        grade_criterion = hybrid_loss.grade_loss
        seg_criterion = hybrid_loss.segmentation_loss
    else:
        position_criterion = nn.CrossEntropyLoss()
        grade_criterion = nn.MSELoss()
        seg_criterion = nn.BCEWithLogitsLoss()
    
    # 分割相关指标
    seg_loss_sum = 0.0
    seg_iou_sum = 0.0
    seg_dice_sum = 0.0
    
    # 锈病特定指标
    rust_metrics_sum = {
        'grade_accuracy_tol1': 0.0,
        'joint_accuracy': 0.0,
        'lesion_detection_rate': 0.0,
        'lesion_false_positive': 0.0,
        'lesion_area_error': 0.0,
        'boundary_iou': 0.0,
        'ssim': 0.0
    }
    
    # 计时
    start_time = time.time()
    
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    batch_count = 0
    with torch.no_grad():
        for batch_idx, (images, position_labels, grade_labels) in enumerate(test_loader):
            # 将数据移到设备上
            images = images.to(device)
            position_labels = position_labels.to(device).long()
            grade_labels = grade_labels.float().unsqueeze(1).to(device)
            
            # 使用混合精度计算
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                # 前向传播 - SegNet模型输出三部分
                position_logits, grade_values, segmentation = model(images)
                
                # 创建粗略分割标签
                seg_targets = (torch.sum(images, dim=1, keepdim=True) > 0.1).float()
                
                # 计算损失
                if use_hybrid_loss:
                    loss_position = position_criterion(position_logits, position_labels)
                    loss_grade = grade_criterion(grade_values, grade_labels)
                    loss_segmentation = seg_criterion(segmentation, seg_targets)
                else:
                    loss_position = position_criterion(position_logits, position_labels)
                    loss_grade = grade_criterion(grade_values, grade_labels)
                    loss_segmentation = seg_criterion(segmentation, seg_targets)
                
                # 总损失
                loss = 0.5 * loss_position + 0.3 * loss_grade + 0.2 * loss_segmentation
            
            # 获取预测
            _, position_preds = torch.max(position_logits, 1)
            
            # 累加损失
            batch_size = images.size(0)
            loss_sum += loss.item() * batch_size
            seg_loss_sum += loss_segmentation.item() * batch_size
            
            # 计算分割指标
            seg_probs = torch.sigmoid(segmentation)
            seg_preds = (seg_probs > 0.5).float()
            
            # IoU = 交集/并集
            intersection = (seg_preds * seg_targets).sum(dim=[1, 2, 3])
            union = seg_preds.sum(dim=[1, 2, 3]) + seg_targets.sum(dim=[1, 2, 3]) - intersection
            batch_iou = (intersection / (union + 1e-8)).mean().item()
            seg_iou_sum += batch_iou * batch_size
            
            # Dice = 2*交集/(A+B)
            dice = (2 * intersection / (seg_preds.sum(dim=[1, 2, 3]) + seg_targets.sum(dim=[1, 2, 3]) + 1e-8)).mean().item()
            seg_dice_sum += dice * batch_size
            
            # 计算锈病特定指标
            rust_batch_metrics = calculate_rust_metrics(
                seg_probs, seg_targets, position_preds, position_labels, grade_values, grade_labels
            )
            
            # 累加锈病指标
            for key in rust_metrics_sum:
                rust_metrics_sum[key] += rust_batch_metrics[key] * batch_size
            
            # 收集预测和标签
            position_preds_all.extend(position_preds.cpu().numpy())
            position_labels_all.extend(position_labels.cpu().numpy())
            grade_preds_all.extend(grade_values.cpu().numpy())
            grade_labels_all.extend(grade_labels.cpu().numpy())
            
            # 可视化一些结果
            if save_dir and batch_idx < 3:  # 只保存前3个批次的可视化结果
                visualize_segmentation(
                    images, segmentation, seg_targets, 
                    position_preds, position_labels, 
                    grade_values, grade_labels, 
                    save_dir, batch_idx
                )
            
            batch_count += 1
    
    # 计算平均指标
    total_samples = len(test_loader.dataset)
    avg_loss = loss_sum / total_samples
    avg_seg_loss = seg_loss_sum / total_samples
    avg_iou = seg_iou_sum / total_samples
    avg_dice = seg_dice_sum / total_samples
    
    # 计算位置分类指标
    position_accuracy = accuracy_score(position_labels_all, position_preds_all)
    position_f1 = f1_score(position_labels_all, position_preds_all, average='macro')
    position_f1_per_class = f1_score(position_labels_all, position_preds_all, average=None)
    position_cm = confusion_matrix(position_labels_all, position_preds_all)
    
    # 计算等级回归指标
    grade_preds_all = np.array(grade_preds_all).flatten()
    grade_labels_all = np.array(grade_labels_all).flatten()
    grade_mae = np.mean(np.abs(grade_preds_all - grade_labels_all))
    
    # 计算平均锈病指标
    rust_metrics_avg = {}
    for key in rust_metrics_sum:
        rust_metrics_avg[key] = rust_metrics_sum[key] / total_samples
    
    # 计算执行时间
    test_time = time.time() - start_time
    
    # 绘制混淆矩阵
    if save_dir:
        plt.figure(figsize=(10, 8))
        sns.heatmap(position_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['下部', '中部', '上部'],
                   yticklabels=['下部', '中部', '上部'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'位置分类混淆矩阵 (准确率: {position_accuracy:.4f})')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
    
    # 打印结果
    print("\n===== 测试结果 =====")
    print(f"总样本数: {total_samples}")
    print(f"总测试时间: {test_time:.2f} 秒 (每样本 {test_time/total_samples*1000:.1f} 毫秒)")
    
    print(f"\n位置分类指标:")
    print(f"  准确率: {position_accuracy:.4f}")
    print(f"  F1分数: {position_f1:.4f}")
    print(f"  各类F1: {[f'{f:.4f}' for f in position_f1_per_class]}")
    
    print(f"\n等级预测指标:")
    print(f"  MAE: {grade_mae:.4f}")
    print(f"  容错准确率(±1): {rust_metrics_avg['grade_accuracy_tol1']:.4f}")
    
    print(f"\n分割指标:")
    print(f"  损失: {avg_seg_loss:.4f}")
    print(f"  IoU: {avg_iou:.4f}")
    print(f"  Dice系数: {avg_dice:.4f}")
    
    print(f"\n锈病特定指标:")
    print(f"  病斑检出率: {rust_metrics_avg['lesion_detection_rate']:.4f}")
    print(f"  假阳性率: {rust_metrics_avg['lesion_false_positive']:.4f}")
    print(f"  病斑面积误差: {rust_metrics_avg['lesion_area_error']:.4f}")
    print(f"  边界IoU: {rust_metrics_avg['boundary_iou']:.4f}")
    print(f"  结构相似性: {rust_metrics_avg['ssim']:.4f}")
    print(f"  位置+等级联合准确率: {rust_metrics_avg['joint_accuracy']:.4f}")
    
    print(f"\n总损失: {avg_loss:.4f}")
    
    # 返回指标字典
    metrics = {
        'position_accuracy': position_accuracy,
        'position_f1': position_f1,
        'position_f1_per_class': position_f1_per_class.tolist(),
        'grade_mae': grade_mae,
        'seg_loss': avg_seg_loss,
        'seg_iou': avg_iou,
        'seg_dice': avg_dice,
        'total_loss': avg_loss,
        'test_time': test_time,
        'rust_metrics': rust_metrics_avg
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='SegNet模型测试脚本')
    
    # 必要参数
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录路径')
    parser.add_argument('--json_root', type=str, required=True, 
                        help='JSON标注根目录路径')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重文件路径')
    
    # 可选参数
    parser.add_argument('--model_type', type=str, default='segnet+',
                        choices=['segnet', 'segnet+'],
                        help='模型类型')
    parser.add_argument('--img_size', type=int, default=128,
                        help='图像大小')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='输入通道数')
    parser.add_argument('--output_dir', type=str, default='./segnet_test_results',
                        help='结果输出目录')
    parser.add_argument('--no_cuda', action='store_true',
                        help='不使用CUDA')
    parser.add_argument('--use_hybrid_loss', action='store_true',
                        help='使用混合损失函数')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available() and not args.no_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载测试数据
    _, test_loader = get_dataloaders(
        data_root=args.data_root,
        json_root=args.json_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        train_ratio=0.8,  # 使用相同的分割比例，但只返回测试集
        use_extended_dataset=True,
        pin_memory=torch.cuda.is_available()
    )
    
    # 创建模型
    model = get_model(model_type=args.model_type, in_channels=args.in_channels, img_size=args.img_size)
    model = model.to(device)
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"成功加载模型权重: {args.model_path}")
    except Exception as e:
        print(f"加载模型权重出错: {e}")
        try:
            # 尝试加载完整检查点
            checkpoint = torch.load(args.model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("从检查点加载模型状态")
            else:
                print("检查点中不包含模型状态字典")
                return
        except:
            print("无法加载模型，请检查路径和模型类型是否匹配")
            return
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 测试模型
    metrics = test_model(model, test_loader, device, args.output_dir, args.use_hybrid_loss)
    
    # 保存指标到文件
    if args.output_dir:
        # 将numpy数组转换为列表以便JSON序列化
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                metrics[k] = v.tolist()
                
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        
        # 生成可读的摘要报告
        with open(os.path.join(args.output_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
            f.write("==== 玉米南方锈病SegNet模型测试报告 ====\n\n")
            f.write(f"模型类型: {args.model_type}\n")
            f.write(f"模型路径: {args.model_path}\n")
            f.write(f"测试样本数: {len(test_loader.dataset)}\n")
            f.write(f"测试时间: {metrics['test_time']:.2f}秒\n\n")
            
            f.write("=== 位置分类性能 ===\n")
            f.write(f"准确率: {metrics['position_accuracy']:.4f}\n")
            f.write(f"F1分数: {metrics['position_f1']:.4f}\n")
            f.write(f"各类F1: {[f'{f:.4f}' for f in metrics['position_f1_per_class']]}\n\n")
            
            f.write("=== 等级预测性能 ===\n")
            f.write(f"平均绝对误差: {metrics['grade_mae']:.4f}\n")
            f.write(f"容错准确率(±1): {metrics['rust_metrics']['grade_accuracy_tol1']:.4f}\n\n")
            
            f.write("=== 分割性能 ===\n")
            f.write(f"IoU: {metrics['seg_iou']:.4f}\n")
            f.write(f"Dice系数: {metrics['seg_dice']:.4f}\n\n")
            
            f.write("=== 锈病特定指标 ===\n")
            for key, value in metrics['rust_metrics'].items():
                f.write(f"{key}: {value:.4f}\n")
            
            f.write("\n总评: ")
            # 简单评估总体性能
            avg_score = (metrics['position_accuracy'] + 
                         metrics['rust_metrics']['grade_accuracy_tol1'] + 
                         metrics['seg_iou'] + 
                         metrics['rust_metrics']['joint_accuracy']) / 4
            
            if avg_score > 0.85:
                f.write("优秀 - 模型在各方面表现均衡且出色\n")
            elif avg_score > 0.75:
                f.write("良好 - 模型表现良好，可用于实际应用\n")
            elif avg_score > 0.65:
                f.write("一般 - 模型表现尚可，但仍有改进空间\n")
            else:
                f.write("需改进 - 模型表现不佳，需要进一步优化\n")
        
        print(f"测试结果已保存到 {args.output_dir}")
        print(f"- metrics.json: 详细指标数据")
        print(f"- summary_report.txt: 可读性摘要报告")

if __name__ == "__main__":
    main() 