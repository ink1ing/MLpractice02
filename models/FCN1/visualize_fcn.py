#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCN模型分割结果可视化脚本
用于可视化FCN模型的分割预测结果，帮助理解模型的工作原理
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from skimage.transform import resize
from matplotlib.colors import ListedColormap

# 导入自定义模块
from dataset import CornRustDataset
from model import get_model

def load_model(model_path):
    """
    加载预训练FCN模型
    
    参数:
        model_path: 模型权重文件路径
        
    返回:
        model: 加载了权重的FCN模型
    """
    # 创建FCN模型实例
    model = get_model(model_type='fcn', in_channels=3, img_size=128)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估模式
    model.eval()
    
    return model

def get_segmentation_overlay(image, segmentation_map, alpha=0.5):
    """
    创建分割结果的叠加可视化
    
    参数:
        image: 原始图像，形状为[C, H, W]
        segmentation_map: 分割图，形状为[3, H, W]
        alpha: 透明度
        
    返回:
        overlay: 叠加后的可视化图像
    """
    # 确保图像为RGB格式
    if image.shape[0] == 1:
        # 如果是单通道，复制三次成RGB
        rgb_image = np.repeat(image, 3, axis=0)
    elif image.shape[0] == 3:
        rgb_image = image
    else:
        # 如果是多通道，选择三个代表性通道
        rgb_image = image[:3]
    
    # 转换为HWC格式，便于matplotlib显示
    rgb_image = np.transpose(rgb_image, (1, 2, 0))
    
    # 归一化到0-1范围，便于显示
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0
    
    # 获取分割结果的类别索引
    seg_idx = np.argmax(segmentation_map, axis=0)
    
    # 创建分割掩码的彩色表示
    # 使用易区分的颜色: 蓝色(下部)、绿色(中部)、红色(上部)
    colors = np.array([
        [0, 0, 1],    # 蓝色 - 下部
        [0, 1, 0],    # 绿色 - 中部
        [1, 0, 0]     # 红色 - 上部
    ])
    
    # 创建分割掩码的彩色图像
    seg_color = colors[seg_idx]
    
    # 创建叠加图像
    overlay = (1 - alpha) * rgb_image + alpha * seg_color
    
    return overlay

def visualize_batch(model, dataset, device, num_samples=4, save_dir=None):
    """
    可视化一批样本的分割结果
    
    参数:
        model: 预训练的FCN模型
        dataset: 数据集实例
        device: 计算设备
        num_samples: 要可视化的样本数量
        save_dir: 保存结果的目录，如果为None则只显示不保存
    """
    # 创建保存目录
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 定义位置标签映射
    position_names = ["下部", "中部", "上部"]
    
    # 获取样本并生成可视化
    samples_processed = 0
    
    with torch.no_grad():
        for image, position_label, grade_label in dataloader:
            # 如果达到指定样本数量，退出循环
            if samples_processed >= num_samples:
                break
                
            # 将图像数据移到设备
            image = image.to(device)
            
            # 获取FCN模型的分割预测
            segmentation_map = model.get_segmentation_map(image)
            position_logits, grade_value = model(image)
            
            # 获取位置分类预测
            _, position_pred = torch.max(position_logits, 1)
            
            # 将数据转移到CPU并转换为NumPy数组
            image_np = image.cpu().squeeze(0).numpy()
            segmentation_map_np = segmentation_map.cpu().squeeze(0).numpy()
            position_pred = position_pred.cpu().item()
            position_true = position_label.item()
            grade_pred = grade_value.cpu().item()
            grade_true = grade_label.item()
            
            # 创建分割叠加图
            overlay = get_segmentation_overlay(image_np, segmentation_map_np)
            
            # 创建可视化图像
            plt.figure(figsize=(15, 7))
            
            # 原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(np.transpose(image_np, (1, 2, 0)))
            plt.title("原始图像")
            plt.axis('off')
            
            # 分割结果
            plt.subplot(1, 3, 2)
            seg_idx = np.argmax(segmentation_map_np, axis=0)
            cmap = ListedColormap(['blue', 'green', 'red'])
            plt.imshow(seg_idx, cmap=cmap)
            plt.title("分割结果")
            plt.axis('off')
            
            # 分割叠加图
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title("分割叠加")
            plt.axis('off')
            
            # 添加整体标题
            plt.suptitle(f"位置: 真实={position_names[position_true]}, 预测={position_names[position_pred]}\n" +
                        f"等级: 真实={grade_true:.1f}, 预测={grade_pred:.1f}", fontsize=14)
                        
            plt.tight_layout()
            
            # 保存或显示
            if save_dir is not None:
                save_path = os.path.join(save_dir, f"sample_{samples_processed}.png")
                plt.savefig(save_path, dpi=150)
                plt.close()
            else:
                plt.show()
                
            samples_processed += 1

def visualize_attention_maps(model, dataset, device, num_samples=4, save_dir=None):
    """
    可视化FCN模型的注意力图
    
    参数:
        model: 预训练的FCN模型
        dataset: 数据集实例
        device: 计算设备
        num_samples: 要可视化的样本数量
        save_dir: 保存结果的目录，如果为None则只显示不保存
    """
    # 创建保存目录
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 获取样本并生成可视化
    samples_processed = 0
    
    class AttentionHook:
        def __init__(self):
            self.attention_map = None
            
        def __call__(self, module, input, output):
            self.attention_map = output.detach()
    
    # 注册钩子以获取注意力门控的输出
    attention_hook = AttentionHook()
    hook_handle = model.attention_gate.psi.register_forward_hook(attention_hook)
    
    with torch.no_grad():
        for image, position_label, grade_label in dataloader:
            # 如果达到指定样本数量，退出循环
            if samples_processed >= num_samples:
                break
                
            # 将图像数据移到设备
            image = image.to(device)
            
            # 获取FCN模型的预测
            features = model.backbone(image)
            attended_features = model.attention_gate(features, features)
            
            # 获取注意力图
            attention_map = attention_hook.attention_map.cpu().squeeze().numpy()
            
            # 将原始图像转换为NumPy数组
            image_np = image.cpu().squeeze(0).numpy()
            
            # 调整注意力图大小以匹配原始图像
            if attention_map.shape != image_np.shape[1:]:
                attention_map = resize(attention_map, image_np.shape[1:], anti_aliasing=True)
            
            # 创建可视化图像
            plt.figure(figsize=(12, 4))
            
            # 原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(np.transpose(image_np, (1, 2, 0)))
            plt.title("原始图像")
            plt.axis('off')
            
            # 注意力图
            plt.subplot(1, 3, 2)
            plt.imshow(attention_map, cmap='hot')
            plt.title("注意力图")
            plt.axis('off')
            
            # 注意力叠加图
            plt.subplot(1, 3, 3)
            rgb_image = np.transpose(image_np, (1, 2, 0))
            if rgb_image.max() > 1.0:
                rgb_image = rgb_image / 255.0
            
            attention_color = plt.cm.hot(attention_map)
            attention_overlay = (0.7 * rgb_image + 0.3 * attention_color[:, :, :3])
            
            plt.imshow(attention_overlay)
            plt.title("注意力叠加")
            plt.axis('off')
            
            plt.tight_layout()
            
            # 保存或显示
            if save_dir is not None:
                save_path = os.path.join(save_dir, f"attention_{samples_processed}.png")
                plt.savefig(save_path, dpi=150)
                plt.close()
            else:
                plt.show()
                
            samples_processed += 1
    
    # 移除钩子
    hook_handle.remove()

def main(args):
    """
    主函数
    
    参数:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args.model_path)
    model = model.to(device)
    
    # 创建数据集
    dataset = CornRustDataset(
        data_dir=args.data_root,
        json_dir=args.json_root,
        img_size=args.img_size,
        use_extended_dataset=True
    )
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        segmentation_dir = os.path.join(args.output_dir, 'segmentation')
        attention_dir = os.path.join(args.output_dir, 'attention')
    else:
        segmentation_dir = None
        attention_dir = None
    
    # 可视化分割结果
    print("生成分割结果可视化...")
    visualize_batch(model, dataset, device, args.num_samples, segmentation_dir)
    
    # 可视化注意力图
    print("生成注意力图可视化...")
    visualize_attention_maps(model, dataset, device, args.num_samples, attention_dir)
    
    print("可视化完成！")
    if args.output_dir:
        print(f"结果保存在: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FCN模型分割结果可视化')
    parser.add_argument('--data_root', type=str, required=True, help='多光谱图像数据目录')
    parser.add_argument('--json_root', type=str, default=None, help='标注JSON文件目录')
    parser.add_argument('--model_path', type=str, required=True, help='预训练FCN模型路径')
    parser.add_argument('--output_dir', type=str, default='./fcn_visualization', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=8, help='要可视化的样本数量')
    parser.add_argument('--img_size', type=int, default=128, help='图像大小')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU索引')
    
    args = parser.parse_args()
    
    main(args) 