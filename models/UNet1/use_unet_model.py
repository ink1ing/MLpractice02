#!/usr/bin/env python
# -*- coding: utf-8 -*-
# [核心文件] 使用UNet模型推理脚本：演示如何加载训练好的UNet模型并用于预测

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import rasterio
from skimage.transform import resize as sk_resize
from model import get_model

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='UNet模型推理脚本 - 玉米南方锈病识别')
    
    # 数据和模型参数
    parser.add_argument('--image_path', type=str, required=True,
                        help='待预测图像路径(.tif文件)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重文件路径(.pth文件)')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='输出目录，用于保存可视化结果')
    
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
    parser.add_argument('--attention', action='store_true', default=True,
                        help='是否使用注意力机制')
    parser.add_argument('--bilinear', action='store_true', default=True,
                        help='是否使用双线性插值进行上采样')
    parser.add_argument('--features', type=str, default='64,128,256,512',
                        help='特征通道数，以逗号分隔')
    
    # 其他参数
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA，即使可用也使用CPU')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='分割结果二值化阈值')
    
    args = parser.parse_args()
    
    # 解析特征通道数
    args.features = [int(f) for f in args.features.split(',')]
    
    # 检查必要目录是否存在，不存在则创建
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def load_model(args, device):
    """加载UNet模型"""
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
    
    return model

def preprocess_image(image_path, img_size=128):
    """预处理TIF图像用于模型输入"""
    print(f"正在加载和预处理图像: {image_path}")
    
    try:
        # 读取.tif多光谱图像
        with rasterio.open(image_path) as src:
            # 选择代表性通道
            selected_bands = [0, 250, 499]  # 第一个、中间和最后一个波段
            selected_image = np.zeros((3, img_size, img_size), dtype=np.float32)
            
            # 确保有足够的波段
            if src.count < 3:
                # 如果通道数小于3，复制现有通道
                available_bands = min(src.count, 3)
                for i in range(3):
                    band_idx = i % available_bands
                    band_data = src.read(band_idx + 1)
                    # 调整大小
                    band_data = sk_resize(band_data, (img_size, img_size), 
                                          mode='constant', anti_aliasing=True)
                    selected_image[i] = band_data
            else:
                # 如果有足够的通道，选择代表性通道
                for i, band_idx in enumerate(selected_bands[:3]):
                    # 确保波段索引有效
                    band_idx = min(band_idx, src.count - 1)
                    band_data = src.read(band_idx + 1)
                    # 调整大小
                    band_data = sk_resize(band_data, (img_size, img_size), 
                                          mode='constant', anti_aliasing=True)
                    selected_image[i] = band_data
            
            # 标准化图像，保证像素值在[0, 1]范围内
            selected_image = np.clip(selected_image / 255.0, 0, 1)
            
            # 转换为PyTorch张量
            image_tensor = torch.from_numpy(selected_image).float()
            
            # 添加批次维度
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor
            
    except Exception as e:
        print(f"预处理图像时出错: {e}")
        raise

def run_inference(model, image_tensor, device, threshold=0.5):
    """使用模型进行推理"""
    print("正在进行模型推理...")
    
    with torch.no_grad():
        # 将图像张量移动到设备
        image_tensor = image_tensor.to(device)
        
        # 使用混合精度进行推理
        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            # 前向传播
            segmentation, position_logits, grade_values = model(image_tensor)
        
        # 处理分割结果
        segmentation = torch.sigmoid(segmentation)  # 应用sigmoid获取概率值
        segmentation_binary = (segmentation > threshold).float()  # 二值化
        
        # 处理位置分类结果
        _, position_pred = torch.max(position_logits, 1)
        
        # 将结果移回CPU
        segmentation = segmentation.cpu()
        segmentation_binary = segmentation_binary.cpu()
        position_pred = position_pred.cpu().item()
        grade_pred = grade_values.cpu().item()
        
        return {
            'segmentation': segmentation.numpy(),
            'segmentation_binary': segmentation_binary.numpy(),
            'position_pred': position_pred,
            'grade_pred': grade_pred
        }

def visualize_results(image_tensor, results, args):
    """可视化推理结果"""
    print("正在可视化结果...")
    
    # 位置类别名称
    position_names = ['下部', '中部', '上部']
    
    # 获取预测结果
    position_pred = results['position_pred']
    grade_pred = results['grade_pred']
    segmentation = results['segmentation'][0, 0]  # 取第一个样本的第一个通道
    segmentation_binary = results['segmentation_binary'][0, 0]
    
    # 将图像张量转换为NumPy数组
    image = image_tensor[0].cpu().numpy().transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
    
    # 创建图像
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 原始图像
    axes[0, 0].imshow(np.clip(image, 0, 1))
    axes[0, 0].set_title("原始图像")
    axes[0, 0].axis('off')
    
    # 分割概率图
    axes[0, 1].imshow(segmentation, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title("分割概率图")
    axes[0, 1].axis('off')
    
    # 二值化分割结果
    axes[1, 0].imshow(segmentation_binary, cmap='gray')
    axes[1, 0].set_title("二值化分割结果")
    axes[1, 0].axis('off')
    
    # 叠加显示
    axes[1, 1].imshow(np.clip(image, 0, 1))
    overlay = np.zeros_like(image)
    overlay[:, :, 0] = segmentation_binary * 1.0  # 红色通道表示分割区域
    axes[1, 1].imshow(overlay, alpha=0.5)
    axes[1, 1].set_title(f"预测结果\n位置: {position_names[position_pred]}, 等级: {grade_pred:.2f}")
    axes[1, 1].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(args.output_dir, 'inference_result.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"预测结果已保存到: {output_path}")
    print(f"预测位置: {position_names[position_pred]}")
    print(f"预测等级: {grade_pred:.2f}")
    
    # 保存详细结果到文本文件
    with open(os.path.join(args.output_dir, 'inference_result.txt'), 'w', encoding='utf-8') as f:
        f.write("UNet模型推理结果\n\n")
        f.write(f"图像路径: {args.image_path}\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"模型类型: {args.model_type}\n\n")
        f.write(f"预测位置: {position_names[position_pred]}\n")
        f.write(f"预测等级: {grade_pred:.2f}\n")
        f.write(f"分割结果: 病害区域占比 {np.mean(segmentation_binary):.2%}\n")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args, device)
    
    # 预处理图像
    image_tensor = preprocess_image(args.image_path, args.img_size)
    
    # 运行推理
    results = run_inference(model, image_tensor, device, args.threshold)
    
    # 可视化结果
    visualize_results(image_tensor, results, args)
    
    print("推理完成!")

if __name__ == "__main__":
    main() 