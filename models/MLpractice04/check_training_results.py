#!/usr/bin/env python
# 训练结果检查脚本：用于查看模型性能和训练指标

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def check_output_directories(base_dir='.'):
    """查找并列出包含训练结果的目录"""
    output_dirs = []
    
    for dir_name in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, dir_name)) and dir_name.startswith('output'):
            # 检查是否包含训练结果文件
            if any(f.endswith('.pth') for f in os.listdir(os.path.join(base_dir, dir_name))):
                output_dirs.append(dir_name)
    
    return output_dirs

def display_checkpoints(output_dir):
    """列出目录中的检查点文件"""
    if not os.path.exists(output_dir):
        print(f"目录不存在: {output_dir}")
        return []
    
    checkpoint_files = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
    
    print(f"\n在 {output_dir} 中找到 {len(checkpoint_files)} 个检查点文件:")
    for i, cp_file in enumerate(checkpoint_files):
        file_path = os.path.join(output_dir, cp_file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # 尝试获取训练轮次
        epoch = None
        if 'epoch' in cp_file:
            try:
                epoch = int(cp_file.split('epoch_')[1].split('.')[0])
            except:
                pass
        
        # 显示文件信息
        if epoch:
            print(f"  {i+1}. {cp_file} (轮次 {epoch}, 大小: {size_mb:.1f} MB)")
        else:
            print(f"  {i+1}. {cp_file} (大小: {size_mb:.1f} MB)")
    
    return checkpoint_files

def display_training_metrics(output_dir):
    """显示训练指标图像"""
    metrics_image = os.path.join(output_dir, 'training_metrics.png')
    
    if os.path.exists(metrics_image):
        print(f"\n找到训练指标图: {metrics_image}")
        try:
            img = Image.open(metrics_image)
            plt.figure(figsize=(12, 10))
            plt.imshow(np.asarray(img))
            plt.axis('off')
            plt.title("训练指标")
            plt.show()
        except Exception as e:
            print(f"无法显示训练指标图: {e}")
    else:
        print(f"\n未找到训练指标图: {metrics_image}")

def display_confusion_matrix(output_dir):
    """显示混淆矩阵图像"""
    cm_image = os.path.join(output_dir, 'position_confusion_matrix.png')
    
    if os.path.exists(cm_image):
        print(f"\n找到混淆矩阵图: {cm_image}")
        try:
            img = Image.open(cm_image)
            plt.figure(figsize=(8, 8))
            plt.imshow(np.asarray(img))
            plt.axis('off')
            plt.title("位置分类混淆矩阵")
            plt.show()
        except Exception as e:
            print(f"无法显示混淆矩阵图: {e}")
    else:
        print(f"\n未找到混淆矩阵图: {cm_image}")

def check_model_info(output_dir, model_file):
    """检查模型文件信息"""
    model_path = os.path.join(output_dir, model_file)
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    print(f"\n检查模型文件: {model_path}")
    try:
        # 加载模型
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 检查是否为完整检查点还是仅权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整检查点
            print(f"  类型: 完整检查点")
            print(f"  训练轮次: {checkpoint.get('epoch', 'N/A')}")
            
            # 尝试获取最佳性能
            if 'best_val_loss' in checkpoint:
                print(f"  最佳验证损失: {checkpoint['best_val_loss']:.4f}")
            if 'best_f1' in checkpoint:
                print(f"  最佳F1分数: {checkpoint['best_f1']:.4f}")
                
            # 输出优化器信息
            if 'optimizer_state_dict' in checkpoint:
                print(f"  优化器: {type(checkpoint['optimizer_state_dict']).__name__}")
                
            # 检查metrics_history
            if 'metrics_history' in checkpoint:
                metrics = checkpoint['metrics_history']
                if 'val' in metrics and metrics['val']:
                    last_metrics = metrics['val'][-1]
                    print(f"  最后一轮验证指标:")
                    print(f"    - 损失: {last_metrics.get('loss', 'N/A')}")
                    print(f"    - 位置准确率: {last_metrics.get('position_accuracy', 'N/A')}")
                    print(f"    - 位置F1: {last_metrics.get('position_f1', 'N/A')}")
                    print(f"    - 等级MAE: {last_metrics.get('grade_mae', 'N/A')}")
        else:
            # 仅权重
            print(f"  类型: 仅模型权重")
            
        # 输出模型大小
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  文件大小: {size_mb:.1f} MB")
    
    except Exception as e:
        print(f"  检查模型文件时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='训练结果检查工具')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='要检查的输出目录')
    parser.add_argument('--model_file', type=str, default='best_model.pth',
                        help='要检查的模型文件名')
    
    args = parser.parse_args()
    
    print("====== 训练结果检查 ======")
    
    # 查找输出目录
    if args.output_dir:
        # 使用指定的输出目录
        output_dir = args.output_dir
        print(f"检查指定输出目录: {output_dir}")
        
        # 检查检查点文件
        checkpoint_files = display_checkpoints(output_dir)
        
        # 检查训练指标
        display_training_metrics(output_dir)
        
        # 检查混淆矩阵
        display_confusion_matrix(output_dir)
        
        # 检查模型信息
        if args.model_file in checkpoint_files:
            check_model_info(output_dir, args.model_file)
        elif checkpoint_files:
            check_model_info(output_dir, checkpoint_files[0])
    else:
        # 自动查找输出目录
        output_dirs = check_output_directories()
        
        if not output_dirs:
            print("未找到任何输出目录。")
        else:
            print(f"找到 {len(output_dirs)} 个输出目录:")
            for i, dir_name in enumerate(output_dirs):
                print(f"  {i+1}. {dir_name}")
            
            # 用户选择要检查的目录
            choice = input("\n请输入要检查的目录编号，或按回车检查第一个: ")
            if choice.strip() == '':
                choice = 1
            else:
                try:
                    choice = int(choice)
                except:
                    choice = 1
            
            # 验证选择
            if 1 <= choice <= len(output_dirs):
                output_dir = output_dirs[choice-1]
                
                # 检查检查点文件
                checkpoint_files = display_checkpoints(output_dir)
                
                # 检查训练指标
                display_training_metrics(output_dir)
                
                # 检查混淆矩阵
                display_confusion_matrix(output_dir)
                
                # 检查模型信息
                if args.model_file in checkpoint_files:
                    check_model_info(output_dir, args.model_file)
                elif checkpoint_files:
                    check_model_info(output_dir, checkpoint_files[0])
            else:
                print(f"无效的选择: {choice}")
    
    print("\n====== 检查完成 ======")

if __name__ == "__main__":
    main() 