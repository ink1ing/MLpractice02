#!/usr/bin/env python
# 训练启动脚本：设置最优参数并启动训练

import os
import argparse
import subprocess
import torch

def main():
    """
    使用最佳参数配置启动玉米南方锈病模型训练
    支持GPU和CPU训练
    """
    parser = argparse.ArgumentParser(description='玉米南方锈病模型训练启动脚本')
    
    # 基本参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='数据根目录路径')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注根目录路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    parser.add_argument('--no_cuda', action='store_true',
                        help='不使用CUDA')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式，使用较小批次和轮数')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出模式')
    
    args = parser.parse_args()
    
    # 打印数据路径
    print("\n数据路径信息:")
    print(f"数据根目录: {os.path.abspath(args.data_root)}")
    print(f"JSON标注目录: {os.path.abspath(args.json_root)}")
    
    # 检查目录是否存在
    if not os.path.exists(args.data_root):
        print(f"警告: 数据根目录不存在: {args.data_root}")
    if not os.path.exists(args.json_root):
        print(f"警告: JSON标注目录不存在: {args.json_root}")
    
    # 设置训练参数 - 针对第二阶段优化的配置
    train_params = [
        'python', 'train.py',
        '--data_root', args.data_root,
        '--json_root', args.json_root,
        '--output_dir', args.output_dir,
        '--model_type', 'resnet_plus',    # 使用带注意力的增强ResNet
        '--img_size', '128',              # 图像大小
        '--in_channels', '3',             # 输入通道数改为3（我们选择3个有代表性的波段）
        '--loss_type', 'focal',           # 使用Focal Loss
        '--focal_gamma', '2.0',           # Focal Loss参数
        '--lr', '0.0001',                 # 进一步降低学习率以提高稳定性
        '--lr_scheduler', 'plateau',      # 使用ReduceLROnPlateau调度
        '--patience', '10',               # 早停耐心值
        '--optimizer', 'adam',            # 使用Adam优化器
        '--task_weights', '0.5,0.5',      # 任务权重均等
        '--aug_prob', '0.7',              # 数据增强概率
    ]
    
    # 如果有CUDA设备，启用混合精度训练
    if torch.cuda.is_available() and not args.no_cuda:
        train_params.append('--amp')      # 启用混合精度训练，适用于RTX GPU
    
    # 添加详细输出参数
    if args.verbose:
        train_params.append('--verbose')
    
    # 根据系统和调试状态调整参数
    if args.debug:
        # 调试模式使用较小配置便于快速验证
        train_params.extend([
            '--batch_size', '4',          # 减小批次大小以适应大型图像
            '--epochs', '3',
            '--num_workers', '2'
        ])
    else:
        # 生产模式使用最佳配置
        if args.no_cuda:
            # CPU模式
            train_params.extend([
                '--batch_size', '4',      # 更小批次，适合CPU和复杂图像处理
                '--epochs', '30',         # 减少训练轮数
                '--num_workers', '4',
                '--no_cuda'
            ])
        else:
            # GPU模式
            train_params.extend([
                '--batch_size', '8',      # 减小批次大小以适应多通道大型图像
                '--epochs', '50',         # 完整训练轮数
                '--num_workers', '4'      # 适度减少数据加载线程
            ])
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行训练脚本
    print("\n启动训练，使用以下参数:")
    print(' '.join(train_params))
    print("\n")
    
    subprocess.run(train_params)
    
    print("\n训练完成，结果已保存到:", args.output_dir)
    
if __name__ == "__main__":
    main() 