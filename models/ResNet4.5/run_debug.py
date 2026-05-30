#!/usr/bin/env python
# 调试脚本：使用最小配置从头开始训练，帮助排查训练卡住问题

import os
import argparse
import subprocess
import torch
import sys

def main():
    """
    简化配置调试训练，帮助排查训练卡住的问题
    """
    parser = argparse.ArgumentParser(description='玉米南方锈病模型调试训练脚本')
    
    # 基本参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='数据根目录路径')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注根目录路径')
    parser.add_argument('--output_dir', type=str, default='./output_debug',
                        help='输出目录')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数，设为0使用主线程加载')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小，设置较小值以快速测试')
    
    args = parser.parse_args()
    
    # 打印系统和Python信息
    print(f"\n=== 系统信息 ===")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    print("\n=== 调试配置信息 ===")
    print(f"数据根目录: {os.path.abspath(args.data_root)}")
    print(f"JSON标注目录: {os.path.abspath(args.json_root)}")
    print(f"使用主线程加载数据 (num_workers=0)" if args.num_workers == 0 else f"使用 {args.num_workers} 个工作线程")
    print(f"批次大小: {args.batch_size}")
    
    # 检查目录
    print("\n=== 检查目录 ===")
    if not os.path.exists(args.data_root):
        print(f"错误: 数据根目录不存在: {args.data_root}")
        return
    if not os.path.exists(args.json_root):
        print(f"错误: JSON标注目录不存在: {args.json_root}")
        return
    
    print(f"数据根目录内容:")
    for item in os.listdir(args.data_root):
        if os.path.isdir(os.path.join(args.data_root, item)):
            print(f"  - {item}/ ({len(os.listdir(os.path.join(args.data_root, item)))} 个文件)")
        else:
            print(f"  - {item}")
    
    print(f"JSON标注目录内容:")
    for item in os.listdir(args.json_root):
        if os.path.isdir(os.path.join(args.json_root, item)):
            print(f"  - {item}/ ({len(os.listdir(os.path.join(args.json_root, item)))} 个文件)")
        else:
            print(f"  - {item}")
    
    # 设置训练参数 - 简化配置
    train_params = [
        'python', '-u', 'train.py',  # -u参数使Python输出不缓冲
        '--data_root', args.data_root,
        '--json_root', args.json_root,
        '--output_dir', args.output_dir,
        '--model_type', 'simple',    # 使用简单模型以加快训练
        '--img_size', '128',
        '--in_channels', '3',
        '--loss_type', 'ce',         # 使用标准交叉熵损失
        '--lr', '0.001',
        '--epochs', '2',             # 只训练2轮
        '--num_workers', str(args.num_workers),  # 使用主线程加载数据
        '--batch_size', str(args.batch_size),
    ]
    
    if torch.cuda.is_available():
        print("\n使用GPU训练，但禁用混合精度以简化调试")
    else:
        train_params.append('--no_cuda')
        print("\n使用CPU训练")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行训练脚本
    print("\n=== 开始调试训练 ===")
    print(f"运行命令: {' '.join(train_params)}")
    print("\n---------- 训练日志 ----------")
    
    # 运行时不缓冲输出
    process = subprocess.Popen(
        train_params,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # 实时打印输出
    for line in process.stdout:
        print(line, end='')
    
    # 等待进程结束
    process.wait()
    
    print("\n---------- 训练结束 ----------")
    print(f"退出代码: {process.returncode}")
    
    if process.returncode == 0:
        print("\n训练成功完成！")
    else:
        print("\n训练过程出错！")
    
    print(f"调试输出已保存到: {args.output_dir}")
    
if __name__ == "__main__":
    main() 