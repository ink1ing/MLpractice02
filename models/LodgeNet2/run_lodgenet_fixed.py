#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 核心文件：修复版本的训练启动脚本，调用修复后的数据加载器和训练脚本

import os
import sys
import subprocess
import argparse
from datetime import datetime
import json

def create_training_command(args):
    """创建训练命令"""
    cmd = [
        sys.executable, 'lodgenet_train_fixed.py',
        '--data_root', args.data_root,
        '--json_root', args.json_root,
        '--batch_size', str(args.batch_size),
        '--num_epochs', str(args.num_epochs),
        '--learning_rate', str(args.learning_rate),
        '--num_workers', str(args.num_workers),
        '--img_size', str(args.img_size),
        '--output_dir', args.output_dir
    ]
    
    return cmd

def run_with_logging(cmd, log_file):
    """运行命令并记录日志"""
    print(f"开始训练，日志将保存到: {log_file}")
    print(f"执行命令: {' '.join(cmd)}")
    print("-" * 80)
    
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    try:
        # 直接运行Python脚本并捕获输出
        with open(log_file, 'w', encoding='utf-8') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时输出并写入日志
            for line in process.stdout:
                print(line, end='')  # 输出到控制台
                f.write(line)  # 写入日志文件
                f.flush()  # 立即刷新到文件
            
            process.wait()
            return_code = process.returncode
            
        if return_code == 0:
            print(f"\n✅ 训练成功完成！日志已保存到: {log_file}")
        else:
            print(f"\n❌ 训练失败，返回码: {return_code}")
            
    except Exception as e:
        print(f"❌ 执行训练时出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LodgeNet训练启动脚本（修复版）')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='TIF图像数据根目录')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注数据根目录')
    
    # 训练参数 - 从低配置开始逐步增加
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载器工作进程数')
    
    # 模型参数
    parser.add_argument('--img_size', type=int, default=128,
                        help='输入图像尺寸')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（如果不指定，将自动生成）')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志目录')
    
    args = parser.parse_args()
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 如果没有指定输出目录，自动生成
    if args.output_dir is None:
        args.output_dir = f'./output_lodgenet_fixed_{timestamp}'
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成日志文件路径
    log_file = os.path.join(args.log_dir, f'lodgenet_training_fixed_{timestamp}.log')
    
    print("====== LodgeNet训练启动器（修复版）======")
    print(f"训练配置:")
    print(f"  - 数据根目录: {args.data_root}")
    print(f"  - JSON根目录: {args.json_root}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 日志文件: {log_file}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 学习率: {args.learning_rate}")
    print(f"  - 训练轮数: {args.num_epochs}")
    print(f"  - 图像尺寸: {args.img_size}")
    print(f"  - 工作进程数: {args.num_workers}")
    
    # 保存训练配置
    config = {
        'timestamp': timestamp,
        'data_root': args.data_root,
        'json_root': args.json_root,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'num_workers': args.num_workers,
        'img_size': args.img_size,
        'log_file': log_file
    }
    
    config_file = os.path.join(args.output_dir, 'training_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"  - 配置文件: {config_file}")
    print()
    
    # 检查数据目录
    if not os.path.exists(args.data_root):
        print(f"❌ 错误: 数据目录不存在: {args.data_root}")
        return
    
    if not os.path.exists(args.json_root):
        print(f"❌ 错误: JSON目录不存在: {args.json_root}")
        return
    
    # 检查是否已创建修复版训练脚本
    if not os.path.exists('lodgenet_train_fixed.py'):
        print("ℹ️ 需要创建修复版训练脚本，正在复制...")
        # 复制现有的训练脚本并修改
        with open('lodgenet_train.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换导入语句
        content = content.replace('from dataset import CornRustDataset, get_dataloaders', 
                                 'from dataset_fixed import CornRustDataset, get_dataloaders')
        
        # 保存修复版训练脚本
        with open('lodgenet_train_fixed.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 修复版训练脚本已创建: lodgenet_train_fixed.py")
    
    # 创建训练命令
    cmd = create_training_command(args)
    
    # 确认开始训练
    response = input("是否开始训练？(y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("训练已取消。")
        return
    
    # 运行训练并记录日志
    run_with_logging(cmd, log_file)
    
    print("\n====== 训练完成 ======")
    print(f"输出目录: {args.output_dir}")
    print(f"日志文件: {log_file}")
    print(f"配置文件: {config_file}")

if __name__ == "__main__":
    main() 