#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 核心文件：最终版本的训练启动脚本，优化配置实现最佳训练效果
# LodgeNet最终训练启动脚本：50轮完整训练，包含日志记录

import os
import sys
import subprocess
import argparse
from datetime import datetime

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LodgeNet最终训练启动器')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='TIF图像数据根目录')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注数据根目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载器工作进程数')
    parser.add_argument('--img_size', type=int, default=256,
                        help='输入图像尺寸')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志目录')
    
    args = parser.parse_args()
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 如果没有指定输出目录，自动生成
    if args.output_dir is None:
        args.output_dir = f'./output_lodgenet_final_{timestamp}'
    
    # 创建目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 生成日志文件路径
    log_file = os.path.join(args.log_dir, f'lodgenet_final_{timestamp}.log')
    
    print("=" * 80)
    print("🚀 LodgeNet最终训练启动器")
    print("=" * 80)
    print(f"训练配置:")
    print(f"  - 数据根目录: {args.data_root}")
    print(f"  - JSON根目录: {args.json_root}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 日志文件: {log_file}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 图像尺寸: {args.img_size}")
    print(f"  - 工作进程数: {args.num_workers}")
    print(f"  - 学习率: {args.learning_rate}")
    print(f"  - 训练轮数: {args.num_epochs}")
    
    # 构建训练命令
    cmd = [
        sys.executable, 'train_lodgenet_simple.py',
        '--data_root', args.data_root,
        '--json_root', args.json_root,
        '--batch_size', str(args.batch_size),
        '--num_epochs', str(args.num_epochs),
        '--learning_rate', str(args.learning_rate),
        '--num_workers', str(args.num_workers),
        '--img_size', str(args.img_size),
        '--output_dir', args.output_dir
    ]
    
    print(f"\n执行命令: {' '.join(cmd)}")
    
    # 确认开始训练
    response = input(f"\n🚀 准备开始50轮完整训练，继续？(y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ 训练已取消")
        return 0
    
    print(f"\n🚀 开始训练...")
    print(f"📝 日志将保存到: {log_file}")
    print("-" * 80)
    
    # 运行训练并记录日志
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            # 写入训练配置到日志
            f.write("=" * 80 + "\n")
            f.write("LodgeNet训练日志\n")
            f.write("=" * 80 + "\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write("=" * 80 + "\n\n")
            f.flush()
            
            # 启动训练进程
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
            
            # 写入结束信息
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"返回码: {return_code}\n")
            f.write("=" * 80 + "\n")
        
        print("\n" + "=" * 80)
        if return_code == 0:
            print("🎉 训练成功完成！")
        else:
            print(f"❌ 训练失败，返回码: {return_code}")
        
        print(f"📁 输出目录: {args.output_dir}")
        print(f"📝 训练日志: {log_file}")
        print("=" * 80)
        
        return return_code
        
    except Exception as e:
        print(f"❌ 训练执行出错: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 