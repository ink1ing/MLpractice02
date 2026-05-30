#!/usr/bin/env python
# -*- coding: utf-8 -*-
# LodgeNet最终优化训练脚本：50轮完整训练，充分利用RTX6000性能

import os
import sys
import subprocess
import argparse
from datetime import datetime

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LodgeNet最终优化训练启动器')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='TIF图像数据根目录')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注数据根目录')
    
    # 训练参数 - 优化配置
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小 (RTX6000优化)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='学习率 (适配大批次)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载器工作进程数')
    parser.add_argument('--img_size', type=int, default=256,
                        help='输入图像尺寸 (更高分辨率)')
    
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
    
    print("=" * 100)
    print("LodgeNet最终优化训练启动器 - RTX6000专用配置")
    print("=" * 100)
    print(f"训练配置 (RTX6000优化):")
    print(f"  数据根目录: {args.data_root}")
    print(f"  JSON根目录: {args.json_root}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  日志文件: {log_file}")
    print(f"  批次大小: {args.batch_size} (4x优化)")
    print(f"  图像尺寸: {args.img_size}x{args.img_size} (4x像素)")
    print(f"  工作进程数: {args.num_workers}")
    print(f"  学习率: {args.learning_rate} (适配大批次)")
    print(f"  训练轮数: {args.num_epochs}")
    
    # 估算GPU内存使用
    estimated_memory = args.batch_size * 3 * args.img_size * args.img_size * 4 / (1024**3)  # 输入数据
    estimated_memory += estimated_memory * 3  # 模型参数和梯度
    print(f"  预估GPU内存使用: {estimated_memory:.1f}GB / 22.5GB")
    
    # 构建训练命令
    cmd = [
        sys.executable, 'train_lodgenet_optimized.py',
        '--data_root', args.data_root,
        '--json_root', args.json_root,
        '--batch_size', str(args.batch_size),
        '--num_epochs', str(args.num_epochs),
        '--learning_rate', str(args.learning_rate),
        '--num_workers', str(args.num_workers),
        '--img_size', str(args.img_size),
        '--output_dir', args.output_dir
    ]
    
    print(f"\n执行命令:")
    print(f"   {' '.join(cmd)}")
    
    # 性能预测
    samples_per_epoch = 727  # 训练集大小
    batches_per_epoch = samples_per_epoch // args.batch_size
    estimated_time_per_epoch = batches_per_epoch * 0.1  # 估算每批次0.1秒
    total_estimated_time = estimated_time_per_epoch * args.num_epochs / 60  # 分钟
    
    print(f"\n性能预测:")
    print(f"   每轮批次数: {batches_per_epoch}")
    print(f"   预估每轮时间: {estimated_time_per_epoch:.1f}秒")
    print(f"   预估总训练时间: {total_estimated_time:.1f}分钟")
    print(f"   预估样本吞吐量: {args.batch_size / 0.1:.0f} 样本/秒")
    
    # 确认开始训练
    print("\n" + "=" * 100)
    response = input(f"准备开始50轮RTX6000优化训练，预计耗时{total_estimated_time:.0f}分钟，继续？(y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("训练已取消")
        return 0
    
    print(f"\n开始训练...")
    print(f"日志将保存到: {log_file}")
    print(f"实时监控GPU使用情况...")
    print("=" * 100)
    
    # 运行训练并记录日志
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            # 写入训练配置到日志
            f.write("=" * 100 + "\n")
            f.write("LodgeNet最终优化训练日志 - RTX6000专用配置\n")
            f.write("=" * 100 + "\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write(f"批次大小: {args.batch_size}\n")
            f.write(f"图像尺寸: {args.img_size}x{args.img_size}\n")
            f.write(f"学习率: {args.learning_rate}\n")
            f.write(f"工作进程数: {args.num_workers}\n")
            f.write("=" * 100 + "\n\n")
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
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"返回码: {return_code}\n")
            f.write("=" * 100 + "\n")
        
        print("\n" + "=" * 100)
        if return_code == 0:
            print("训练成功完成！")
            print("\n训练结果:")
            
            # 尝试读取训练历史
            try:
                import json
                history_file = os.path.join(args.output_dir, 'training_history.json')
                if os.path.exists(history_file):
                    with open(history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                    
                    if history['val']:
                        final_metrics = history['val'][-1]
                        print(f"   最终验证损失: {final_metrics['loss']:.4f}")
                        print(f"   最终位置准确率: {final_metrics['position_accuracy']:.4f}")
                        print(f"   最终等级MAE: {final_metrics['grade_mae']:.4f}")
                        
                        # 检查是否达到目标指标
                        target_met = (final_metrics['position_accuracy'] > 0.90 and 
                                    final_metrics['grade_mae'] < 0.15 and
                                    final_metrics['loss'] < 0.2)
                        
                        if target_met:
                            print("已达到目标指标！")
                        else:
                            print("未完全达到目标指标，可能需要更多训练")
                            
            except Exception as e:
                print(f"   无法读取训练历史: {e}")
        else:
            print(f"训练失败，返回码: {return_code}")
        
        print(f"\n输出目录: {args.output_dir}")
        print(f"训练日志: {log_file}")
        print("=" * 100)
        
        return return_code
        
    except Exception as e:
        print(f"训练执行出错: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 