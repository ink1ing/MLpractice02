#!/usr/bin/env python
# -*- coding: utf-8 -*-
# LodgeNet优化训练启动脚本：充分利用RTX6000 22.5GB GPU性能
# 支持自动批次大小调整、内存监控和性能优化

import os
import sys
import subprocess
import argparse
import torch
import psutil
from datetime import datetime
import json
import time

def check_system_resources():
    """
    检查系统资源并返回推荐配置
    """
    print("=" * 60)
    print("系统资源检查")
    print("=" * 60)
    
    # GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU内存: {gpu_memory:.1f} GB")
        
        # 清空GPU缓存
        torch.cuda.empty_cache()
        
        # 根据GPU内存推荐批次大小
        if gpu_memory >= 20:  # RTX 6000级别
            recommended_batch_size = 64
            recommended_img_size = 256
        elif gpu_memory >= 10:
            recommended_batch_size = 32
            recommended_img_size = 224
        elif gpu_memory >= 6:
            recommended_batch_size = 16
            recommended_img_size = 128
        else:
            recommended_batch_size = 8
            recommended_img_size = 128
    else:
        print("警告: 未检测到CUDA GPU，将使用CPU训练")
        recommended_batch_size = 4
        recommended_img_size = 128
    
    # 系统内存
    memory = psutil.virtual_memory()
    system_memory_gb = memory.total / 1024**3
    available_memory_gb = memory.available / 1024**3
    
    print(f"系统内存: {system_memory_gb:.1f} GB (总计), {available_memory_gb:.1f} GB (可用)")
    
    # 根据系统内存推荐工作进程数
    if system_memory_gb >= 128:  # 大内存系统
        recommended_workers = 16
    elif system_memory_gb >= 64:
        recommended_workers = 12
    elif system_memory_gb >= 32:
        recommended_workers = 8
    else:
        recommended_workers = 4
    
    # CPU信息
    cpu_count = psutil.cpu_count()
    print(f"CPU核心数: {cpu_count}")
    
    print("=" * 60)
    
    return {
        'batch_size': recommended_batch_size,
        'img_size': recommended_img_size,
        'num_workers': min(recommended_workers, cpu_count),
        'gpu_available': torch.cuda.is_available(),
        'gpu_memory': gpu_memory if torch.cuda.is_available() else 0
    }

def auto_tune_batch_size(base_batch_size, img_size, max_memory_usage=0.85):
    """
    自动调整批次大小以最大化GPU利用率
    """
    if not torch.cuda.is_available():
        return base_batch_size
    
    print(f"\n自动调整批次大小...")
    print(f"基础批次大小: {base_batch_size}")
    
    # 导入模型进行测试
    try:
        from lodgenet_model import get_lodgenet_model
        
        device = torch.device('cuda')
        model = get_lodgenet_model(n_channels=3, n_classes=2, img_size=img_size).to(device)
        model.train()
        
        # 测试不同批次大小
        test_batch_sizes = [base_batch_size * i for i in [1, 2, 3, 4]]
        optimal_batch_size = base_batch_size
        
        for test_batch_size in test_batch_sizes:
            try:
                torch.cuda.empty_cache()
                
                # 创建测试数据
                test_input = torch.randn(test_batch_size, 3, img_size, img_size).to(device)
                
                # 前向传播测试
                with torch.no_grad():
                    seg_out, pos_out, grade_out = model(test_input)
                
                # 检查内存使用
                memory_used = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
                
                print(f"批次大小 {test_batch_size}: 内存使用 {memory_used:.2%}")
                
                if memory_used < max_memory_usage:
                    optimal_batch_size = test_batch_size
                else:
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"批次大小 {test_batch_size}: 内存不足")
                    break
                else:
                    raise e
        
        del model
        torch.cuda.empty_cache()
        
        print(f"推荐批次大小: {optimal_batch_size}")
        return optimal_batch_size
        
    except Exception as e:
        print(f"自动调整失败: {e}")
        return base_batch_size

def monitor_training_resources(log_file):
    """
    监控训练过程中的资源使用情况
    """
    def log_resources():
        while True:
            try:
                # GPU监控
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
                    gpu_memory_cached = torch.cuda.memory_reserved(0) / 1024**3
                    gpu_utilization = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 0
                else:
                    gpu_memory_used = 0
                    gpu_memory_cached = 0
                    gpu_utilization = 0
                
                # CPU和内存监控
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # 记录到日志
                timestamp = datetime.now().strftime("%H:%M:%S")
                resource_info = (f"[{timestamp}] GPU: {gpu_memory_used:.1f}GB/{gpu_memory_cached:.1f}GB "
                               f"({gpu_utilization}%), CPU: {cpu_percent:.1f}%, RAM: {memory_percent:.1f}%")
                
                with open(log_file.replace('.log', '_resources.log'), 'a', encoding='utf-8') as f:
                    f.write(resource_info + '\n')
                
                time.sleep(30)  # 每30秒记录一次
                
            except Exception:
                break
    
    import threading
    monitor_thread = threading.Thread(target=log_resources, daemon=True)
    monitor_thread.start()

def create_optimized_training_command(args, recommendations):
    """
    创建优化的训练命令
    """
    # 使用推荐配置或用户指定配置
    batch_size = args.batch_size if args.batch_size > 0 else recommendations['batch_size']
    img_size = args.img_size if args.img_size > 0 else recommendations['img_size']
    num_workers = args.num_workers if args.num_workers >= 0 else recommendations['num_workers']
    
    # 自动调整批次大小
    if args.auto_tune_batch_size:
        batch_size = auto_tune_batch_size(batch_size, img_size)
    
    cmd = [
        sys.executable, 'lodgenet_train.py',
        '--data_root', args.data_root,
        '--json_root', args.json_root,
        '--batch_size', str(batch_size),
        '--num_epochs', str(args.num_epochs),
        '--learning_rate', str(args.learning_rate),
        '--num_workers', str(num_workers),
        '--img_size', str(img_size),
        '--output_dir', args.output_dir
    ]
    
    return cmd, {
        'batch_size': batch_size,
        'img_size': img_size,
        'num_workers': num_workers
    }

def run_training_with_monitoring(cmd, log_file, config):
    """
    运行训练并监控资源使用
    """
    print(f"开始训练，日志将保存到: {log_file}")
    print(f"执行命令: {' '.join(cmd)}")
    print(f"优化配置: 批次大小={config['batch_size']}, 图像尺寸={config['img_size']}, 工作进程={config['num_workers']}")
    print("-" * 80)
    
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 启动资源监控
    monitor_training_resources(log_file)
    
    try:
        # 运行训练
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
            print(f"\n训练成功完成！")
            print(f"训练日志: {log_file}")
            print(f"资源监控日志: {log_file.replace('.log', '_resources.log')}")
        else:
            print(f"\n训练失败，返回码: {return_code}")
            
        return return_code
            
    except Exception as e:
        print(f"执行训练时出错: {e}")
        return 1

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='LodgeNet优化训练启动脚本')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='TIF图像数据根目录')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注数据根目录')
    
    # 训练参数 (0或负数表示使用自动推荐值)
    parser.add_argument('--batch_size', type=int, default=0,
                        help='批次大小 (0=自动推荐)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=-1,
                        help='数据加载器工作进程数 (-1=自动推荐)')
    
    # 模型参数
    parser.add_argument('--img_size', type=int, default=0,
                        help='输入图像尺寸 (0=自动推荐)')
    
    # 优化参数
    parser.add_argument('--auto_tune_batch_size', action='store_true',
                        help='自动调整批次大小以最大化GPU利用率')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（如果不指定，将自动生成）')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志目录')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LodgeNet优化训练启动器 - RTX6000性能优化版")
    print("=" * 80)
    
    # 检查系统资源
    recommendations = check_system_resources()
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 如果没有指定输出目录，自动生成
    if args.output_dir is None:
        args.output_dir = f'./output_lodgenet_optimized_{timestamp}'
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成日志文件路径
    log_file = os.path.join(args.log_dir, f'lodgenet_optimized_{timestamp}.log')
    
    # 创建优化的训练命令
    cmd, final_config = create_optimized_training_command(args, recommendations)
    
    print(f"\n最终训练配置:")
    print(f"  - 数据根目录: {args.data_root}")
    print(f"  - JSON根目录: {args.json_root}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 日志文件: {log_file}")
    print(f"  - 批次大小: {final_config['batch_size']}")
    print(f"  - 图像尺寸: {final_config['img_size']}")
    print(f"  - 工作进程数: {final_config['num_workers']}")
    print(f"  - 学习率: {args.learning_rate}")
    print(f"  - 训练轮数: {args.num_epochs}")
    
    # 保存训练配置
    config = {
        'timestamp': timestamp,
        'system_recommendations': recommendations,
        'final_config': final_config,
        'args': vars(args),
        'command': ' '.join(cmd)
    }
    
    config_file = os.path.join(args.output_dir, 'optimized_training_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"  - 配置文件: {config_file}")
    
    # 检查数据目录
    if not os.path.exists(args.data_root):
        print(f"\n错误: 数据目录不存在: {args.data_root}")
        return 1
    
    if not os.path.exists(args.json_root):
        print(f"\n错误: JSON目录不存在: {args.json_root}")
        return 1
    
    # 检查必要的文件
    required_files = ['lodgenet_train.py', 'lodgenet_model.py', 'dataset.py', 'utils.py']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"\n错误: 缺少必要文件: {', '.join(missing_files)}")
        return 1
    
    # 确认开始训练
    print(f"\n预计GPU内存使用: ~{final_config['batch_size'] * final_config['img_size']**2 * 3 * 4 / 1024**3:.1f} GB")
    if recommendations['gpu_available']:
        print(f"可用GPU内存: {recommendations['gpu_memory']:.1f} GB")
    
    response = input("\n是否开始优化训练？(y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("训练已取消。")
        return 0
    
    # 运行训练
    return_code = run_training_with_monitoring(cmd, log_file, final_config)
    
    print("\n" + "=" * 80)
    print("训练完成")
    print("=" * 80)
    print(f"输出目录: {args.output_dir}")
    print(f"训练日志: {log_file}")
    print(f"资源监控: {log_file.replace('.log', '_resources.log')}")
    print(f"配置文件: {config_file}")
    
    return return_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 