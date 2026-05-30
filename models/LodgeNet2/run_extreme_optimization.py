#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 旧版本文件：极端优化版训练启动脚本，尝试最大化性能
# LodgeNet极限性能优化脚本：榨干RTX6000 22.5GB GPU的每一分性能
# 目标：达到80%以上GPU内存使用率，最大化训练速度和精度

import os
import sys
import subprocess
import argparse
import torch
import psutil
from datetime import datetime
import json
import time

def extreme_performance_config():
    """
    极限性能配置：最大化利用RTX6000资源
    """
    print("=" * 80)
    print("🚀 RTX6000 极限性能配置")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到CUDA GPU")
        return None
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🎯 目标GPU: {gpu_name}")
    print(f"🎯 GPU内存: {gpu_memory:.1f} GB")
    
    # RTX6000专用极限配置
    if "RTX 6000" in gpu_name or "Quadro RTX 6000" in gpu_name:
        extreme_config = {
            'batch_size': 512,      # 极大批次大小
            'img_size': 384,        # 高分辨率图像
            'num_workers': 20,      # 最大工作进程
            'learning_rate': 0.002, # 更高学习率配合大批次
            'target_memory_usage': 0.90,  # 目标90%内存使用
            'gradient_accumulation': 2,    # 梯度累积
        }
        print("🔥 启用RTX6000极限配置")
    else:
        # 通用高性能配置
        extreme_config = {
            'batch_size': 256,
            'img_size': 256,
            'num_workers': 16,
            'learning_rate': 0.001,
            'target_memory_usage': 0.85,
            'gradient_accumulation': 1,
        }
        print("⚡ 启用通用高性能配置")
    
    # 系统资源检查
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    print(f"💾 系统内存: {memory.total / 1024**3:.1f} GB")
    print(f"🖥️  CPU核心: {cpu_count}")
    print(f"⚙️  推荐配置:")
    print(f"   - 批次大小: {extreme_config['batch_size']}")
    print(f"   - 图像尺寸: {extreme_config['img_size']}")
    print(f"   - 工作进程: {extreme_config['num_workers']}")
    print(f"   - 学习率: {extreme_config['learning_rate']}")
    print(f"   - 目标内存使用: {extreme_config['target_memory_usage']:.0%}")
    
    return extreme_config

def stress_test_gpu(config):
    """
    GPU压力测试：确保配置可行
    """
    print(f"\n🧪 GPU压力测试...")
    
    try:
        from lodgenet_model import get_lodgenet_model
        
        device = torch.device('cuda')
        model = get_lodgenet_model(
            n_channels=3, 
            n_classes=2, 
            img_size=config['img_size']
        ).to(device)
        
        # 启用混合精度
        scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        
        # 测试最大批次大小
        max_batch_size = config['batch_size']
        test_passed = False
        
        for test_batch in [max_batch_size, max_batch_size//2, max_batch_size//4]:
            try:
                torch.cuda.empty_cache()
                
                # 创建测试数据
                test_input = torch.randn(test_batch, 3, config['img_size'], config['img_size']).to(device)
                test_pos_labels = torch.randint(0, 3, (test_batch,)).to(device)
                test_grade_labels = torch.rand(test_batch, 1).to(device) * 9
                
                # 模拟训练步骤
                model.train()
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    seg_out, pos_out, grade_out = model(test_input)
                    
                    # 简单损失计算
                    pos_loss = torch.nn.functional.cross_entropy(pos_out, test_pos_labels)
                    grade_loss = torch.nn.functional.mse_loss(grade_out, test_grade_labels)
                    total_loss = pos_loss + grade_loss
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # 检查内存使用
                memory_used = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
                memory_cached = torch.cuda.memory_reserved(0) / torch.cuda.get_device_properties(0).total_memory
                
                print(f"✅ 批次大小 {test_batch}: 内存使用 {memory_used:.1%}, 缓存 {memory_cached:.1%}")
                
                if memory_used < config['target_memory_usage']:
                    config['batch_size'] = test_batch
                    test_passed = True
                    break
                else:
                    print(f"⚠️  批次大小 {test_batch}: 内存使用过高 ({memory_used:.1%})")
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ 批次大小 {test_batch}: GPU内存不足")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        del model, optimizer, scaler
        torch.cuda.empty_cache()
        
        if test_passed:
            print(f"🎉 压力测试通过！最终批次大小: {config['batch_size']}")
            return True
        else:
            print(f"❌ 压力测试失败，降级到安全配置")
            config['batch_size'] = 32
            config['img_size'] = 128
            return False
            
    except Exception as e:
        print(f"❌ 压力测试出错: {e}")
        return False

def create_extreme_training_command(config, args):
    """
    创建极限性能训练命令
    """
    cmd = [
        sys.executable, 'lodgenet_train.py',
        '--data_root', args.data_root,
        '--json_root', args.json_root,
        '--batch_size', str(config['batch_size']),
        '--num_epochs', str(args.num_epochs),
        '--learning_rate', str(config['learning_rate']),
        '--num_workers', str(config['num_workers']),
        '--img_size', str(config['img_size']),
        '--output_dir', args.output_dir
    ]
    
    return cmd

def monitor_extreme_performance(log_file):
    """
    极限性能监控
    """
    def performance_monitor():
        max_gpu_memory = 0
        max_gpu_utilization = 0
        
        while True:
            try:
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_memory_percent = gpu_memory_used / gpu_memory_total * 100
                    
                    max_gpu_memory = max(max_gpu_memory, gpu_memory_percent)
                    
                    # GPU利用率（如果可用）
                    try:
                        import nvidia_ml_py3 as nvml
                        nvml.nvmlInit()
                        handle = nvml.nvmlDeviceGetHandleByIndex(0)
                        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = utilization.gpu
                        max_gpu_utilization = max(max_gpu_utilization, gpu_util)
                    except:
                        gpu_util = 0
                else:
                    gpu_memory_percent = 0
                    gpu_util = 0
                
                # CPU和内存
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                # 记录性能数据
                timestamp = datetime.now().strftime("%H:%M:%S")
                perf_info = (f"[{timestamp}] 🚀 GPU: {gpu_memory_percent:.1f}% "
                           f"({gpu_memory_used:.1f}GB), GPU利用率: {gpu_util}%, "
                           f"CPU: {cpu_percent:.1f}%, RAM: {memory.percent:.1f}%")
                
                print(f"\r{perf_info}", end='', flush=True)
                
                with open(log_file.replace('.log', '_extreme_perf.log'), 'a', encoding='utf-8') as f:
                    f.write(perf_info + '\n')
                
                # 每10秒更新一次
                time.sleep(10)
                
            except Exception:
                break
    
    import threading
    monitor_thread = threading.Thread(target=performance_monitor, daemon=True)
    monitor_thread.start()

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='LodgeNet极限性能优化训练')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='TIF图像数据根目录')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注数据根目录')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志目录')
    
    # 极限优化选项
    parser.add_argument('--skip_stress_test', action='store_true',
                        help='跳过GPU压力测试')
    
    args = parser.parse_args()
    
    print("🔥" * 40)
    print("🚀 LodgeNet 极限性能优化训练")
    print("🎯 目标：榨干RTX6000的每一分性能")
    print("🔥" * 40)
    
    # 获取极限配置
    config = extreme_performance_config()
    if config is None:
        return 1
    
    # GPU压力测试
    if not args.skip_stress_test:
        if not stress_test_gpu(config):
            print("⚠️  使用降级配置继续训练")
    
    # 生成时间戳和目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f'./output_lodgenet_extreme_{timestamp}'
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'lodgenet_extreme_{timestamp}.log')
    
    # 创建训练命令
    cmd = create_extreme_training_command(config, args)
    
    print(f"\n🚀 极限训练配置:")
    print(f"   📊 批次大小: {config['batch_size']}")
    print(f"   🖼️  图像尺寸: {config['img_size']}x{config['img_size']}")
    print(f"   ⚡ 工作进程: {config['num_workers']}")
    print(f"   📈 学习率: {config['learning_rate']}")
    print(f"   💾 目标内存使用: {config['target_memory_usage']:.0%}")
    print(f"   📁 输出目录: {args.output_dir}")
    print(f"   📝 日志文件: {log_file}")
    
    # 预估性能
    estimated_memory = config['batch_size'] * config['img_size']**2 * 3 * 4 / 1024**3
    print(f"   🔮 预估GPU内存: ~{estimated_memory:.1f} GB")
    
    # 保存配置
    config_file = os.path.join(args.output_dir, 'extreme_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'config': config,
            'command': ' '.join(cmd),
            'args': vars(args)
        }, f, indent=2, ensure_ascii=False)
    
    # 确认启动
    response = input(f"\n🔥 准备启动极限性能训练！继续？(y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ 训练已取消")
        return 0
    
    print(f"\n🚀 启动极限训练...")
    print(f"📊 实时性能监控已启用")
    print("-" * 80)
    
    # 启动性能监控
    monitor_extreme_performance(log_file)
    
    # 运行训练
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
            return_code = process.returncode
        
        print(f"\n" + "🔥" * 40)
        if return_code == 0:
            print("🎉 极限训练成功完成！")
        else:
            print(f"❌ 训练失败，返回码: {return_code}")
        
        print(f"📁 输出目录: {args.output_dir}")
        print(f"📝 训练日志: {log_file}")
        print(f"📊 性能日志: {log_file.replace('.log', '_extreme_perf.log')}")
        print("🔥" * 40)
        
        return return_code
        
    except Exception as e:
        print(f"❌ 训练执行出错: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 