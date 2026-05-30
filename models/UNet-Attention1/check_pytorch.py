#!/usr/bin/env python
# PyTorch环境检查脚本：检查PyTorch、CUDA和系统信息，并进行简单的数据加载测试

import os
import sys
import torch
import platform
import psutil
import time
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

# 定义一个简单的测试数据集
class SimpleDataset(Dataset):
    def __init__(self, size=1000, dim=100):
        self.size = size
        self.dim = dim
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 生成随机数据
        x = torch.randn(self.dim)
        y = torch.randint(0, 3, (1,)).item()  # 随机类别
        return x, y

def print_system_info():
    print("=== 系统信息 ===")
    print(f"操作系统: {platform.platform()}")
    print(f"Python版本: {sys.version}")
    print(f"CPU核心数: {psutil.cpu_count(logical=False)} 物理核心, {psutil.cpu_count()} 逻辑核心")
    mem = psutil.virtual_memory()
    print(f"系统内存: 总计 {mem.total / (1024**3):.2f} GB, 可用 {mem.available / (1024**3):.2f} GB")

def print_pytorch_info():
    print("\n=== PyTorch信息 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        
        # 打印所有可用GPU信息
        device_count = torch.cuda.device_count()
        print(f"可用GPU数量: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  - 显存: {props.total_memory / (1024**3):.2f} GB")
            print(f"  - CUDA兼容性: {props.major}.{props.minor}")
            print(f"  - 多处理器数量: {props.multi_processor_count}")
            
        # 检查当前默认设备
        print(f"当前默认设备: cuda:{torch.cuda.current_device()}")
        
        # 检查cudnn状态
        print(f"cuDNN已启用: {torch.backends.cudnn.enabled}")
        print(f"cuDNN benchmark模式: {torch.backends.cudnn.benchmark}")
        print(f"cuDNN确定性模式: {torch.backends.cudnn.deterministic}")

def test_cuda_tensor():
    print("\n=== CUDA张量测试 ===")
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过测试")
        return
    
    try:
        # 创建CPU张量
        print("创建CPU张量...")
        cpu_tensor = torch.randn(1000, 1000)
        
        # 将张量移动到GPU
        print("将张量移动到GPU...")
        start_time = time.time()
        gpu_tensor = cpu_tensor.cuda()
        end_time = time.time()
        print(f"CPU -> GPU传输时间: {(end_time - start_time) * 1000:.2f} ms")
        
        # 在GPU上执行操作
        print("在GPU上执行矩阵乘法...")
        start_time = time.time()
        result = torch.matmul(gpu_tensor, gpu_tensor)
        torch.cuda.synchronize()  # 确保GPU操作完成
        end_time = time.time()
        print(f"GPU矩阵乘法时间: {(end_time - start_time) * 1000:.2f} ms")
        
        # 将结果移回CPU
        print("将结果移回CPU...")
        start_time = time.time()
        result_cpu = result.cpu()
        end_time = time.time()
        print(f"GPU -> CPU传输时间: {(end_time - start_time) * 1000:.2f} ms")
        
        print("张量测试完成，CUDA功能正常")
    except Exception as e:
        print(f"张量测试失败: {str(e)}")

def test_dataloader(num_workers=0, use_gpu=False, batch_size=32):
    print(f"\n=== DataLoader测试 (workers={num_workers}, {'GPU' if use_gpu else 'CPU'}) ===")
    
    # 创建数据集
    dataset = SimpleDataset(size=10000, dim=128)
    
    # 创建数据加载器
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu
    )
    
    # 测试数据加载速度
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    start_time = time.time()
    batches_processed = 0
    total_samples = 0
    
    print(f"开始加载数据，加载 {min(100, len(loader))} 个批次...")
    
    try:
        # 加载最多100个批次
        for i, (x, y) in enumerate(loader):
            if i >= 100:  # 最多加载100个批次以快速测试
                break
                
            # 如果使用GPU，将数据移动到GPU
            if use_gpu and torch.cuda.is_available():
                x = x.to(device)
                y = y.to(device)
            
            # 模拟一些简单的处理
            result = torch.nn.functional.softmax(x.sum(dim=1, keepdim=True), dim=1)
            
            batches_processed += 1
            total_samples += len(x)
            
            # 每10个批次报告一次进度
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"已处理 {i+1} 个批次，{total_samples} 个样本，{elapsed:.2f} 秒")
                
        elapsed_time = time.time() - start_time
        print(f"完成 {batches_processed} 个批次，{total_samples} 个样本")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"吞吐量: {total_samples / elapsed_time:.2f} 样本/秒")
        print(f"每批次耗时: {(elapsed_time / batches_processed) * 1000:.2f} ms")
        
        return True
    except Exception as e:
        print(f"数据加载测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print_system_info()
    print_pytorch_info()
    test_cuda_tensor()
    
    # 测试不同worker配置下的DataLoader性能
    for workers in [0, 2, 4]:
        for use_gpu in [False, True]:
            if use_gpu and not torch.cuda.is_available():
                continue
            test_dataloader(workers, use_gpu)
    
if __name__ == "__main__":
    # Windows上必须保护主模块
    mp.freeze_support()
    main() 