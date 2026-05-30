#!/usr/bin/env python
# 旧版本文件：系统检查脚本，检查系统环境和配置

import torch
import sys
import os

def main():
    print("=" * 60)
    print("系统环境检查")
    print("=" * 60)
    
    # Python版本
    print(f"Python版本: {sys.version}")
    
    # PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # CUDA信息
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
        # 当前GPU内存使用情况
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"当前GPU内存使用: {allocated:.2f} GB (已分配), {cached:.2f} GB (已缓存)")
    else:
        print("CUDA不可用，将使用CPU训练")
    
    # 混合精度支持
    print(f"混合精度支持: {torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')}")
    
    # 系统内存
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"系统内存: {memory.total / 1024**3:.1f} GB (总计), {memory.available / 1024**3:.1f} GB (可用)")
    except ImportError:
        print("系统内存: 无法检测 (需要安装psutil)")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 