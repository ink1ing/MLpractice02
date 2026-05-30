#!/usr/bin/env python
# 数据加载器调试脚本：专注测试数据加载过程性能，找出瓶颈

import os
import torch
import time
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入自定义模块
from dataset import CornRustDataset, get_dataloaders

def test_dataset_only(data_root, json_root, img_size=128, use_extended_dataset=True):
    """测试数据集创建和单个样本加载性能"""
    print("\n=== 测试数据集创建和单个样本加载 ===")
    
    # 计时数据集初始化
    start_time = time.time()
    try:
        dataset = CornRustDataset(
            data_dir=data_root,
            json_dir=json_root,
            img_size=img_size,
            use_extended_dataset=use_extended_dataset
        )
        init_time = time.time() - start_time
        print(f"数据集初始化时间: {init_time:.2f}秒")
        print(f"数据集大小: {len(dataset)} 个样本")
    except Exception as e:
        print(f"创建数据集时出错: {e}")
        return None
    
    # 测试单个样本加载时间
    sample_times = []
    print("\n测试随机样本加载时间 (10个样本)...")
    
    # 随机测试10个样本的加载时间
    for i in range(min(10, len(dataset))):
        idx = i  # 顺序加载可以利用缓存优势
        start_time = time.time()
        try:
            _ = dataset[idx]
            load_time = time.time() - start_time
            sample_times.append(load_time)
            print(f"样本 {idx}: {load_time:.4f}秒")
        except Exception as e:
            print(f"加载样本 {idx} 时出错: {e}")
    
    # 输出统计信息
    if sample_times:
        avg_time = sum(sample_times) / len(sample_times)
        max_time = max(sample_times)
        min_time = min(sample_times)
        print(f"\n样本加载时间统计:")
        print(f"平均: {avg_time:.4f}秒")
        print(f"最长: {max_time:.4f}秒")
        print(f"最短: {min_time:.4f}秒")
    
    return dataset

def test_dataloader(dataset, batch_size=4, num_workers=0, prefetch_factor=2, pin_memory=False):
    """测试数据加载器性能"""
    if dataset is None or len(dataset) == 0:
        print("数据集为空，无法测试数据加载器")
        return
    
    print(f"\n=== 测试数据加载器 (批次大小={batch_size}, workers={num_workers}) ===")
    
    # 创建数据加载器
    try:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0
        )
        print(f"数据加载器创建成功，包含 {len(loader)} 个批次")
    except Exception as e:
        print(f"创建数据加载器时出错: {e}")
        return
    
    # 测试整个epoch的加载时间
    print("\n测试完整数据加载时间...")
    start_time = time.time()
    batch_times = []
    total_samples = 0
    
    try:
        for i, batch in enumerate(tqdm(loader, desc="加载批次")):
            # 记录每个批次加载时间
            batch_end_time = time.time()
            batch_time = batch_end_time - start_time
            batch_times.append(batch_time)
            start_time = batch_end_time  # 为下一批次重置时间
            
            # 计算加载的样本数
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                if torch.is_tensor(batch[0]):
                    total_samples += batch[0].size(0)
                else:
                    total_samples += len(batch[0])
            
            # 打印每10个批次的平均时间
            if (i + 1) % 10 == 0:
                recent_avg = sum(batch_times[-10:]) / min(10, len(batch_times[-10:]))
                print(f"最近10个批次平均加载时间: {recent_avg:.4f}秒/批次")
    except Exception as e:
        print(f"数据加载过程出错: {e}")
    
    # 输出统计信息
    if batch_times:
        total_time = sum(batch_times)
        avg_time = sum(batch_times) / len(batch_times)
        max_time = max(batch_times)
        min_time = min(batch_times)
        
        print(f"\n数据加载器性能统计:")
        print(f"总加载时间: {total_time:.2f}秒")
        print(f"平均批次时间: {avg_time:.4f}秒")
        print(f"最长批次时间: {max_time:.4f}秒")
        print(f"最短批次时间: {min_time:.4f}秒")
        print(f"每秒处理样本数: {total_samples / total_time:.2f}")

def investigate_data_structure(data_root, json_root):
    """检查数据目录结构，为优化提供信息"""
    print("\n=== 数据结构检查 ===")
    
    # 检查数据根目录
    if not os.path.exists(data_root):
        print(f"错误: 数据根目录不存在: {data_root}")
        return
    
    # 检查JSON目录
    if not os.path.exists(json_root):
        print(f"错误: JSON目录不存在: {json_root}")
        return
    
    # 分析数据根目录
    print(f"\n数据根目录: {os.path.abspath(data_root)}")
    data_subdirs = []
    data_files = []
    
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            data_subdirs.append(item)
            files_count = len([f for f in os.listdir(item_path) if f.endswith('.tif')])
            print(f"  - 子目录: {item} ({files_count} 个TIF文件)")
        elif item.endswith('.tif'):
            data_files.append(item)
    
    if data_files:
        print(f"  - 根目录直接包含 {len(data_files)} 个TIF文件")
    
    # 分析JSON目录
    print(f"\nJSON目录: {os.path.abspath(json_root)}")
    json_subdirs = []
    json_files = []
    
    for item in os.listdir(json_root):
        item_path = os.path.join(json_root, item)
        if os.path.isdir(item_path):
            json_subdirs.append(item)
            files_count = len([f for f in os.listdir(item_path) if f.endswith('.json')])
            print(f"  - 子目录: {item} ({files_count} 个JSON文件)")
        elif item.endswith('.json'):
            json_files.append(item)
    
    if json_files:
        print(f"  - 根目录直接包含 {len(json_files)} 个JSON文件")
    
    # 分析数据集结构模式
    if data_subdirs and all(subdir.endswith(('l', 'm', 't')) or any(x in subdir for x in ['9l', '9m', '9t', '14l', '14m', '14t', '19l', '19m', '19t']) for subdir in data_subdirs):
        print("\n检测到叶片位置目录结构 (l/m/t)，适合扩展数据集模式")
    else:
        print("\n未检测到标准叶片位置目录结构，建议使用基本数据集模式")

def main():
    parser = argparse.ArgumentParser(description='玉米南方锈病数据加载调试工具')
    
    # 基本参数
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='数据根目录路径')
    parser.add_argument('--json_root', type=str, default='./biaozhu_json',
                        help='JSON标注根目录路径')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数')
    parser.add_argument('--img_size', type=int, default=128,
                        help='图像大小')
    parser.add_argument('--no_extended', action='store_true',
                        help='不使用扩展数据集模式')
    parser.add_argument('--pin_memory', action='store_true',
                        help='使用pin_memory')
    
    args = parser.parse_args()
    
    # 打印系统信息
    print("=== 系统信息 ===")
    print(f"Python版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    # 检查数据结构
    investigate_data_structure(args.data_root, args.json_root)
    
    # 测试数据集
    use_extended = not args.no_extended
    dataset = test_dataset_only(args.data_root, args.json_root, args.img_size, use_extended)
    
    # 测试不同worker配置的数据加载器
    if dataset:
        # 测试当前参数配置
        test_dataloader(dataset, args.batch_size, args.num_workers, 2, args.pin_memory)
        
        # 如果用户指定了多个workers，也测试单线程版本进行对比
        if args.num_workers > 0:
            print("\n对比测试: 单线程数据加载...")
            test_dataloader(dataset, args.batch_size, 0, 2, args.pin_memory)

if __name__ == "__main__":
    main() 