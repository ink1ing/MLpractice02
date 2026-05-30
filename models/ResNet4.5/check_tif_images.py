#!/usr/bin/env python
# TIF图像检查脚本：检查图像结构和维度

import os
import argparse
import numpy as np
import rasterio
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def check_tif_file(file_path):
    """检查单个TIF文件的结构和维度"""
    try:
        with rasterio.open(file_path) as src:
            # 读取图像元数据
            meta = src.meta
            # 读取所有波段
            image = src.read()
            
            return {
                'path': file_path,
                'shape': image.shape,
                'channels': image.shape[0],
                'height': image.shape[1],
                'width': image.shape[2],
                'dtype': image.dtype,
                'min_value': image.min(),
                'max_value': image.max(),
                'meta': meta
            }
    except Exception as e:
        return {
            'path': file_path,
            'error': str(e)
        }

def visualize_tif(file_path, output_dir=None, bands=[0, 100, 200]):
    """可视化TIF文件的选定波段"""
    try:
        with rasterio.open(file_path) as src:
            # 读取所有波段
            image = src.read()
            
            # 选择要显示的波段
            valid_bands = []
            for band in bands:
                if band < image.shape[0]:
                    valid_bands.append(band)
                else:
                    print(f"警告：波段 {band} 超出范围，最大波段索引为 {image.shape[0]-1}")
            
            # 如果没有有效波段，使用前3个或所有可用波段
            if not valid_bands:
                valid_bands = list(range(min(3, image.shape[0])))
            
            # 创建RGB合成图像
            rgb_bands = [image[b] for b in valid_bands[:3]]
            while len(rgb_bands) < 3:
                rgb_bands.append(rgb_bands[-1])  # 复制最后一个波段以填充到3个
            
            rgb = np.dstack(rgb_bands)
            
            # 归一化到0-1范围
            if rgb.max() > 0:
                rgb = rgb / rgb.max()
            
            # 创建图像
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb)
            plt.title(f"文件: {os.path.basename(file_path)}\n波段: {valid_bands}")
            plt.axis('off')
            
            # 保存或显示
            if output_dir:
                save_path = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.tif', '')}_viz.png")
                plt.savefig(save_path)
                plt.close()
                print(f"图像已保存到: {save_path}")
            else:
                plt.show()
                
    except Exception as e:
        print(f"可视化图像时出错: {file_path}, 错误: {e}")

def main():
    parser = argparse.ArgumentParser(description='TIF图像结构检查工具')
    
    parser.add_argument('--data_root', type=str, default='./guanceng-bit',
                        help='数据根目录路径')
    parser.add_argument('--sample_count', type=int, default=5,
                        help='每个子文件夹检查的样本数量')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化样本')
    parser.add_argument('--output_dir', type=str, default='./tif_viz',
                        help='可视化图像保存目录')
    
    args = parser.parse_args()
    
    if args.visualize and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print("====== TIF图像结构检查 ======")
    print(f"数据根目录: {os.path.abspath(args.data_root)}")
    
    # 查找所有叶片目录
    leaf_dirs = []
    for pattern in ['9*', '14*', '19*']:
        matching_dirs = glob.glob(os.path.join(args.data_root, pattern))
        leaf_dirs.extend(matching_dirs)
    
    if not leaf_dirs:
        print("警告: 未找到任何叶片目录!")
        return
    
    print(f"找到 {len(leaf_dirs)} 个叶片目录")
    
    # 检查TIF文件
    total_files = 0
    shape_counts = {}
    channel_counts = {}
    
    for leaf_dir in leaf_dirs:
        dir_name = os.path.basename(leaf_dir)
        print(f"\n检查目录: {dir_name}")
        
        # 获取TIF文件列表
        tif_files = [f for f in os.listdir(leaf_dir) if f.endswith('.tif')]
        
        # 随机抽样检查
        sample_count = min(args.sample_count, len(tif_files))
        sample_files = np.random.choice(tif_files, sample_count, replace=False)
        
        for tif_file in sample_files:
            tif_path = os.path.join(leaf_dir, tif_file)
            info = check_tif_file(tif_path)
            
            if 'error' in info:
                print(f"  错误: {tif_file} - {info['error']}")
                continue
            
            # 记录图像形状统计
            shape_str = str(info['shape'])
            shape_counts[shape_str] = shape_counts.get(shape_str, 0) + 1
            
            # 记录通道数统计
            channel_counts[info['channels']] = channel_counts.get(info['channels'], 0) + 1
            
            print(f"  {tif_file} - 形状: {info['shape']}, 类型: {info['dtype']}, 值范围: [{info['min_value']}, {info['max_value']}]")
            
            # 可视化图像
            if args.visualize:
                visualize_tif(tif_path, args.output_dir)
        
        total_files += len(tif_files)
        print(f"  目录 {dir_name} 包含 {len(tif_files)} 个TIF文件")
    
    print("\n====== 统计信息 ======")
    print(f"总计 {total_files} 个TIF文件")
    
    print("\n图像形状分布:")
    for shape, count in shape_counts.items():
        print(f"  {shape}: {count} 个文件 ({count/sum(shape_counts.values())*100:.1f}%)")
    
    print("\n通道数分布:")
    for channels, count in sorted(channel_counts.items()):
        print(f"  {channels} 通道: {count} 个文件 ({count/sum(channel_counts.values())*100:.1f}%)")
    
    print("\n====== 检查完成 ======")

if __name__ == "__main__":
    main() 