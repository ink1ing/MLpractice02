#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 基础数据文件测试脚本

import os
import sys
import time
import rasterio
import json
import numpy as np
import torch
from PIL import Image

def test_file_access():
    """测试基本文件访问"""
    print("=" * 60)
    print("测试基本文件访问")
    print("=" * 60)
    
    # 检查目录
    data_root = './guanceng-bit'
    json_root = './biaozhu_json'
    
    print(f"数据目录: {data_root}")
    print(f"JSON目录: {json_root}")
    
    if not os.path.exists(data_root):
        print(f"错误: 数据目录不存在")
        return False
        
    if not os.path.exists(json_root):
        print(f"错误: JSON目录不存在")
        return False
    
    # 列出子目录
    data_subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    print(f"数据子目录: {data_subdirs}")
    
    json_subdirs = [d for d in os.listdir(json_root) if os.path.isdir(os.path.join(json_root, d))]
    print(f"JSON子目录: {json_subdirs}")
    
    return True

def test_single_tif_file():
    """测试单个TIF文件读取"""
    print("\n" + "=" * 60)
    print("测试单个TIF文件读取")
    print("=" * 60)
    
    # 找到第一个TIF文件
    data_root = './guanceng-bit'
    tif_file = None
    
    for subdir in os.listdir(data_root):
        subdir_path = os.path.join(data_root, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.tif'):
                    tif_file = os.path.join(subdir_path, file)
                    break
            if tif_file:
                break
    
    if not tif_file:
        print("错误: 找不到TIF文件")
        return False
    
    print(f"测试文件: {tif_file}")
    
    # 测试rasterio读取
    try:
        print("使用rasterio读取...")
        start_time = time.time()
        
        with rasterio.open(tif_file) as src:
            print(f"  文件信息:")
            print(f"    形状: {src.shape}")
            print(f"    波段数: {src.count}")
            print(f"    数据类型: {src.dtypes}")
            print(f"    CRS: {src.crs}")
            
            # 读取数据
            data = src.read()
            print(f"    数据形状: {data.shape}")
            print(f"    数据范围: {data.min()} - {data.max()}")
        
        load_time = time.time() - start_time
        print(f"  rasterio读取成功，耗时: {load_time:.3f}秒")
        
    except Exception as e:
        print(f"  rasterio读取失败: {e}")
        return False
    
    # 测试PIL读取
    try:
        print("使用PIL读取...")
        start_time = time.time()
        
        img = Image.open(tif_file)
        print(f"  PIL图像信息:")
        print(f"    尺寸: {img.size}")
        print(f"    模式: {img.mode}")
        
        # 转换为numpy数组
        img_array = np.array(img)
        print(f"    数组形状: {img_array.shape}")
        print(f"    数据类型: {img_array.dtype}")
        
        load_time = time.time() - start_time
        print(f"  PIL读取成功，耗时: {load_time:.3f}秒")
        
    except Exception as e:
        print(f"  PIL读取失败: {e}")
    
    return True

def test_single_json_file():
    """测试单个JSON文件读取"""
    print("\n" + "=" * 60)
    print("测试单个JSON文件读取")
    print("=" * 60)
    
    # 找到第一个JSON文件
    json_root = './biaozhu_json'
    json_file = None
    
    for subdir in os.listdir(json_root):
        subdir_path = os.path.join(json_root, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.json'):
                    json_file = os.path.join(subdir_path, file)
                    break
            if json_file:
                break
    
    if not json_file:
        print("错误: 找不到JSON文件")
        return False
    
    print(f"测试文件: {json_file}")
    
    try:
        start_time = time.time()
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        load_time = time.time() - start_time
        print(f"JSON读取成功，耗时: {load_time:.3f}秒")
        print(f"JSON内容类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"JSON键: {list(data.keys())}")
            for key, value in data.items():
                print(f"  {key}: {type(value)} = {value}")
        elif isinstance(data, list):
            print(f"JSON列表长度: {len(data)}")
            if data:
                print(f"第一个元素: {data[0]}")
        
        return True
        
    except Exception as e:
        print(f"JSON读取失败: {e}")
        return False

def test_dataset_import():
    """测试数据集模块导入"""
    print("\n" + "=" * 60)
    print("测试数据集模块导入")
    print("=" * 60)
    
    try:
        print("导入dataset模块...")
        from dataset import CornRustDataset, get_dataloaders
        print("dataset模块导入成功")
        
        print("导入lodgenet_model模块...")
        from lodgenet_model import get_lodgenet_model
        print("lodgenet_model模块导入成功")
        
        print("导入utils模块...")
        from utils import FocalLoss
        print("utils模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_dataset_creation():
    """测试简单数据集创建"""
    print("\n" + "=" * 60)
    print("测试简单数据集创建")
    print("=" * 60)
    
    try:
        from dataset import CornRustDataset
        
        print("创建数据集对象...")
        start_time = time.time()
        
        # 使用最简单的配置
        dataset = CornRustDataset(
            data_root='./guanceng-bit',
            json_root='./biaozhu_json',
            img_size=64,  # 使用更小的图像尺寸
            use_extended_dataset=False  # 不使用扩展模式
        )
        
        create_time = time.time() - start_time
        print(f"数据集创建成功，耗时: {create_time:.3f}秒")
        print(f"数据集大小: {len(dataset)}")
        
        return dataset
        
    except Exception as e:
        print(f"数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_single_sample(dataset):
    """测试单个样本获取"""
    print("\n" + "=" * 60)
    print("测试单个样本获取")
    print("=" * 60)
    
    if dataset is None:
        print("数据集为空，跳过测试")
        return False
    
    try:
        print("获取第一个样本...")
        start_time = time.time()
        
        sample = dataset[0]
        
        load_time = time.time() - start_time
        print(f"样本获取成功，耗时: {load_time:.3f}秒")
        
        if isinstance(sample, tuple):
            print(f"样本元组长度: {len(sample)}")
            for i, item in enumerate(sample):
                if torch.is_tensor(item):
                    print(f"  元素{i}: tensor, 形状={item.shape}, 类型={item.dtype}")
                else:
                    print(f"  元素{i}: {type(item)}, 值={item}")
        else:
            print(f"样本类型: {type(sample)}")
        
        return True
        
    except Exception as e:
        print(f"样本获取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始基础数据测试...")
    
    # 1. 测试文件访问
    if not test_file_access():
        print("文件访问测试失败")
        return
    
    # 2. 测试TIF文件读取
    if not test_single_tif_file():
        print("TIF文件读取测试失败")
        return
    
    # 3. 测试JSON文件读取
    if not test_single_json_file():
        print("JSON文件读取测试失败")
        return
    
    # 4. 测试模块导入
    if not test_dataset_import():
        print("模块导入测试失败")
        return
    
    # 5. 测试数据集创建
    dataset = test_simple_dataset_creation()
    if dataset is None:
        print("数据集创建测试失败")
        return
    
    # 6. 测试样本获取
    if not test_single_sample(dataset):
        print("样本获取测试失败")
        return
    
    print("\n" + "=" * 60)
    print("所有基础测试通过！")
    print("=" * 60)

if __name__ == "__main__":
    main() 