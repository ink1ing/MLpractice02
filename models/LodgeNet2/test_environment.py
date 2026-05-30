#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 旧版本文件：环境测试脚本，验证深度学习环境配置

import os
import sys
import torch
import numpy as np
import time
from datetime import datetime

def test_basic_imports():
    """测试基础库导入"""
    print("=" * 60)
    print("1. 基础库导入测试")
    print("=" * 60)
    
    try:
        import torch
        import torchvision
        import numpy as np
        import matplotlib.pyplot as plt
        import sklearn
        import tqdm
        import rasterio
        import json
        print("✅ 所有基础库导入成功")
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"NumPy版本: {np.__version__}")
        print(f"Rasterio版本: {rasterio.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ 基础库导入失败: {e}")
        return False

def test_custom_modules():
    """测试自定义模块导入"""
    print("\n" + "=" * 60)
    print("2. 自定义模块导入测试")
    print("=" * 60)
    
    modules_to_test = [
        'dataset',
        'lodgenet_model', 
        'utils'
    ]
    
    success_count = 0
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}.py 导入成功")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module}.py 导入失败: {e}")
    
    print(f"\n模块导入成功率: {success_count}/{len(modules_to_test)}")
    return success_count == len(modules_to_test)

def test_gpu_environment():
    """测试GPU环境"""
    print("\n" + "=" * 60)
    print("3. GPU环境测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    print(f"✅ CUDA可用")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # 测试GPU内存分配
    try:
        test_tensor = torch.randn(100, 100).cuda()
        print("✅ GPU内存分配测试成功")
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ GPU内存分配测试失败: {e}")
        return False

def test_data_directories():
    """测试数据目录结构"""
    print("\n" + "=" * 60)
    print("4. 数据目录结构测试")
    print("=" * 60)
    
    data_root = './guanceng-bit'
    json_root = './biaozhu_json'
    
    # 检查主目录
    if not os.path.exists(data_root):
        print(f"❌ 数据目录不存在: {data_root}")
        return False
    
    if not os.path.exists(json_root):
        print(f"❌ JSON目录不存在: {json_root}")
        return False
    
    print(f"✅ 主目录存在")
    
    # 检查子目录
    expected_subdirs = ['9l', '9m', '9t', '14l', '14m', '14t', '19l', '19m', '19t']
    data_subdirs = []
    json_subdirs = []
    
    for subdir in expected_subdirs:
        data_path = os.path.join(data_root, subdir)
        json_path = os.path.join(json_root, f"{subdir}_json")
        
        if os.path.exists(data_path):
            data_subdirs.append(subdir)
            tif_count = len([f for f in os.listdir(data_path) if f.endswith('.tif')])
            print(f"✅ {subdir}: {tif_count} TIF文件")
        
        if os.path.exists(json_path):
            json_subdirs.append(subdir)
            json_count = len([f for f in os.listdir(json_path) if f.endswith('.json')])
            print(f"✅ {subdir}_json: {json_count} JSON文件")
    
    print(f"\n数据子目录: {len(data_subdirs)}/{len(expected_subdirs)}")
    print(f"JSON子目录: {len(json_subdirs)}/{len(expected_subdirs)}")
    
    return len(data_subdirs) > 0 and len(json_subdirs) > 0

def test_dataset_loading():
    """测试数据集加载"""
    print("\n" + "=" * 60)
    print("5. 数据集加载测试")
    print("=" * 60)
    
    try:
        from dataset import CornRustDataset, get_dataloaders
        
        # 创建数据集
        print("创建数据集...")
        dataset = CornRustDataset(
            data_dir='./guanceng-bit',
            json_dir='./biaozhu_json',
            img_size=128,
            use_extended_dataset=True
        )
        
        print(f"✅ 数据集创建成功，样本数量: {len(dataset)}")
        
        if len(dataset) == 0:
            print("❌ 数据集为空")
            return False
        
        # 测试单个样本加载
        print("测试样本加载...")
        start_time = time.time()
        sample = dataset[0]
        load_time = time.time() - start_time
        
        image, position, grade = sample
        print(f"✅ 样本加载成功")
        print(f"图像形状: {image.shape}")
        print(f"位置标签: {position}")
        print(f"等级标签: {grade}")
        print(f"加载时间: {load_time:.4f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("6. 模型创建测试")
    print("=" * 60)
    
    try:
        from lodgenet_model import get_lodgenet_model, count_parameters
        
        print("创建LodgeNet模型...")
        model = get_lodgenet_model(
            n_channels=3,
            n_classes=2,
            img_size=128,
            bilinear=True
        )
        
        param_count = count_parameters(model)
        print(f"✅ 模型创建成功")
        print(f"参数数量: {param_count:,}")
        
        # 测试前向传播
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        test_input = torch.randn(2, 3, 128, 128).to(device)
        
        print("测试前向传播...")
        with torch.no_grad():
            seg_out, pos_out, grade_out = model(test_input)
        
        print(f"✅ 前向传播成功")
        print(f"分割输出形状: {seg_out.shape}")
        print(f"位置输出形状: {pos_out.shape}")
        print(f"等级输出形状: {grade_out.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader_creation():
    """测试数据加载器创建"""
    print("\n" + "=" * 60)
    print("7. 数据加载器创建测试")
    print("=" * 60)
    
    try:
        from dataset import get_dataloaders
        
        print("创建数据加载器...")
        train_loader, val_loader = get_dataloaders(
            data_root='./guanceng-bit',
            json_root='./biaozhu_json',
            batch_size=4,
            num_workers=2,
            img_size=128,
            train_ratio=0.8,
            use_extended_dataset=True
        )
        
        print(f"✅ 数据加载器创建成功")
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        
        # 测试一个批次
        print("测试批次加载...")
        start_time = time.time()
        for batch in train_loader:
            images, positions, grades = batch
            load_time = time.time() - start_time
            
            print(f"✅ 批次加载成功")
            print(f"批次图像形状: {images.shape}")
            print(f"批次位置形状: {positions.shape}")
            print(f"批次等级形状: {grades.shape}")
            print(f"批次加载时间: {load_time:.4f}秒")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_components():
    """测试训练组件"""
    print("\n" + "=" * 60)
    print("8. 训练组件测试")
    print("=" * 60)
    
    try:
        from utils import FocalLoss
        import torch.nn as nn
        import torch.optim as optim
        
        # 测试损失函数
        print("测试损失函数...")
        focal_loss = FocalLoss(alpha=None, gamma=2.0)
        mse_loss = nn.MSELoss()
        
        # 创建测试数据
        test_logits = torch.randn(4, 3)
        test_labels = torch.randint(0, 3, (4,))
        test_grades = torch.randn(4, 1)
        test_grade_labels = torch.randn(4, 1)
        
        # 计算损失
        pos_loss = focal_loss(test_logits, test_labels)
        grade_loss = mse_loss(test_grades, test_grade_labels)
        
        print(f"✅ 损失函数测试成功")
        print(f"位置损失: {pos_loss.item():.4f}")
        print(f"等级损失: {grade_loss.item():.4f}")
        
        # 测试优化器
        print("测试优化器...")
        from lodgenet_model import get_lodgenet_model
        model = get_lodgenet_model(3, 2, 128)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        
        print(f"✅ 优化器创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 LodgeNet环境测试开始")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        test_basic_imports,
        test_custom_modules,
        test_gpu_environment,
        test_data_directories,
        test_dataset_loading,
        test_model_creation,
        test_dataloader_creation,
        test_training_components
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 出现异常: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 测试结果汇总")
    print("=" * 60)
    print(f"通过测试: {passed_tests}/{total_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！环境配置完善，可以开始训练")
        return True
    else:
        print("⚠️ 部分测试失败，请检查环境配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 