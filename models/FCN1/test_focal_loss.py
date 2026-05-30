#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FocalLoss测试脚本
用于测试FocalLoss类在不同输入条件下的行为
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 添加当前目录到导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入FocalLoss类
try:
    from utils import FocalLoss
    print("成功导入FocalLoss类")
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

def test_focal_loss_basic():
    """测试FocalLoss在基本场景下的行为"""
    print("\n=== 测试FocalLoss基本功能 ===")
    
    # 创建FocalLoss实例
    focal_loss = FocalLoss(gamma=2.0)
    
    # 测试不同维度的输入
    test_scenarios = [
        {
            "name": "标准分类场景",
            "inputs": torch.randn(4, 3),  # 批次大小为4，类别数为3的logits
            "targets": torch.tensor([0, 1, 2, 1]),  # 对应的类别索引
            "expected_shape": torch.Size([])  # 期望得到标量损失
        },
        {
            "name": "回归场景",
            "inputs": torch.randn(4, 1),  # 批次大小为4，单个输出值
            "targets": torch.randn(4, 1),  # 对应的连续值标签
            "expected_shape": torch.Size([])  # 期望得到标量损失
        },
        {
            "name": "多标签分类",
            "inputs": torch.randn(4, 3),  # 批次大小为4，类别数为3的logits
            "targets": torch.randint(0, 2, (4, 3)).float(),  # 多标签，每个样本可能有多个为1
            "expected_shape": torch.Size([])  # 期望得到标量损失
        },
        {
            "name": "FCN输出场景 (4D tensor)",
            "inputs": torch.randn(4, 3, 32, 32),  # 批次大小为4，类别数为3，特征图大小32x32
            "targets": torch.tensor([0, 1, 2, 1]),  # 对应的类别索引
            "expected_shape": torch.Size([])  # 期望得到标量损失
        }
    ]
    
    # 执行测试
    for scenario in test_scenarios:
        try:
            print(f"\n测试场景: {scenario['name']}")
            print(f"输入形状: {scenario['inputs'].shape}")
            print(f"目标形状: {scenario['targets'].shape}")
            
            # 计算损失
            loss = focal_loss(scenario['inputs'], scenario['targets'])
            
            print(f"损失值: {loss.item():.4f}")
            print(f"损失形状: {loss.shape}")
            
            # 验证损失形状
            assert loss.shape == scenario['expected_shape'], f"期望形状: {scenario['expected_shape']}, 实际形状: {loss.shape}"
            print("测试通过: 损失形状符合预期")
            
            # 验证损失值
            assert not torch.isnan(loss).any(), "损失值包含NaN"
            assert not torch.isinf(loss).any(), "损失值包含Inf"
            print("测试通过: 损失值有效")
            
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()

def test_fcn_scenario():
    """测试FCN模型使用场景"""
    print("\n=== 测试FCN模型场景 ===")
    
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)
    
    # 模拟FCN模型的输出
    batch_size = 8
    img_size = 128
    position_classes = 3
    
    # 创建一个简化的FCN模型类来模拟实际行为
    class SimpleFCN(nn.Module):
        def __init__(self, in_channels=3, position_classes=3, img_size=128):
            super(SimpleFCN, self).__init__()
            self.img_size = img_size
            
            # 简化的编码器
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            
            # 位置头 - 输出每个像素的类别
            self.position_head = nn.Conv2d(32, position_classes, kernel_size=1)
            
            # 等级头 - 输出整体等级
            self.grade_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            # 编码
            features = self.encoder(x)
            
            # 位置分类
            position_logits = self.position_head(features)
            
            # 从[B, C, H, W]的分割图转为[B, C]的分类logits
            # 这里我们对空间维度进行全局平均池化，模拟FCN的分类输出
            position_logits = F.adaptive_avg_pool2d(position_logits, 1).view(position_logits.size(0), -1)
            
            # 等级预测
            grade_values = self.grade_head(features).view(x.size(0), -1)
            
            return position_logits, grade_values
    
    # 创建模型和损失函数
    model = SimpleFCN(in_channels=3, position_classes=position_classes, img_size=img_size)
    position_criterion = FocalLoss(gamma=2.0)
    grade_criterion = nn.MSELoss()
    
    # 创建模拟输入
    images = torch.randn(batch_size, 3, img_size, img_size)
    position_labels = torch.randint(0, position_classes, (batch_size,))
    grade_labels = torch.rand(batch_size, 1)
    
    print(f"输入批次大小: {batch_size}")
    print(f"图像形状: {images.shape}")
    print(f"位置标签形状: {position_labels.shape}")
    print(f"等级标签形状: {grade_labels.shape}")
    
    try:
        # 前向传播
        position_logits, grade_values = model(images)
        
        print(f"位置logits形状: {position_logits.shape}")
        print(f"等级值形状: {grade_values.shape}")
        
        # 计算损失
        pos_loss = position_criterion(position_logits, position_labels)
        grade_loss = grade_criterion(grade_values, grade_labels)
        
        print(f"位置损失值: {pos_loss.item():.4f}")
        print(f"等级损失值: {grade_loss.item():.4f}")
        
        # 组合损失
        task_weights = [0.7, 0.3]
        loss = task_weights[0] * pos_loss + task_weights[1] * grade_loss
        
        print(f"组合损失值: {loss.item():.4f}")
        
        # 验证损失值
        assert not torch.isnan(loss).any(), "损失值包含NaN"
        assert not torch.isinf(loss).any(), "损失值包含Inf"
        print("测试通过: 损失值有效")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def inspect_model_outputs():
    """调查模型输出的形状和特性"""
    print("\n=== 调查真实FCN模型输出 ===")
    
    try:
        # 导入FCN模型
        from model import get_model
        
        # 设置参数
        batch_size = 4
        img_size = 128
        
        # 创建模型
        model = get_model(model_type='fcn', in_channels=3, img_size=img_size)
        print(f"模型创建成功，类型: {type(model).__name__}")
        
        # 创建随机输入
        images = torch.randn(batch_size, 3, img_size, img_size)
        
        # 创建随机目标标签
        position_labels = torch.randint(0, 3, (batch_size,))
        
        # 创建FocalLoss实例
        focal_loss = FocalLoss(gamma=2.0)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            position_logits, grade_values = model(images)
            
            # 检查输出形状
            print(f"位置logits形状: {position_logits.shape}")
            print(f"等级值形状: {grade_values.shape}")
            
            # 检查输出值范围
            print(f"位置logits范围: [{position_logits.min().item():.4f}, {position_logits.max().item():.4f}]")
            print(f"等级值范围: [{grade_values.min().item():.4f}, {grade_values.max().item():.4f}]")
            
            # 尝试计算损失
            try:
                print("尝试计算损失...")
                loss = focal_loss(position_logits, position_labels)
                print(f"损失值: {loss.item():.4f}")
                print("损失计算成功!")
            except Exception as loss_err:
                print(f"损失计算失败: {loss_err}")
            
        print("测试完成")
    except Exception as e:
        print(f"模型检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("========== FocalLoss测试脚本 ==========")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 运行测试
    test_focal_loss_basic()
    test_fcn_scenario()
    
    try:
        # 如果模型可用，检查其输出
        from model import get_model
        inspect_model_outputs()
    except ImportError:
        print("无法导入模型，跳过模型输出检查")
    
    print("\n所有测试完成!") 