# 玉米南方锈病遥感识别模型 - 项目交接文档

## 项目概述

本项目旨在构建多任务深度学习模型，利用多光谱遥感图像（.tif格式）和标注数据（.json格式），实现玉米南方锈病的感染部位识别（下部/中部/上部）和病害等级预测（0/3/5/7/9）。

项目分为两个阶段：
1. **阶段一（已完成）**：使用14叶片数量的小样本验证模型框架和训练流程
2. **阶段二（当前）**：扩展到9个文件夹数据集（9l/m/t、14l/m/t、19l/m/t），每类101张，共909张样本对

## 数据结构

### 图像数据
- 多光谱遥感图像（.tif格式）
- 图像维度：[500, H, W]，其中500是光谱通道数
- 存储路径：`./guanceng-bit/`目录下，按叶片和位置分为9个子目录：
  - 9l, 9m, 9t（9叶期下/中/上部）
  - 14l, 14m, 14t（14叶期下/中/上部）
  - 19l, 19m, 19t（19叶期下/中/上部）

### 标注数据
- JSON格式标注文件，与图像文件一一对应
- 存储路径：`./biaozhu_json/`目录下，按叶片和位置分为9个子目录（与图像目录对应）
- 标注内容：
  - 感染部位：通过文件路径中的l/m/t标识（下/中/上部）
  - 感染等级：通过标签名称中的数字标识（0/3/5/7/9）

## 模型架构

项目实现了三种模型架构：
1. **简单CNN**（`DiseaseClassifier`）：基础卷积神经网络，双头输出
2. **ResNet**（`DiseaseResNet`）：基于残差连接的深度网络
3. **ResNet+**（`DiseaseResNetPlus`）：增强版ResNet，加入通道和空间注意力机制

所有模型都采用多任务学习方法，同时预测：
- 感染部位（3分类问题）：下部/中部/上部
- 感染等级（回归问题）：0-9范围内的连续值

## 关键文件说明

- **model.py**: 模型定义，包含三种网络架构
- **dataset.py**: 数据集加载和预处理，处理TIF图像和JSON标注
- **train.py**: 训练主脚本，包含训练循环和评估函数
- **run_training.py**: 训练启动脚本，设置最优参数配置
- **utils.py**: 工具函数，包含损失函数、指标计算等
- **check_tif_images.py**: TIF图像检查工具，用于分析图像结构
- **check_training_results.py**: 训练结果查看工具，用于分析模型性能

## 最近修复的问题

在扩展到完整数据集时，我们修复了以下关键问题：

1. **图像维度不一致**：
   - 问题：图像形状不一致，有的是[3, 128, 128]，有的是[128, 600, 128]
   - 解决方案：在dataset.py中修改了__getitem__方法，从500通道中选择3个有代表性的通道

2. **图像通道数过多**：
   - 问题：图像实际有500个通道而非原本假设的3个通道
   - 解决方案：选择第1、第250和第500通道作为代表性通道，保留关键信息

3. **图像缩放错误**：
   - 问题：transforms_functional.resize函数不支持多通道图像
   - 解决方案：使用scikit-image库的resize函数正确处理图像缩放

4. **FocalLoss实现错误**：
   - 问题：Target size (torch.Size([4])) must be the same as input size (torch.Size([4, 3]))
   - 解决方案：修改FocalLoss实现，增加维度检查和处理，支持不同形状的输入

## 使用指南

### 环境准备
```bash
pip install torch torchvision rasterio scikit-image matplotlib tqdm seaborn numpy pillow
```

### 检查TIF图像
```bash
python check_tif_images.py --data_root ./guanceng-bit --sample_count 5 --visualize
```

### 启动训练
```bash
python run_training.py --data_root ./guanceng-bit --json_root ./biaozhu_json --output_dir ./output
```

### 检查训练结果
```bash
python check_training_results.py --output_dir ./output
```

### 测试模型
```bash
python run_testing.py --data_root ./guanceng-bit --json_root ./biaozhu_json --model_path ./output/best_model.pth
```

## 训练参数配置

当前最优参数配置（针对多光谱图像优化）：
- 模型类型：ResNet+（带注意力机制）
- 图像大小：128x128
- 输入通道数：3（从500通道中选择）
- 损失函数：Focal Loss（γ=2.0）
- 学习率：0.0001（降低以提高稳定性）
- 批次大小：8（GPU）/ 4（CPU）
- 优化器：Adam
- 学习率调度：ReduceLROnPlateau

## 性能指标

模型性能通过以下指标评估：
- 位置分类：准确率、F1分数、精确率、召回率、混淆矩阵
- 等级预测：平均绝对误差(MAE)、±2误差容忍率

## 注意事项和建议

1. **数据预处理**：
   - 多光谱图像处理需要特别注意通道选择和维度处理
   - 确保图像和标注文件正确匹配

2. **训练资源**：
   - 处理多光谱图像需要较大内存，建议使用GPU训练
   - 如果出现内存不足，可进一步减小批次大小

3. **模型选择**：
   - 简单任务可使用基础CNN模型
   - 复杂场景建议使用ResNet+模型，注意力机制有助于捕捉关键特征

4. **后续优化方向**：
   - 探索更多通道选择策略，提取更有代表性的光谱信息
   - 尝试更复杂的多任务学习权重策略
   - 考虑引入迁移学习，利用预训练模型

## 联系方式

如有任何问题，请联系项目负责人。 