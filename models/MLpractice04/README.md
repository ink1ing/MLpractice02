# 玉米南方锈病遥感识别模型

本项目旨在利用深度学习技术，构建玉米南方锈病的遥感识别模型，实现对感染部位和感染等级的自动识别。

## 项目更新说明

### 2023年最新更新
我们已完成了从阶段一（小样本验证）到阶段二（完整数据集）的扩展，并修复了以下关键问题：

1. **多光谱图像处理**：
   - 修复了图像维度不一致问题（从[128, 600, 128]到[3, 128, 128]）
   - 实现了从500通道中选择3个代表性通道的策略
   - 使用scikit-image库替代transforms_functional.resize函数处理多通道图像

2. **损失函数优化**：
   - 修复了FocalLoss实现中的维度不匹配问题
   - 增强了多任务学习的损失函数，更好地平衡位置分类和等级回归任务

3. **训练稳定性**：
   - 降低了学习率（0.0001）和批次大小（8/4）以提高训练稳定性
   - 优化了数据加载和预处理流程，提高训练效率

详细的修复和改进内容请参考 [README_phase2.md](README_phase2.md)。

## 快速开始

### 环境准备
```bash
pip install torch torchvision rasterio scikit-image matplotlib tqdm seaborn numpy pillow
```

### 检查图像数据
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

## 项目结构

- **model.py**: 模型定义文件
- **dataset.py**: 数据集加载和预处理
- **train.py**: 训练主脚本
- **run_training.py**: 训练启动脚本
- **utils.py**: 工具函数和损失函数
- **check_tif_images.py**: TIF图像检查工具
- **check_training_results.py**: 训练结果分析工具
- **test.py**: 模型测试脚本
- **run_testing.py**: 测试启动脚本

## 模型架构

项目实现了三种模型架构：
1. **简单CNN**（`DiseaseClassifier`）：基础卷积神经网络，双头输出
2. **ResNet**（`DiseaseResNet`）：基于残差连接的深度网络
3. **ResNet+**（`DiseaseResNetPlus`）：增强版ResNet，加入通道和空间注意力机制

## 数据集

- **图像数据**: 多光谱遥感图像（.tif格式），500通道
- **标注数据**: JSON格式，包含感染部位和感染等级信息
- **数据结构**: 9个子文件夹，对应9叶期/14叶期/19叶期的下/中/上部位置