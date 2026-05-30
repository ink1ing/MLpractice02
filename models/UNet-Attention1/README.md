# 玉米南方锈病遥感识别系统 - UNet+Attention模型

本项目基于UNet+Attention架构，实现了玉米南方锈病的多任务学习模型，同时完成三个任务：

1. **病害区域分割**：识别图像中的病害区域
2. **感染部位分类**：下部/中部/上部 (3分类)
3. **感染等级判断**：无/轻度/中度/重度/极重度 (对应病害等级：0/3/5/7/9)

## 项目概述

项目使用无人机获取的多光谱图像(.tif)和人工标注文件(.json)，通过深度学习模型实现玉米南方锈病的智能识别。项目分为两个阶段：

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

UNet+Attention模型结合了UNet的编码器-解码器架构和注意力机制，具有以下优势：

1. **分割能力**：UNet结构专为医学图像分割设计，能够精确定位病害区域
2. **注意力机制**：通道注意力和空间注意力帮助模型关注重要特征
3. **多任务学习**：同时完成分割、分类和回归任务，提高模型泛化能力
4. **跳跃连接**：保留高分辨率特征，避免信息丢失

## 文件结构

- **unet_model.py**: UNet+Attention模型定义
- **train_unet.py**: 模型训练脚本
- **run_unet_training.py**: 训练启动脚本
- **test_unet.py**: 模型测试脚本
- **dataset.py**: 数据集加载和预处理
- **utils.py**: 工具函数

## 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install torch torchvision rasterio scikit-image matplotlib tqdm seaborn numpy pillow
```

## 使用方法

### 训练模型

```bash
# 使用默认参数训练
python run_unet_training.py --amp

# 自定义参数训练
python run_unet_training.py \
    --data_root ./guanceng-bit \
    --json_root ./biaozhu_json \
    --img_size 128 \
    --in_channels 3 \
    --batch_size 16 \
    --epochs 50 \
    --lr 0.0001 \
    --optimizer adam \
    --lr_scheduler plateau \
    --seg_weight 0.4 \
    --position_weight 0.3 \
    --grade_weight 0.3 \
    --output_dir ./unet_output \
    --amp
```

### 测试模型

```bash
# 在验证集上测试
python test_unet.py \
    --model_path ./unet_output/best_model.pth \
    --output_dir ./unet_test_results

# 在整个数据集上测试
python test_unet.py \
    --model_path ./unet_output/best_model.pth \
    --output_dir ./unet_test_results \
    --test_only
```

## 参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|------|------|
| `--data_root` | ./guanceng-bit | 数据根目录路径 |
| `--json_root` | ./biaozhu_json | JSON标注根目录路径 |
| `--img_size` | 128 | 图像大小 |
| `--in_channels` | 3 | 输入图像通道数 |
| `--batch_size` | 16 | 批次大小 |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 0.0001 | 学习率 |
| `--optimizer` | adam | 优化器类型 (adam/sgd) |
| `--lr_scheduler` | plateau | 学习率调度器类型 |
| `--seg_weight` | 0.4 | 分割任务的损失权重 |
| `--position_weight` | 0.3 | 位置分类任务的损失权重 |
| `--grade_weight` | 0.3 | 等级回归任务的损失权重 |
| `--amp` | False | 是否启用混合精度训练 |
| `--output_dir` | ./unet_output | 输出目录路径 |

### 测试参数

| 参数 | 默认值 | 说明 |
|------|------|------|
| `--model_path` | (必填) | 模型权重文件路径 |
| `--test_only` | False | 仅使用测试集评估 |
| `--output_dir` | ./unet_test_results | 输出目录路径 |

## 性能指标

UNet+Attention模型的目标性能指标：

- **位置分类**:
  - 准确率 > 90%
  - F1 > 0.85
  - 召回率 > 0.85
  - 精确率 > 0.85

- **等级回归**:
  - MAE < 0.15
  - Loss < 0.2

- **分割任务**:
  - Dice系数 > 0.8

## 优化技巧

1. **混合精度训练**：启用`--amp`参数可显著加速训练过程
2. **任务权重调整**：根据需要调整`--seg_weight`、`--position_weight`和`--grade_weight`
3. **批次大小优化**：RTX 6000可使用较大批次大小(16-32)，充分利用GPU性能
4. **学习率调度**：使用ReduceLROnPlateau自动调整学习率，提高训练稳定性

## 常见问题

**Q: 如何处理不同波段数的.tif文件?**  
A: 数据集类会自动处理不同波段数的.tif文件，如果通道数小于3，会复制现有通道；如果大于3，会选择代表性通道。

**Q: 无法读取.tif文件怎么办?**  
A: 确保安装了rasterio库并有适当的GDAL支持。如果遇到读取问题，检查.tif文件的格式和完整性。

**Q: 如何调整多任务学习的任务权重?**  
A: 在训练脚本中可以通过`--seg_weight`、`--position_weight`和`--grade_weight`参数调整，默认为0.4/0.3/0.3。

**Q: 如何使用自己的数据集?**  
A: 准备好.tif图像和对应的.json标注文件，按照项目的目录结构组织，然后按需调整`CornRustDataset`类中的标签解析逻辑。

## 注意事项

1. 首次运行时，模型将生成分割掩码。由于当前数据集缺少真实的分割标注，系统使用病害等级生成临时掩码。
2. 在实际应用中，建议使用真实的分割标注数据进行训练，以获得更准确的分割结果。
3. 参数调整应谨慎，特别是任务权重的调整可能会显著影响模型性能。
4. 如需提高分辨率，请相应增加批次大小和内存使用，确保GPU资源充足。
5. 多光谱图像处理需要特别注意通道选择和维度处理，确保图像和标注文件正确匹配。