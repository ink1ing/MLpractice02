# 玉米南方锈病UNet识别与预测系统

本项目基于UNet深度学习架构，构建玉米南方锈病的图像分割和病害等级预测模型，实现多任务学习，同时完成三个目标：

1. **病害区域分割**：检测并分割图像中的病害区域
2. **感染部位分类**：下部/中部/上部 (3分类)
3. **感染等级判断**：无/轻度/中度/重度/极重度 (对应病害等级：0/3/5/7/9)

## 项目概述

本项目使用UNet架构，一种专为医学图像分割而设计的神经网络，结合多任务学习，同时处理分割和分类任务。UNet能够精确地定位和分割病害区域，提供更详细的病害信息。

项目使用无人机获取的多光谱图像(.tif)和人工标注文件(.json)，通过深度学习模型实现玉米南方锈病的智能识别。项目分为两个阶段：

1. **阶段一（已完成）**：使用14叶片数量的小样本验证模型框架和训练流程
2. **阶段二（当前）**：扩展到9个文件夹数据集（9l/m/t、14l/m/t、19l/m/t），每类101张，共909张样本对

## 模型架构

### UNet模型

UNet是一种U形架构的全卷积网络，最初为医学图像分割设计，特点是：

1. **编码器-解码器结构**：编码器通过卷积和下采样提取特征，解码器通过上采样恢复空间分辨率
2. **跳跃连接**：将编码器中对应层的特征图与解码器中的特征图连接，保留高分辨率细节
3. **全卷积设计**：没有全连接层，可以处理任意大小的输入图像

### 多任务UNet

我们的模型扩展了标准UNet，增加了两个额外的任务头：

1. **分割头**：保留UNet原始的分割输出，用于病害区域分割
2. **位置分类头**：从编码器的瓶颈层提取特征，通过全连接层预测病害位置
3. **等级回归头**：从编码器的瓶颈层提取特征，通过全连接层预测病害等级

### UNet++变体

除了标准UNet，项目还实现了UNet++变体，特点是：

1. **嵌套式跳跃连接**：相比UNet更密集的跳跃连接结构
2. **深度监督**：可以在不同深度进行监督，提高特征提取效果
3. **更好的细节保留**：改进的特征融合方式，提高分割精度

### 注意力机制

模型集成了通道注意力和空间注意力机制，增强模型对重要特征的感知能力：

1. **通道注意力**：自适应地调整不同通道的重要性，突出关键特征
2. **空间注意力**：关注图像中的重要区域，提高分割和分类精度

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

## 项目结构

主要文件说明：

- **[核心文件] model.py**: 模型定义文件，包含UNet和UNet++架构
- **[核心文件] dataset.py**: 数据集加载和预处理，处理TIF图像和JSON标注
- **[核心文件] train.py**: 训练主脚本，包含训练循环和评估函数
- **[核心文件] run_unet.py**: UNet训练启动脚本
- **[核心文件] test_unet.py**: UNet测试脚本
- **[核心文件] use_unet_model.py**: 使用训练好的模型进行推理
- **[核心文件] utils.py**: 工具函数，包含损失函数、指标计算等
- **[参考文件/调试文件]**: 其他辅助脚本，用于环境检查、数据分析等

## 环境配置

### 依赖安装

```bash
# 创建虚拟环境
conda create -n unet_corn python=3.8
conda activate unet_corn

# 安装依赖
pip install torch torchvision torchaudio
pip install rasterio scikit-image matplotlib tqdm seaborn numpy pillow
```

### 硬件要求

- GPU: 建议使用支持CUDA的GPU (例如RTX 6000)
- 内存: 至少16GB
- 存储: 至少20GB可用空间

## 快速开始

### 检查图像数据
```bash
python check_tif_images.py --data_root ./guanceng-bit --sample_count 5 --visualize
```

### 启动训练

```bash
python run_unet.py \
  --data_root ./guanceng-bit \
  --json_root ./biaozhu_json \
  --output_dir ./unet_output \
  --model_type unet \
  --in_channels 3 \
  --out_channels 1 \
  --img_size 128 \
  --attention \
  --batch_size 8 \
  --epochs 50 \
  --lr 0.001 \
  --optimizer adam \
  --lr_scheduler plateau \
  --task_weights 0.7,0.2,0.1 \
  --loss_type focal \
  --amp \
  --num_workers 2
```

### 测试模型

```bash
python test_unet.py \
  --data_root ./guanceng-bit \
  --json_root ./biaozhu_json \
  --output_dir ./unet_test_results \
  --model_path ./unet_output/best_model.pth \
  --model_type unet \
  --in_channels 3 \
  --out_channels 1 \
  --img_size 128 \
  --attention \
  --visualize
```

### 使用模型进行预测

```bash
python use_unet_model.py \
  --image_path ./test_images/sample.tif \
  --model_path ./unet_output/best_model.pth \
  --model_type unet \
  --in_channels 3 \
  --out_channels 1 \
  --img_size 128 \
  --attention \
  --visualize
```

## 模型参数说明

### 训练参数

- **model_type**: 模型类型，'unet'或'unet++'
- **in_channels**: 输入通道数，默认为3
- **out_channels**: 分割输出通道数，默认为1
- **img_size**: 图像大小，默认为128
- **attention**: 是否使用注意力机制
- **bilinear**: 是否使用双线性插值代替转置卷积
- **batch_size**: 批次大小，建议为8-32（根据GPU内存调整）
- **epochs**: 训练轮数，默认为50
- **lr**: 学习率，默认为0.001
- **task_weights**: 任务权重，格式为"分割权重,位置分类权重,等级回归权重"
- **loss_type**: 损失函数类型，'focal'、'ce'或'dice'
- **amp**: 是否启用混合精度训练（推荐启用，加快训练速度）

### 超参数优化

通过多次实验，我们发现以下超参数组合效果最佳：

- 图像大小：128x128
- 批次大小：8（对于大多数GPU）
- 学习率：0.001
- 优化器：Adam
- 任务权重：分割0.7，位置分类0.2，等级回归0.1
- 损失函数：分割使用BCE，位置分类使用Focal Loss，等级回归使用MSE

## 性能指标

模型性能通过以下指标评估：

- **分割**：
  - IoU（交并比）
  - Dice系数

- **位置分类**：
  - 准确率
  - F1分数
  - 精确率
  - 召回率

- **等级预测**：
  - 平均绝对误差(MAE)
  - ±2误差容忍率

## 训练优化策略

1. **多光谱图像处理**：
   - 从500通道中选择3个代表性通道进行处理
   - 使用scikit-image库处理多通道图像的调整大小操作

2. **训练稳定性**：
   - 批次大小优化：从32降到8，减轻内存压力，避免OOM问题
   - 降低工作线程数：从4降到1，减少资源争用
   - 禁用混合精度训练：在某些情况下可能导致不稳定，可选择性启用

3. **损失函数优化**：
   - 使用Focal Loss解决类别不平衡问题
   - 任务权重平衡：分割0.7，位置分类0.2，等级回归0.1

## 常见问题

**Q: 如何处理不同波段数的.tif文件?**  
A: 数据集类会自动处理不同波段数的.tif文件，如果通道数小于3，会复制现有通道；如果大于3，会选择代表性通道。

**Q: 无法读取.tif文件怎么办?**  
A: 确保安装了rasterio库并有适当的GDAL支持。如果遇到读取问题，检查.tif文件的格式和完整性。

**Q: 训练过程中遇到OOM（内存溢出）怎么办?**  
A: 尝试减小批次大小、减少图像尺寸或减少模型复杂度。也可以禁用混合精度训练。

**Q: 如何调整三个任务的损失权重?**  
A: 在训练脚本中可以通过task_weights参数调整，默认为[0.7, 0.2, 0.1]，分别表示分割、位置分类和等级回归任务的权重。

**Q: 如何使用自己的数据集?**  
A: 准备好.tif图像和对应的.json标注文件，按照项目的目录结构组织，然后按需调整`CornRustDataset`类中的标签解析逻辑。

## 预测示例

使用训练好的模型进行预测的代码示例：

```python
import torch
from model import get_model
import matplotlib.pyplot as plt
import numpy as np

# 加载模型
model = get_model(
    model_type='unet',
    in_channels=3,
    out_channels=1,
    img_size=128,
    bilinear=True,
    with_attention=True
)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 预处理图像
# ... (图像加载和预处理代码)

# 使用模型进行预测
with torch.no_grad():
    segmentation, position_logits, grade_values = model(image)
    
    # 处理分割结果
    segmentation = torch.sigmoid(segmentation) > 0.5  # 二值化
    
    # 处理位置分类结果
    position_pred = torch.argmax(position_logits, dim=1)  # 获取预测类别
    position_names = ['下部', '中部', '上部']
    predicted_position = position_names[position_pred.item()]
    
    # 处理等级回归结果
    predicted_grade = grade_values.item()  # 获取预测等级值

# 可视化结果
plt.figure(figsize=(15, 5))

# 原始图像
plt.subplot(1, 3, 1)
plt.imshow(image.numpy().transpose(1, 2, 0))
plt.title("原始图像")
plt.axis('off')

# 分割结果
plt.subplot(1, 3, 2)
plt.imshow(segmentation.squeeze().numpy(), cmap='jet')
plt.title(f"分割结果\n位置: {predicted_position}, 等级: {predicted_grade:.2f}")
plt.axis('off')

# 叠加显示
plt.subplot(1, 3, 3)
plt.imshow(image.numpy().transpose(1, 2, 0))
mask = segmentation.squeeze().numpy()
overlay = np.zeros_like(image.numpy().transpose(1, 2, 0))
overlay[:, :, 0] = mask * 1.0  # 红色通道表示分割区域
plt.imshow(overlay, alpha=0.5)  # 叠加分割结果
plt.title("叠加显示")
plt.axis('off')

plt.show()
```

## 未来改进方向

1. **通道选择策略**：开发更智能的多光谱通道选择算法，提取更有代表性的光谱信息
2. **自适应任务权重**：实现在训练过程中动态调整任务权重的机制
3. **模型轻量化**：优化模型结构，减少参数量，适应低算力设备部署
4. **迁移学习**：利用预训练的视觉模型，提高模型收敛速度和泛化能力
5. **不确定性估计**：增加预测的不确定性估计，提高实际应用中的可靠性