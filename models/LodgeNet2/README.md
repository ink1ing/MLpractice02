# LodgeNet：玉米锈病识别与预测模型

## 项目简介

LodgeNet 是专为玉米南方锈病设计的多任务深度学习模型，支持：
- **分割**：像素级病斑区域识别
- **位置分类**：下/中/上部感染部位
- **病害等级回归**：0-9连续值

模型基于 U-Net，集成注意力机制与 ASPP，支持混合精度训练，适配多光谱遥感数据。

## 快速上手

### 环境依赖

```bash
pip install torch torchvision rasterio scikit-image matplotlib tqdm seaborn numpy pillow scikit-learn
```

### 训练

```bash
python run_lodgenet.py --data_root ./guanceng-bit --json_root ./biaozhu_json --num_epochs 50
```
或自定义参数：
```bash
python run_lodgenet.py --batch_size 16 --learning_rate 0.0005 --output_dir ./output_custom
```

### 测试与评估

```bash
python test_lodgenet.py --model_path ./output_lodgenet/best_model.pth --data_root ./guanceng-bit --json_root ./biaozhu_json --visualize
```

## 推荐配置

- batch_size: 32（视GPU内存调整）
- num_epochs: 50
- learning_rate: 0.001
- img_size: 256
- 优化器: AdamW
- 调度器: CosineAnnealingLR
- 混合精度: 自动启用

## 主要指标

- 位置分类准确率 > 90%
- F1/召回/精确率 > 0.85
- 病害等级 MAE < 0.15
- 总损失 < 0.2

## 目录结构

```
LodgeNet/
├── lodgenet_model.py      # 模型定义
├── lodgenet_train.py      # 训练脚本
├── run_lodgenet.py        # 启动脚本
├── test_lodgenet.py       # 测试评估
├── dataset.py             # 数据加载
├── utils.py               # 工具函数
├── guanceng-bit/          # 图像数据
├── biaozhu_json/          # 标注数据
├── output_lodgenet/       # 训练输出
├── logs/                  # 日志
└── test_results/          # 测试结果
```

## 常见问题

- 内存不足：减小 batch_size 或优化数据加载
- 收敛慢：调整学习率、任务权重
